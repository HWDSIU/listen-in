"""Local speech-to-text with openai-whisper (optional extra ``listen-in[whisper]``).

Whisper is imported lazily so the base package imports without it. Loaded models
are cached per process, keyed by (model, device, download_root).
"""

from __future__ import annotations

import contextlib
import logging
import os
import ssl
import threading
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .exceptions import InvalidInputError, OptionalDependencyError, TranscriptionError
from .models import AudioCapture, Transcript, TranscriptSegment
from .utils import Stopwatch

log = logging.getLogger(__name__)

_MODELS: dict[tuple[str, str, str | None], Any] = {}
_MODELS_LOCK = threading.Lock()


def _import_whisper() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import torch
        import whisper
    except ImportError as exc:
        missing = getattr(exc, "name", None) or "whisper"
        package = {"whisper": "openai-whisper"}.get(missing, missing)
        raise OptionalDependencyError(package, "whisper", "Transcription") from exc
    return whisper, torch, np


def resolve_device(device: str, torch: Any) -> str:
    """``auto`` -> ``cuda`` when available, else ``cpu``.

    Apple MPS is not selected automatically: openai-whisper relies on sparse tensor
    ops that MPS does not support. Pass ``device="mps"`` explicitly to experiment.
    """
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise TranscriptionError(f"device={device!r} requested but CUDA is not available.")
    return device


@contextlib.contextmanager
def _ca_bundle_for_download() -> Iterator[None]:
    """Use certifi's CA bundle while Whisper downloads weights, if Python has none.

    python.org builds on macOS ship without an OpenSSL CA file until "Install
    Certificates.command" is run, which makes urllib (used by Whisper) fail with
    CERTIFICATE_VERIFY_FAILED. We only touch SSL_CERT_FILE when it is unset and the
    interpreter's default CA file does not exist, and restore it afterwards.
    """
    if os.environ.get("SSL_CERT_FILE") or Path(ssl.get_default_verify_paths().openssl_cafile).exists():
        yield
        return
    try:
        import certifi
    except ImportError:
        yield
        return
    os.environ["SSL_CERT_FILE"] = certifi.where()
    try:
        yield
    finally:
        os.environ.pop("SSL_CERT_FILE", None)


def load_whisper_model(model: str = "small", device: str = "auto", download_root: str | None = None) -> Any:
    """Load (or reuse) a Whisper model. Returns the model instance."""
    whisper, torch, _ = _import_whisper()
    dev = resolve_device(device, torch)
    key = (model, dev, download_root)
    with _MODELS_LOCK:
        if key not in _MODELS:
            log.info("Loading Whisper model %r on %s (first use downloads weights)", model, dev)
            try:
                with _ca_bundle_for_download():
                    _MODELS[key] = whisper.load_model(model, device=dev, download_root=download_root)
            except Exception as exc:
                raise TranscriptionError(f"Could not load Whisper model {model!r}: {exc}") from exc
        return _MODELS[key]


def clear_model_cache() -> None:
    with _MODELS_LOCK:
        _MODELS.clear()


def _load_audio(path: Path, whisper: Any, np: Any) -> tuple[Any, float | None]:
    """Read 16 kHz mono PCM WAV directly (no ffmpeg needed); else defer to Whisper."""
    try:
        with wave.open(str(path), "rb") as w:
            if w.getframerate() == 16_000 and w.getnchannels() == 1 and w.getsampwidth() == 2:
                frames = w.readframes(w.getnframes())
                audio = np.frombuffer(frames, np.int16).astype(np.float32) / 32768.0
                return audio, len(audio) / 16_000.0
    except (wave.Error, EOFError):
        pass
    # Other formats: Whisper decodes via the ffmpeg executable on PATH.
    return whisper.load_audio(str(path)), None


def transcribe_audio(
    audio: str | Path | AudioCapture,
    language: str | None = None,
    model: str = "small",
    device: str = "auto",
    *,
    download_root: str | None = None,
    initial_prompt: str | None = None,
    condition_on_previous_text: bool = False,
    **whisper_options: Any,
) -> Transcript:
    """Transcribe an audio file in its original language (never translates).

    ``language`` is an ISO 639-1 hint such as ``"fr"`` or ``"en"``; ``None`` lets
    Whisper detect it. ``condition_on_previous_text`` defaults to False because radio
    audio mixes music and speech and conditioning makes repetition loops likelier.
    """
    path = Path(audio.path if isinstance(audio, AudioCapture) else audio)
    if not path.is_file():
        raise InvalidInputError(f"Audio file not found: {path}")
    if "task" in whisper_options:
        raise InvalidInputError("listen-in always uses task='transcribe'; translation is done by the LLM.")
    whisper, _torch, np = _import_whisper()
    if language is not None:
        language = language.strip().lower()
        if language not in whisper.tokenizer.LANGUAGES:
            raise InvalidInputError(f"Unsupported Whisper language code: {language!r}")

    sw = Stopwatch()
    m = load_whisper_model(model, device, download_root)
    dev = str(getattr(m, "device", device))
    samples, seconds = _load_audio(path, whisper, np)
    try:
        result = m.transcribe(
            samples,
            language=language,
            task="transcribe",
            fp16=dev.startswith("cuda"),
            initial_prompt=initial_prompt,
            condition_on_previous_text=condition_on_previous_text,
            verbose=None,
            **whisper_options,
        )
    except Exception as exc:
        raise TranscriptionError(f"Whisper failed on {path.name}: {exc}") from exc

    segments = [
        TranscriptSegment(
            id=int(s.get("id", i)),
            start=round(float(s["start"]), 2),
            end=round(float(s["end"]), 2),
            text=str(s["text"]).strip(),
            avg_logprob=s.get("avg_logprob"),
            no_speech_prob=s.get("no_speech_prob"),
            compression_ratio=s.get("compression_ratio"),
        )
        for i, s in enumerate(result.get("segments", []))
    ]
    return Transcript(
        text=str(result.get("text", "")).strip(),
        language=language or str(result.get("language") or ""),
        segments=segments,
        model=model,
        device=dev,
        audio_path=path,
        duration_seconds=round(seconds, 3) if seconds is not None else None,
        processing_seconds=sw.seconds,
    )
