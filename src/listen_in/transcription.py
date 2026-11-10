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
from typing import Any, Literal

from .exceptions import (
    InvalidInputError,
    LanguageMismatchError,
    OptionalDependencyError,
    TranscriptionError,
)
from .models import AudioCapture, Transcript, TranscriptSegment
from .utils import Stopwatch

log = logging.getLogger(__name__)

_MODELS: dict[tuple[str, str, str | None], Any] = {}

LanguageMismatchPolicy = Literal["use_detected", "use_hint", "error"]
#: Whisper's language ID looks at 30 s at a time; we average several windows so a
#: jingle or an advert in another language does not decide on its own.
DETECTION_WINDOWS = 5
#: A hint is overridden only when detection is confident AND gives the hint almost no weight.
MISMATCH_MIN_DETECTED_PROBABILITY = 0.5
MISMATCH_MAX_HINT_PROBABILITY = 0.1
#: Languages that sound the same (or nearly) and differ mostly in script or spelling, so
#: Whisper's detector cannot reliably tell them apart from audio. A hint in one of these
#: groups is judged by the whole group's probability and is never swapped for a sibling.
RELATED_LANGUAGES: tuple[frozenset[str], ...] = (
    frozenset({"hi", "ur"}),          # Hindi / Urdu
    frozenset({"sr", "hr", "bs"}),    # Serbian / Croatian / Bosnian
    frozenset({"ms", "id"}),          # Malay / Indonesian
    frozenset({"no", "nn"}),          # Norwegian Bokmål / Nynorsk
    frozenset({"fa", "tg"}),          # Persian / Tajik
)
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


def language_group(code: str) -> frozenset[str]:
    """The set of languages Whisper cannot reliably separate from ``code`` (at least itself)."""
    for group in RELATED_LANGUAGES:
        if code in group:
            return group
    return frozenset({code})


def check_transcription_ready(
    language: str | None,
    model: str = "small",
    device: str = "auto",
    on_language_mismatch: str = "use_detected",
) -> str | None:
    """Check everything transcription needs, without loading or downloading the model.

    Raises :class:`OptionalDependencyError` if Whisper is not installed,
    :class:`InvalidInputError` for an unknown language code, model name or mismatch
    policy, and :class:`TranscriptionError` if CUDA is requested but unavailable.
    Returns the normalised language code. :func:`listen_in.process_station` calls this
    before recording, so set-up mistakes surface immediately.
    """
    whisper, torch, _ = _import_whisper()
    if language is not None:
        language = language.strip().lower()
        if language not in whisper.tokenizer.LANGUAGES:
            raise InvalidInputError(f"Unsupported Whisper language code: {language!r}")
    if model not in whisper.available_models() and not Path(model).expanduser().is_file():
        raise InvalidInputError(
            f"Unknown Whisper model {model!r}. Use one of {', '.join(whisper.available_models())} "
            "or a path to a checkpoint file."
        )
    if on_language_mismatch not in ("use_detected", "use_hint", "error"):
        raise InvalidInputError(f"on_language_mismatch must be use_detected, use_hint or error, "
                                f"not {on_language_mismatch!r}")
    resolve_device(device, torch)
    return language


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


def release_whisper_model(model: str) -> bool:
    """Drop a cached Whisper model (any device) to free memory. Returns True if one was cached."""
    with _MODELS_LOCK:
        keys = [k for k in _MODELS if k[0] == model]
        for k in keys:
            del _MODELS[k]
    if keys:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover - torch comes with whisper
            pass
        log.info("Released Whisper model %r", model)
    return bool(keys)


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


def _language_probabilities(m: Any, samples: Any, whisper: Any,
                            windows: int = DETECTION_WINDOWS) -> dict[str, float] | None:
    """Average Whisper language-ID probabilities over up to ``windows`` 30 s windows."""
    n = 30 * 16_000
    total = len(samples)
    if total == 0:
        return None
    if total <= n:
        starts = [0]
    else:
        k = max(1, windows)
        starts = sorted({round(i * (total - n) / max(1, k - 1)) for i in range(k)})
    sums: dict[str, float] = {}
    for start in starts:
        chunk = whisper.pad_or_trim(samples[start:start + n])
        mel = whisper.log_mel_spectrogram(chunk, n_mels=m.dims.n_mels).to(m.device)
        _, probs = m.detect_language(mel)
        probs = probs if isinstance(probs, dict) else probs[0]
        for lang, p in probs.items():
            sums[lang] = sums.get(lang, 0.0) + float(p)
    return {lang: p / len(starts) for lang, p in sums.items()}


def transcribe_audio(
    audio: str | Path | AudioCapture,
    language: str | None = None,
    model: str = "small",
    device: str = "auto",
    *,
    on_language_mismatch: LanguageMismatchPolicy = "use_detected",
    download_root: str | None = None,
    initial_prompt: str | None = None,
    condition_on_previous_text: bool = False,
    **whisper_options: Any,
) -> Transcript:
    """Transcribe an audio file in its original language (never translates).

    ``language`` is an ISO 639-1 hint such as ``"fr"`` or ``"en"``; ``None`` uses the
    detected language. Whisper treats its language setting as an instruction, not a
    hint: given the wrong language it quietly *translates* into it. So the hint is
    checked first with Whisper's language ID over several windows of the audio. If
    detection is confident (>= 50 %) and gives the hint almost no weight (< 10 %; for
    closely related languages such as Urdu/Hindi or Serbian/Croatian the whole group counts),
    ``on_language_mismatch`` decides: ``"use_detected"`` (default; keeps the original
    language and logs a warning), ``"use_hint"`` or ``"error"``. The outcome is
    recorded on the transcript (``language_hint``, ``detected_language``,
    ``language_mismatch``).

    ``model`` is any openai-whisper model name: ``"small"`` (default) suits English,
    French and most widely spoken languages; ``"turbo"`` is recommended for less widely
    spoken languages such as Swahili (best on GPU or 16 GB+ RAM).
    ``condition_on_previous_text`` defaults to False because radio audio mixes music
    and speech and conditioning makes repetition loops likelier.
    """
    if isinstance(audio, AudioCapture):
        if audio.path is None:
            raise InvalidInputError("This recording was deleted (keep_audio=False); nothing to transcribe.")
        audio = audio.path
    path = Path(audio)
    if not path.is_file():
        raise InvalidInputError(f"Audio file not found: {path}")
    if "task" in whisper_options:
        raise InvalidInputError("listen-in always uses task='transcribe'; translation is done by the LLM.")
    language = check_transcription_ready(language, model, device, on_language_mismatch)
    whisper, _torch, np = _import_whisper()

    sw = Stopwatch()
    m = load_whisper_model(model, device, download_root)
    dev = str(getattr(m, "device", device))
    samples, seconds = _load_audio(path, whisper, np)

    # Check the hint against what is actually spoken (see docstring).
    detected: str | None = None
    detected_p: float | None = None
    mismatch = False
    use_language = language
    try:
        probs = _language_probabilities(m, samples, whisper)
    except Exception as exc:  # noqa: BLE001 - detection is advisory; never block transcription
        log.warning("Language detection failed (%s); using language=%r as given.", exc, language)
        probs = None
    if probs:
        detected, detected_p = max(probs.items(), key=lambda kv: kv[1])
        if language is None:
            use_language = detected
        elif detected != language:
            # Closely related languages (e.g. Urdu/Hindi) count as one for this check.
            hint_p = sum(probs.get(code, 0.0) for code in language_group(language))
            if detected_p >= MISMATCH_MIN_DETECTED_PROBABILITY and hint_p < MISMATCH_MAX_HINT_PROBABILITY:
                mismatch = True
                msg = (f"Language hint {language!r} does not match the audio: Whisper detected "
                       f"{detected!r} ({detected_p:.0%}) and gives {language!r} {hint_p:.0%}.")
                if on_language_mismatch == "error":
                    raise LanguageMismatchError(msg)
                if on_language_mismatch == "use_detected":
                    log.warning("%s Transcribing in %r to keep the original language.", msg, detected)
                    use_language = detected
                else:
                    log.warning("%s Using the hint as requested; the text may be a translation.", msg)

    try:
        result = m.transcribe(
            samples,
            language=use_language,
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
        language=use_language or str(result.get("language") or ""),
        language_hint=language,
        detected_language=detected,
        detected_language_probability=round(detected_p, 4) if detected_p is not None else None,
        language_mismatch=mismatch,
        segments=segments,
        model=model,
        device=dev,
        audio_path=path,
        duration_seconds=round(seconds, 3) if seconds is not None else None,
        processing_seconds=sw.seconds,
    )
