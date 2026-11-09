from __future__ import annotations

import sys
import types

import pytest

from listen_in import transcription
from listen_in.exceptions import InvalidInputError, OptionalDependencyError
from listen_in.transcription import transcribe_audio

from .conftest import write_wav


class FakeModel:
    def __init__(self, device: str):
        self.device = device
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"n_samples": len(audio), **kwargs})
        return {
            "text": " Habari za asubuhi. ",
            "language": kwargs.get("language"),
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.5, "text": " Habari za asubuhi.",
                 "avg_logprob": -0.3, "no_speech_prob": 0.01, "compression_ratio": 1.2},
            ],
        }


@pytest.fixture
def fake_whisper(monkeypatch):
    pytest.importorskip("numpy")
    loads: list[tuple] = []
    models: list[FakeModel] = []

    def load_model(name, device="cpu", download_root=None):
        loads.append((name, device, download_root))
        m = FakeModel(device)
        models.append(m)
        return m

    def load_audio(path):  # pragma: no cover - should not be used for 16k mono WAV
        raise AssertionError("load_audio (ffmpeg) should not be needed for 16 kHz mono WAV")

    whisper = types.ModuleType("whisper")
    whisper.load_model = load_model
    whisper.load_audio = load_audio
    whisper.tokenizer = types.SimpleNamespace(LANGUAGES={"en": "english", "sw": "swahili"})
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "whisper", whisper)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return types.SimpleNamespace(module=whisper, torch=torch, loads=loads, models=models)


def test_missing_whisper_raises_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "whisper", None)
    wav = write_wav(tmp_path / "a.wav", 1)
    with pytest.raises(OptionalDependencyError) as exc:
        transcribe_audio(wav, language="sw")
    assert 'pip install "listen-in[whisper]"' in str(exc.value)
    assert isinstance(exc.value, ImportError)


@pytest.mark.parametrize("lang", ["sw", "en"])
def test_language_hint_and_transcribe_task(fake_whisper, tmp_path, lang):
    wav = write_wav(tmp_path / "a.wav", 2)
    t = transcribe_audio(wav, language=lang, model="small")
    call = fake_whisper.models[0].calls[0]
    assert call["language"] == lang
    assert call["task"] == "transcribe"
    assert call["fp16"] is False  # CPU
    assert call["n_samples"] == 32_000
    assert t.language == lang
    assert t.text == "Habari za asubuhi."
    assert t.segments[0].end == 2.5
    assert t.task == "transcribe"
    assert t.duration_seconds == pytest.approx(2.0)
    assert t.device == "cpu"


def test_model_loaded_once_and_cached(fake_whisper, tmp_path):
    wav = write_wav(tmp_path / "a.wav", 1)
    transcribe_audio(wav, language="sw")
    transcribe_audio(wav, language="en")
    assert fake_whisper.loads == [("small", "cpu", None)]


def test_cuda_selected_with_fp16_when_available(fake_whisper, tmp_path):
    fake_whisper.torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    wav = write_wav(tmp_path / "a.wav", 1)
    transcribe_audio(wav, language="en", device="auto")
    assert fake_whisper.loads[0][1] == "cuda"
    assert fake_whisper.models[0].calls[0]["fp16"] is True


def test_rejects_translation_and_unknown_language(fake_whisper, tmp_path):
    wav = write_wav(tmp_path / "a.wav", 1)
    with pytest.raises(InvalidInputError):
        transcribe_audio(wav, language="sw", task="translate")
    with pytest.raises(InvalidInputError):
        transcribe_audio(wav, language="xx")


def test_missing_audio_file(fake_whisper, tmp_path):
    with pytest.raises(InvalidInputError):
        transcribe_audio(tmp_path / "missing.wav", language="sw")


def test_ca_bundle_only_set_when_python_has_none(monkeypatch, tmp_path):
    pytest.importorskip("certifi")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    missing = types.SimpleNamespace(openssl_cafile=str(tmp_path / "no-such-cert.pem"))
    monkeypatch.setattr(transcription.ssl, "get_default_verify_paths", lambda: missing)
    with transcription._ca_bundle_for_download():
        assert transcription.os.environ.get("SSL_CERT_FILE")
    assert "SSL_CERT_FILE" not in transcription.os.environ
