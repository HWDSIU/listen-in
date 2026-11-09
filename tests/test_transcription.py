from __future__ import annotations

import sys
import types

import pytest

from listen_in import transcription
from listen_in.exceptions import InvalidInputError, LanguageMismatchError, OptionalDependencyError
from listen_in.transcription import transcribe_audio

from .conftest import write_wav


class FakeMel:
    def to(self, device):
        return self


class FakeModel:
    dims = types.SimpleNamespace(n_mels=80)

    def __init__(self, device: str, lang_probs: dict):
        self.device = device
        self.calls: list[dict] = []
        self.lang_probs = lang_probs  # shared, so tests can change detection results
        self.detect_calls = 0

    def detect_language(self, mel):
        self.detect_calls += 1
        if isinstance(self.lang_probs, Exception):
            raise self.lang_probs
        return None, dict(self.lang_probs)

    def transcribe(self, audio, **kwargs):
        self.calls.append({"n_samples": len(audio), **kwargs})
        return {
            "text": " Bonjour à tous. ",
            "language": kwargs.get("language"),
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.5, "text": " Bonjour à tous.",
                 "avg_logprob": -0.3, "no_speech_prob": 0.01, "compression_ratio": 1.2},
            ],
        }


@pytest.fixture
def fake_whisper(monkeypatch):
    pytest.importorskip("numpy")
    loads: list[tuple] = []
    models: list[FakeModel] = []
    # Neutral default: detection does not contradict either test language.
    state = types.SimpleNamespace(lang_probs={"en": 0.5, "fr": 0.5})

    def load_model(name, device="cpu", download_root=None):
        loads.append((name, device, download_root))
        m = FakeModel(device, state.lang_probs)
        models.append(m)
        return m

    def load_audio(path):  # pragma: no cover - should not be used for 16k mono WAV
        raise AssertionError("load_audio (ffmpeg) should not be needed for 16 kHz mono WAV")

    whisper = types.ModuleType("whisper")
    whisper.load_model = load_model
    whisper.load_audio = load_audio
    whisper.tokenizer = types.SimpleNamespace(LANGUAGES={"en": "english", "fr": "french", "de": "german"})
    whisper.pad_or_trim = lambda a, length=480_000: a[:length]
    whisper.log_mel_spectrogram = lambda audio, n_mels=80, **kw: FakeMel()
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "whisper", whisper)
    monkeypatch.setitem(sys.modules, "torch", torch)
    def set_probs(probs):
        state.lang_probs = probs
        for m in models:
            m.lang_probs = probs

    return types.SimpleNamespace(module=whisper, torch=torch, loads=loads, models=models,
                                 set_probs=set_probs)


def test_missing_whisper_raises_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "whisper", None)
    wav = write_wav(tmp_path / "a.wav", 1)
    with pytest.raises(OptionalDependencyError) as exc:
        transcribe_audio(wav, language="fr")
    assert 'pip install "listen-in[whisper]"' in str(exc.value)
    assert isinstance(exc.value, ImportError)


@pytest.mark.parametrize("lang", ["fr", "en"])
def test_language_hint_and_transcribe_task(fake_whisper, tmp_path, lang):
    wav = write_wav(tmp_path / "a.wav", 2)
    t = transcribe_audio(wav, language=lang, model="small")
    call = fake_whisper.models[0].calls[0]
    assert call["language"] == lang
    assert call["task"] == "transcribe"
    assert call["fp16"] is False  # CPU
    assert call["n_samples"] == 32_000
    assert t.language == lang
    assert t.text == "Bonjour à tous."
    assert t.segments[0].end == 2.5
    assert t.task == "transcribe"
    assert t.duration_seconds == pytest.approx(2.0)
    assert t.device == "cpu"


def test_model_loaded_once_and_cached(fake_whisper, tmp_path):
    wav = write_wav(tmp_path / "a.wav", 1)
    transcribe_audio(wav, language="fr")
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
        transcribe_audio(wav, language="fr", task="translate")
    with pytest.raises(InvalidInputError):
        transcribe_audio(wav, language="xx")


def test_missing_audio_file(fake_whisper, tmp_path):
    with pytest.raises(InvalidInputError):
        transcribe_audio(tmp_path / "missing.wav", language="fr")


def test_ca_bundle_only_set_when_python_has_none(monkeypatch, tmp_path):
    pytest.importorskip("certifi")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    missing = types.SimpleNamespace(openssl_cafile=str(tmp_path / "no-such-cert.pem"))
    monkeypatch.setattr(transcription.ssl, "get_default_verify_paths", lambda: missing)
    with transcription._ca_bundle_for_download():
        assert transcription.os.environ.get("SSL_CERT_FILE")
    assert "SSL_CERT_FILE" not in transcription.os.environ


# ------------------------------------------------------------ language checking
GERMAN_AUDIO = {"de": 0.98, "en": 0.01, "fr": 0.01}


def test_wrong_hint_is_replaced_by_detected_language(fake_whisper, tmp_path, caplog):
    fake_whisper.set_probs(GERMAN_AUDIO)
    wav = write_wav(tmp_path / "a.wav", 45)
    t = transcribe_audio(wav, language="en")
    assert fake_whisper.models[0].calls[0]["language"] == "de"  # original language kept
    assert fake_whisper.models[0].calls[0]["task"] == "transcribe"
    assert (t.language, t.language_hint, t.detected_language) == ("de", "en", "de")
    assert t.language_mismatch is True
    assert t.detected_language_probability == 0.98
    assert "does not match" in caplog.text


def test_use_hint_policy_keeps_hint_but_flags_it(fake_whisper, tmp_path):
    fake_whisper.set_probs(GERMAN_AUDIO)
    t = transcribe_audio(write_wav(tmp_path / "a.wav", 5), language="en", on_language_mismatch="use_hint")
    assert fake_whisper.models[0].calls[0]["language"] == "en"
    assert t.language == "en" and t.language_mismatch is True


def test_error_policy_raises_before_transcribing(fake_whisper, tmp_path):
    fake_whisper.set_probs(GERMAN_AUDIO)
    with pytest.raises(LanguageMismatchError, match="detected 'de'"):
        transcribe_audio(write_wav(tmp_path / "a.wav", 5), language="en", on_language_mismatch="error")
    assert fake_whisper.models[0].calls == []


@pytest.mark.parametrize("probs", [
    {"de": 0.45, "fr": 0.45, "en": 0.05, "es": 0.05},  # detection not confident
    {"de": 0.80, "en": 0.15, "fr": 0.05},  # hint still has real weight (e.g. code-switching)
])
def test_uncertain_detection_keeps_the_hint(fake_whisper, tmp_path, probs):
    fake_whisper.set_probs(probs)
    t = transcribe_audio(write_wav(tmp_path / "a.wav", 5), language="en")
    assert fake_whisper.models[0].calls[0]["language"] == "en"
    assert t.language == "en" and t.language_mismatch is False


def test_no_hint_uses_detected_language(fake_whisper, tmp_path):
    fake_whisper.set_probs({"fr": 0.9, "en": 0.1})
    t = transcribe_audio(write_wav(tmp_path / "a.wav", 5), language=None)
    assert fake_whisper.models[0].calls[0]["language"] == "fr"
    assert (t.language, t.language_hint, t.language_mismatch) == ("fr", None, False)


@pytest.mark.parametrize(("seconds", "windows"), [(10, 1), (30, 1), (100, 5), (300, 5)])
def test_detection_samples_several_windows(fake_whisper, tmp_path, seconds, windows):
    transcribe_audio(write_wav(tmp_path / "a.wav", seconds), language="en")
    assert fake_whisper.models[0].detect_calls == windows


def test_detection_failure_falls_back_to_hint(fake_whisper, tmp_path):
    fake_whisper.set_probs(RuntimeError("no language tokens"))
    t = transcribe_audio(write_wav(tmp_path / "a.wav", 5), language="fr")
    assert fake_whisper.models[0].calls[0]["language"] == "fr"
    assert t.detected_language is None and t.language_mismatch is False


def test_invalid_mismatch_policy(fake_whisper, tmp_path):
    with pytest.raises(InvalidInputError):
        transcribe_audio(write_wav(tmp_path / "a.wav", 1), language="fr", on_language_mismatch="ignore")
