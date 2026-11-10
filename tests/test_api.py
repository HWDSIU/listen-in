from __future__ import annotations

import json
import subprocess
import sys
from datetime import timedelta

import pytest

from listen_in import api
from listen_in.exceptions import (
    AnalysisError,
    ConfigurationError,
    InvalidInputError,
    StreamUnavailableError,
    TranscriptionError,
)
from listen_in.models import (
    AnalysisResult,
    AudioCapture,
    NewsAnalysis,
    Station,
    Transcript,
    TranslatedSegment,
    Translation,
)

from .conftest import FAKE_KEY, FIXED_NOW, FRENCH_HISTORY, FRENCH_UUID, write_wav

STATION = Station(
    stream_url="http://icecast.radiofrance.fr/franceinfo-hifi.aac",
    name="France Info", source="radio_browser", uuid=FRENCH_UUID,
)
RUN_ID = "france-info_31074f8a_20261109T100000Z"


@pytest.fixture
def pipeline(monkeypatch):
    """Patch every external stage; tests flip individual stages to fail."""
    state = {"capture_error": None, "transcribe_error": None, "analyse_error": None,
             "translate_error": None, "calls": []}

    def resolve(station, **kw):
        state["calls"].append("resolve")
        return STATION

    def capture(st, duration, out_dir, *, options, overwrite, filename):
        state["calls"].append("capture")
        if state["capture_error"]:
            raise state["capture_error"]
        path = write_wav(out_dir / filename, 1)
        return AudioCapture(path=path, station=st, started_at=FIXED_NOW,
                            finished_at=FIXED_NOW + timedelta(seconds=duration),
                            requested_duration_seconds=duration, duration_seconds=duration,
                            sample_rate=16_000, channels=1, size_bytes=path.stat().st_size,
                            ffmpeg_path="/usr/bin/ffmpeg")

    def transcribe(path, *, language, model, device, initial_prompt, on_language_mismatch):
        state["calls"].append(("transcribe", language, model))
        state["on_language_mismatch"] = on_language_mismatch
        if state["transcribe_error"]:
            raise state["transcribe_error"]
        return Transcript(text="Il est treize heures.", language=language, model=model,
                          device="cpu", audio_path=path)

    def analyse(transcript, *, config, station_name, llm_provider=None):
        state["calls"].append(("analyse", config.model, station_name))
        state["llm_config"] = config
        if state["analyse_error"]:
            raise state["analyse_error"]
        return AnalysisResult(
            analysis=NewsAnalysis(news_detected=True, broadcast_language="fr",
                                  translation_used=True, summary_en="Bulletin.", stories=[]),
            provider=config.provider, model=config.model,
        )

    monkeypatch.setattr(api, "resolve_station", resolve)
    monkeypatch.setattr(api, "capture_stream", capture)
    monkeypatch.setattr(api, "transcribe_audio", transcribe)
    def translate(transcript, *, config, llm_provider=None):
        state["calls"].append(("translate", transcript.language))
        if state["translate_error"]:
            raise state["translate_error"]
        seg = TranslatedSegment(id=0, start=0.0, end=2.0, text=transcript.text, text_en="It is one o'clock.")
        return Translation(source_language=transcript.language, segments=[seg], text_en=seg.text_en,
                           provider=config.provider, model=config.model)

    monkeypatch.setattr(api, "analyse_transcript", analyse)
    monkeypatch.setattr(api, "translate_transcript", translate)
    # Pre-recording checks: Whisper itself is not installed in every test environment.
    monkeypatch.setattr(api, "check_transcription_ready",
                        lambda language, model, device, policy: language.strip().lower() if language else None)
    monkeypatch.setattr(api, "get_provider", lambda cfg, key: object())
    monkeypatch.setattr(api, "utc_now", lambda: FIXED_NOW)
    return state


def run(tmp_path, **kw):
    kw.setdefault("llm", None)
    return api.process_station(FRENCH_HISTORY, "fr", 60, tmp_path / "out",
                               staging_dir=tmp_path / "staging", **kw)


def test_transcription_only_when_llm_none(tmp_path, pipeline):
    r = run(tmp_path)
    assert r.ok
    assert r.statuses == {"resolve": "ok", "capture": "ok", "transcribe": "ok",
                          "translate": "skipped", "analyse": "skipped", "persist": "ok"}
    assert r.stages["translate"].detail == "no LLM configured"
    assert r.run_id == RUN_ID
    out = tmp_path / "out"
    assert sorted(p.name for p in out.iterdir()) == ["analysis", "audio", "manifests", "transcripts"]
    assert (out / "audio" / f"{RUN_ID}.wav").exists()
    assert (out / "transcripts" / f"{RUN_ID}.json").exists()
    assert (out / "transcripts" / f"{RUN_ID}.txt").read_text().startswith("Il est")
    assert list((out / "analysis").iterdir()) == []
    manifest = json.loads((out / "manifests" / f"{RUN_ID}.json").read_text())
    assert manifest["stages"]["analyse"]["status"] == "skipped"
    assert manifest["llm"] is None
    assert not any(p.suffix == ".wav" for p in (tmp_path / "staging").rglob("*"))  # staging cleaned


def test_full_run_with_llm_never_writes_key(tmp_path, pipeline):
    r = run(tmp_path, llm={"provider": "anthropic", "model": "claude-haiku-5-5", "api_key": FAKE_KEY})
    assert r.ok and r.statuses["analyse"] == "ok"
    assert ("analyse", "claude-haiku-5-5", "France Info") in pipeline["calls"]
    assert pipeline["llm_config"].api_key.get_secret_value() == FAKE_KEY  # passed through
    assert r.analysis_path.exists()
    for path in (tmp_path / "out").rglob("*.json"):
        assert FAKE_KEY not in path.read_text(), path
    assert FAKE_KEY not in r.model_dump_json()
    assert FAKE_KEY not in repr(r)
    assert "api_key" not in r.llm


def test_api_key_read_from_env(tmp_path, pipeline, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", FAKE_KEY)
    r = run(tmp_path, llm={"model": "claude-haiku-5-5"})
    assert r.statuses["analyse"] == "ok"
    assert FAKE_KEY not in r.manifest_path.read_text()


def test_analysis_failure_preserves_transcript(tmp_path, pipeline):
    pipeline["analyse_error"] = AnalysisError(f"bad gateway {FAKE_KEY}")
    r = run(tmp_path, llm={"api_key": FAKE_KEY})
    assert not r.ok
    assert r.statuses["analyse"] == "failed"
    assert r.transcript_path.exists() and r.audio_path.exists()
    assert FAKE_KEY not in r.stages["analyse"].error
    assert "[REDACTED]" in r.stages["analyse"].error
    assert r.manifest_path.exists()


def test_capture_failure_skips_later_stages(tmp_path, pipeline):
    pipeline["capture_error"] = StreamUnavailableError("404 Not Found")
    r = run(tmp_path, llm={"api_key": FAKE_KEY})
    assert r.statuses == {"resolve": "ok", "capture": "failed", "transcribe": "skipped",
                          "translate": "skipped", "analyse": "skipped", "persist": "ok"}
    assert "404" in r.stages["capture"].error
    assert r.manifest_path.exists()


def test_transcription_failure_keeps_audio(tmp_path, pipeline):
    pipeline["transcribe_error"] = TranscriptionError("model load failed")
    r = run(tmp_path)
    assert r.statuses["transcribe"] == "failed"
    assert r.audio_path.exists()


def test_missing_key_fails_fast_before_capture(tmp_path, pipeline):
    with pytest.raises(ConfigurationError):
        run(tmp_path, llm={"model": "claude-haiku-5-5"})
    assert pipeline["calls"] == []


def test_invalid_arguments_fail_fast(tmp_path, pipeline):
    with pytest.raises(InvalidInputError):
        api.process_station(FRENCH_HISTORY, "fr", 0, tmp_path / "out")
    with pytest.raises(InvalidInputError):
        api.process_station(FRENCH_HISTORY, "fr", 60, "s3://bucket/path")
    assert pipeline["calls"] == []


def test_persist_time_recorded_in_manifest(tmp_path, pipeline):
    r = run(tmp_path)
    assert r.stages["persist"].seconds is not None
    manifest = json.loads(r.manifest_path.read_text())
    assert manifest["stages"]["persist"]["seconds"] == r.stages["persist"].seconds
    assert manifest["stages"]["persist"]["status"] == "ok"


def test_language_hint_passed_to_whisper(tmp_path, pipeline):
    run(tmp_path)
    assert ("transcribe", "fr", "small") in pipeline["calls"]
    assert pipeline["on_language_mismatch"] == "use_detected"


def test_language_mismatch_policy_passed_through(tmp_path, pipeline):
    run(tmp_path, on_language_mismatch="error")
    assert pipeline["on_language_mismatch"] == "error"


def test_keep_audio_false_removes_wav(tmp_path, pipeline):
    r = run(tmp_path, keep_audio=False)
    assert r.audio_path is None
    assert list((tmp_path / "out" / "audio").iterdir()) == []
    assert r.transcript_path.exists()


def test_existing_outputs_not_overwritten(tmp_path, pipeline):
    run(tmp_path)
    r2 = run(tmp_path)  # same fixed clock -> same run id -> files exist
    assert r2.statuses["capture"] == "failed"
    assert "already exists" in r2.stages["capture"].error


def test_base_import_does_not_load_optional_dependencies():
    code = ("import sys, listen_in; "
            "print(sorted(m for m in ('whisper','torch','anthropic','openai') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


# ------------------------------------------------------------------ translation
LLM = {"provider": "anthropic", "model": "claude-haiku-5-5", "api_key": FAKE_KEY}


def test_non_english_broadcast_is_translated_and_stored_side_by_side(tmp_path, pipeline):
    r = run(tmp_path, llm=LLM)
    assert r.ok and r.statuses["translate"] == "ok"
    assert ("translate", "fr") in pipeline["calls"]
    assert r.translation_path.name == f"{RUN_ID}.en.txt"
    assert r.translation_path.read_text().strip() == "It is one o'clock."
    # Original transcript is untouched and still saved separately.
    assert (tmp_path / "out" / "transcripts" / f"{RUN_ID}.txt").read_text().startswith("Il est")
    analysis = json.loads(r.analysis_path.read_text())
    block = analysis["transcript"]
    assert block["language"] == "fr" and block["translated"] is True
    assert block["text"] == "Il est treize heures."
    assert block["text_en"] == "It is one o'clock."
    assert block["segments"][0] == {"id": 0, "start": 0.0, "end": 2.0,
                                    "text": "Il est treize heures.", "text_en": "It is one o'clock."}
    manifest = json.loads(r.manifest_path.read_text())
    assert manifest["translation"]["text_en"] == "It is one o'clock."
    assert FAKE_KEY not in r.translation_path.read_text() + r.analysis_path.read_text()


def test_english_broadcast_skips_translation_but_fills_text_en(tmp_path, pipeline):
    r = api.process_station(FRENCH_HISTORY, "en", 60, tmp_path / "out",
                            staging_dir=tmp_path / "staging", llm=LLM)
    assert r.statuses["translate"] == "skipped"
    assert r.stages["translate"].detail == "broadcast is already in English"
    assert not any(c[0] == "translate" for c in pipeline["calls"] if isinstance(c, tuple))
    block = json.loads(r.analysis_path.read_text())["transcript"]
    assert block["translated"] is False and block["text_en"] == block["text"]


def test_translate_false_turns_it_off(tmp_path, pipeline):
    r = run(tmp_path, llm=LLM, translate=False)
    assert r.statuses["translate"] == "skipped" and r.stages["translate"].detail == "translate=False"
    assert r.translation is None
    assert json.loads(r.analysis_path.read_text())["transcript"]["text_en"] is None


def test_translation_failure_keeps_transcript_and_analysis(tmp_path, pipeline):
    pipeline["translate_error"] = AnalysisError("still invalid after one repair attempt")
    r = run(tmp_path, llm=LLM)
    assert r.statuses["translate"] == "failed" and r.statuses["analyse"] == "ok"
    assert not r.ok
    assert r.transcript_path.exists()
    block = json.loads(r.analysis_path.read_text())["transcript"]
    assert block["text"] == "Il est treize heures." and block["text_en"] is None


# ------------------------------------------------- manifest / analysis file shape
def test_manifest_has_no_repeated_blocks_and_relative_paths(tmp_path, pipeline):
    r = run(tmp_path, llm=LLM)
    m = json.loads(r.manifest_path.read_text())
    assert "station" not in m["audio"]           # same as top-level station
    assert "transcript" not in m["analysis"]     # same as transcript + translation
    assert m["translation"]["text_en"] and m["transcript"]["text"]
    assert m["audio_path"] == f"audio/{RUN_ID}.wav"
    assert m["transcript_path"] == f"transcripts/{RUN_ID}.json"
    assert m["translation_path"] == f"transcripts/{RUN_ID}.en.txt"
    assert m["analysis_path"] == f"analysis/{RUN_ID}.json"
    assert m["manifest_path"] == f"manifests/{RUN_ID}.json"
    assert m["audio"]["path"] == f"audio/{RUN_ID}.wav"
    # The in-memory result keeps full objects and real paths.
    assert r.audio.station is not None and r.analysis.transcript is not None
    assert r.audio_path.exists()


def test_language_hint_and_language_are_separate(tmp_path, pipeline):
    r = run(tmp_path)
    assert (r.language_hint, r.language) == ("fr", "fr")
    m = json.loads(r.manifest_path.read_text())
    assert (m["language_hint"], m["language"]) == ("fr", "fr")
    r2 = api.process_station(FRENCH_HISTORY, None, 60, tmp_path / "out2", staging_dir=tmp_path / "s2")
    assert r2.language_hint is None


def test_analysis_file_is_self_contained(tmp_path, pipeline):
    r = run(tmp_path, llm=LLM)
    a = json.loads(r.analysis_path.read_text())
    assert next(iter(a)) == "run"
    run_block = a["run"]
    assert run_block["run_id"] == RUN_ID
    assert run_block["recorded_at"].startswith("2026-11-09T10:00:00")
    assert run_block["duration_seconds"] == 60 and run_block["language"] == "fr"
    assert run_block["station"]["name"] == "France Info"
    assert set(run_block["station"]) >= {"latitude", "longitude", "location_precision", "countrycode"}


def test_station_overrides_set_exact_location(tmp_path, pipeline):
    r = run(tmp_path, station_overrides={"latitude": 48.85, "longitude": 2.27})
    assert (r.station.latitude, r.station.longitude, r.station.location_precision) == (48.85, 2.27, "station")
    region = {"latitude": 45.0, "longitude": 1.0, "location_precision": "subdivision",
              "subdivision": "Nouvelle-Aquitaine"}
    r2 = run(tmp_path / "b", station_overrides=region)
    assert r2.station.location_precision == "subdivision" and r2.station.subdivision == "Nouvelle-Aquitaine"


def test_invalid_station_overrides_fail_the_resolve_stage(tmp_path, pipeline):
    r = run(tmp_path, station_overrides={"latitude": "north"})
    assert r.statuses["resolve"] == "failed" and "station_overrides" in r.stages["resolve"].error


# ------------------------------------------------------------ 0.4.1 behaviour
def test_unknown_provider_fails_before_recording(tmp_path, pipeline, monkeypatch):
    from listen_in.analysis import get_provider

    monkeypatch.setattr(api, "get_provider", get_provider)
    with pytest.raises(ConfigurationError, match="Unknown LLM provider"):
        run(tmp_path, llm={"provider": "anthropc", "model": "m", "api_key": FAKE_KEY})
    assert pipeline["calls"] == []


def test_missing_whisper_fails_before_recording(tmp_path, pipeline, monkeypatch):
    from listen_in.exceptions import OptionalDependencyError
    from listen_in.transcription import check_transcription_ready

    monkeypatch.setattr(api, "check_transcription_ready", check_transcription_ready)
    monkeypatch.setitem(sys.modules, "whisper", None)
    with pytest.raises(OptionalDependencyError):
        run(tmp_path)
    assert pipeline["calls"] == []


def test_logged_traceback_never_contains_the_key(tmp_path, pipeline, caplog):
    try:
        raise RuntimeError(f"upstream rejected {FAKE_KEY}")
    except RuntimeError as cause:
        err = AnalysisError(f"provider failed for {FAKE_KEY}")
        err.__cause__ = cause
    pipeline["analyse_error"] = err
    r = run(tmp_path, llm=LLM)
    assert r.statuses["analyse"] == "failed"
    assert "Traceback" in caplog.text and "[REDACTED]" in caplog.text
    assert FAKE_KEY not in caplog.text


def test_keep_audio_false_keeps_recording_if_transcription_fails(tmp_path, pipeline, caplog):
    pipeline["transcribe_error"] = TranscriptionError("model load failed")
    r = run(tmp_path, keep_audio=False)
    assert r.statuses["transcribe"] == "failed"
    assert (tmp_path / "out" / "audio" / f"{RUN_ID}.wav").exists()
    assert r.audio_path is not None and "kept for a retry" in caplog.text


def test_keep_audio_false_leaves_no_dangling_paths(tmp_path, pipeline):
    r = run(tmp_path, keep_audio=False)
    assert r.audio.deleted is True and r.audio.path is None and r.audio_path is None
    m = json.loads(r.manifest_path.read_text())
    assert m["audio"]["deleted"] is True and m["audio"]["path"] is None and m["audio_path"] is None
    assert m["transcript"]["audio_path"] is None
    assert json.loads(r.transcript_path.read_text())["audio_path"] is None


def test_manifest_leaves_out_machine_details_and_endpoint(tmp_path, pipeline):
    llm = {**LLM, "base_url": "https://private-resource.example.com"}
    r = run(tmp_path, llm=llm)
    m = json.loads(r.manifest_path.read_text())
    assert "output_dir" not in m
    assert "ffmpeg_path" not in m["audio"]
    assert "base_url" not in m["llm"] and m["llm"]["model"] == "claude-haiku-5-5"
    assert "private-resource" not in r.manifest_path.read_text()
    assert r.output_dir  # still available in Python


def test_transcript_file_uses_relative_audio_path(tmp_path, pipeline):
    r = run(tmp_path)
    assert json.loads(r.transcript_path.read_text())["audio_path"] == f"audio/{RUN_ID}.wav"


def test_impossible_override_coordinates_are_rejected(tmp_path, pipeline):
    r = run(tmp_path, station_overrides={"latitude": 999, "longitude": -500})
    assert r.statuses["resolve"] == "failed" and "station_overrides" in r.stages["resolve"].error
