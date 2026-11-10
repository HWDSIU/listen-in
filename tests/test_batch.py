"""process_stations: simultaneous recording, queued transcription, pooled LLM work."""

from __future__ import annotations

import json
import threading
import time
from datetime import timedelta

import pytest

from listen_in import api
from listen_in.exceptions import InvalidInputError, StreamUnavailableError
from listen_in.models import (
    AnalysisResult,
    AudioCapture,
    NewsAnalysis,
    Station,
    StationRequest,
    Transcript,
    TranslatedSegment,
    Translation,
)

from .conftest import FIXED_NOW, write_wav

UUIDS = {
    "de": "1c3e8be2-5b14-4933-bad3-87cbc227cba4",
    "fr": "31074f8a-e6f4-11e9-a96c-52543be04c81",
    "en": "598c4d0e-6b06-43fb-bff4-717c591213a9",
    "sw": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    "es": "11111111-2222-3333-4444-555555555555",
    "it": "66666666-7777-8888-9999-000000000000",
}
NAMES = {"de": "Deutschlandfunk", "fr": "France Info", "en": "BBC World Service", "sw": "Radio Swahili",
         "es": "Radio Espana", "it": "Radio Italia"}
LLM = {"provider": "anthropic", "model": "claude-haiku-5-5", "api_key": "test-key-0123456789"}


class Recorder:
    """Thread-safe log of what each fake stage did and when."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events: list[tuple[str, str, float]] = []
        self.active = {"transcribe": 0, "llm": 0}
        self.max_active = {"transcribe": 0, "llm": 0}

    def add(self, kind, key):
        with self.lock:
            self.events.append((kind, key, time.monotonic()))

    def enter(self, kind):
        with self.lock:
            self.active[kind] += 1
            self.max_active[kind] = max(self.max_active[kind], self.active[kind])

    def leave(self, kind):
        with self.lock:
            self.active[kind] -= 1

    def times(self, kind):
        return {key: t for k, key, t in self.events if k == kind}

    def order(self, kind):
        return [key for k, key, _ in self.events if k == kind]


@pytest.fixture
def batch(monkeypatch):
    rec = Recorder()
    cfg = {"capture_delay": {}, "capture_error": {}, "transcribe_seconds": 0.15, "llm_seconds": 0.15,
           "releases": [], "ready_checks": []}

    def lang_of(ref):
        return next(lang for lang, u in UUIDS.items() if u in ref)

    def resolve(ref, **kw):
        lang = lang_of(ref)
        return Station(stream_url=f"https://example.org/{lang}", name=NAMES[lang], source="radio_browser",
                       uuid=UUIDS[lang], countrycode="FR")

    def capture(st, duration, out_dir, *, options, overwrite, filename):
        lang = st.stream_url.rsplit("/", 1)[-1]
        rec.add("capture_start", lang)
        time.sleep(cfg["capture_delay"].get(lang, 0.05))
        if lang in cfg["capture_error"]:
            raise cfg["capture_error"][lang]
        path = write_wav(out_dir / filename, 1)
        rec.add("capture_end", lang)
        return AudioCapture(path=path, station=st, started_at=FIXED_NOW,
                            finished_at=FIXED_NOW + timedelta(seconds=time.monotonic()),
                            requested_duration_seconds=duration, duration_seconds=duration,
                            sample_rate=16_000, channels=1, size_bytes=path.stat().st_size,
                            ffmpeg_path="/usr/bin/ffmpeg")

    def transcribe(path, *, language, model, device, initial_prompt, on_language_mismatch):
        rec.enter("transcribe")
        rec.add("transcribe", f"{language}:{model}")
        time.sleep(cfg["transcribe_seconds"])
        rec.leave("transcribe")
        return Transcript(text=f"text in {language}", language=language, model=model, device="cpu",
                          audio_path=path)

    def translate(transcript, *, config, llm_provider=None):
        rec.enter("llm")
        rec.add("translate", transcript.language)
        time.sleep(cfg["llm_seconds"])
        rec.leave("llm")
        seg = TranslatedSegment(id=0, start=0, end=1, text=transcript.text, text_en="english")
        return Translation(source_language=transcript.language, segments=[seg], text_en="english",
                           provider=config.provider, model=config.model)

    def analyse(transcript, *, config, station_name, llm_provider=None):
        rec.add("analyse", transcript.language)
        return AnalysisResult(
            analysis=NewsAnalysis(news_detected=True, broadcast_language=transcript.language,
                                  translation_used=False, summary_en=f"Bulletin from {station_name}.",
                                  stories=[]),
            provider=config.provider, model=config.model)

    def ready(language, model, device, policy):
        cfg["ready_checks"].append((language, model))
        if language == "xx":
            raise InvalidInputError("Unsupported Whisper language code: 'xx'")
        return language

    monkeypatch.setattr(api, "resolve_station", resolve)
    monkeypatch.setattr(api, "capture_stream", capture)
    monkeypatch.setattr(api, "transcribe_audio", transcribe)
    monkeypatch.setattr(api, "translate_transcript", translate)
    monkeypatch.setattr(api, "analyse_transcript", analyse)
    monkeypatch.setattr(api, "check_transcription_ready", ready)
    monkeypatch.setattr(api, "get_provider", lambda cfg_, key: object())
    monkeypatch.setattr(api, "release_whisper_model", lambda model: cfg["releases"].append(model) or True)
    monkeypatch.setattr(api, "utc_now", lambda: FIXED_NOW)
    return rec, cfg


def run(tmp_path, stations, **kw):
    kw.setdefault("llm", LLM)
    return api.process_stations(stations, 60, tmp_path / "out", staging_dir=tmp_path / "staging", **kw)


def entry(lang, **kw):
    return {"station": UUIDS[lang], "language": lang, **kw}


# ------------------------------------------------------------- happy path
def test_recordings_start_together_and_results_keep_input_order(tmp_path, batch):
    rec, cfg = batch
    cfg["capture_delay"] = {"de": 0.3, "fr": 0.3, "en": 0.3}
    b = run(tmp_path, [entry("de"), entry("fr"), entry("en")])

    starts, ends = rec.times("capture_start"), rec.times("capture_end")
    assert max(starts.values()) - min(starts.values()) < 0.1          # all began together
    assert max(ends.values()) - min(starts.values()) < 0.3 + 0.15     # 0.3 s in parallel, not 0.9 s
    assert b.ok
    assert [r.station.name for r in b.results] == ["Deutschlandfunk", "France Info", "BBC World Service"]
    assert len({r.run_id for r in b.results}) == 3
    assert {r.batch_id for r in b.results} == {b.batch_id} and b.batch_id.startswith("batch_")
    for r in b.results:
        m = json.loads(r.manifest_path.read_text())
        a = json.loads(r.analysis_path.read_text())
        assert m["batch_id"] == b.batch_id and a["run"]["batch_id"] == b.batch_id
    assert b.finished_at is not None and set(b.statuses) == {r.run_id for r in b.results}


def test_transcription_one_at_a_time_in_finish_order(tmp_path, batch):
    rec, cfg = batch
    cfg["capture_delay"] = {"de": 0.30, "fr": 0.05, "en": 0.15}
    run(tmp_path, [entry("de"), entry("fr"), entry("en")])
    assert rec.max_active["transcribe"] == 1
    assert rec.order("transcribe") == ["fr:small", "en:small", "de:small"]


def test_llm_work_overlaps_transcription(tmp_path, batch):
    rec, cfg = batch
    cfg["transcribe_seconds"], cfg["llm_seconds"] = 0.2, 0.3
    run(tmp_path, [entry("de"), entry("fr"), entry("es")])
    transcribed = rec.times("transcribe")
    translated = rec.times("translate")
    # The first station's translation starts before the last station's transcription.
    assert min(translated.values()) < max(transcribed.values())
    assert rec.max_active["llm"] >= 2


def test_max_llm_workers_limits_concurrent_llm_calls(tmp_path, batch):
    rec, cfg = batch
    cfg["transcribe_seconds"], cfg["llm_seconds"] = 0.01, 0.1
    run(tmp_path, [entry("de"), entry("fr"), entry("es")], max_llm_workers=1)
    assert rec.max_active["llm"] == 1


# ------------------------------------------------------ several Whisper models
def test_same_model_goes_first_and_unneeded_model_is_released(tmp_path, batch):
    rec, cfg = batch
    # fr finishes first (small); while it is transcribed, sw (turbo) and de (small) finish.
    cfg["capture_delay"] = {"fr": 0.05, "sw": 0.08, "de": 0.10}
    cfg["transcribe_seconds"] = 0.25
    b = run(tmp_path, [entry("sw", whisper_model="turbo"), entry("fr"), entry("de")], whisper_model="small")
    assert b.ok
    assert rec.order("transcribe") == ["fr:small", "de:small", "sw:turbo"]
    assert cfg["releases"] == ["small"]           # small dropped before turbo loads
    assert ("sw", "turbo") in cfg["ready_checks"] and ("fr", "small") in cfg["ready_checks"]


def test_model_still_needed_is_not_released(tmp_path, batch):
    rec, cfg = batch
    # small (fr) first, then turbo (sw) is the only one ready while de (small) still records.
    cfg["capture_delay"] = {"fr": 0.05, "sw": 0.06, "de": 0.6}
    cfg["transcribe_seconds"] = 0.1
    run(tmp_path, [entry("sw", whisper_model="turbo"), entry("fr"), entry("de")], whisper_model="small")
    assert rec.order("transcribe") == ["fr:small", "sw:turbo", "de:small"]
    assert "small" not in cfg["releases"]


# ------------------------------------------------------------ failures
def test_one_station_failing_does_not_affect_the_others(tmp_path, batch):
    rec, cfg = batch
    cfg["capture_error"] = {"fr": StreamUnavailableError("404 Not Found")}
    b = run(tmp_path, [entry("de"), entry("fr"), entry("en")])
    by_name = {r.station.name: r for r in b.results}
    assert by_name["France Info"].statuses["capture"] == "failed"
    assert by_name["France Info"].statuses["transcribe"] == "skipped"
    assert by_name["Deutschlandfunk"].ok and by_name["BBC World Service"].ok
    assert not b.ok
    assert all(r.manifest_path.exists() for r in b.results)
    assert "fr:small" not in rec.order("transcribe")


def test_set_up_of_every_station_is_checked_before_recording(tmp_path, batch):
    rec, _cfg = batch
    with pytest.raises(InvalidInputError, match="'xx'"):
        run(tmp_path, [entry("de"), {"station": UUIDS["fr"], "language": "xx"}])
    assert rec.order("capture_start") == []


@pytest.mark.parametrize("stations", [
    [],
    [entry(lang) for lang in ("de", "fr", "en", "sw", "es", "it")],   # 6 > 5
    "1c3e8be2-5b14-4933-bad3-87cbc227cba4",                         # not a list
])
def test_station_count_limits(tmp_path, batch, stations):
    with pytest.raises(InvalidInputError):
        run(tmp_path, stations)


def test_five_stations_are_allowed(tmp_path, batch):
    _rec, cfg = batch
    cfg["transcribe_seconds"] = cfg["llm_seconds"] = 0.01
    b = run(tmp_path, [entry(lang) for lang in ("de", "fr", "en", "sw", "es")])
    assert len(b.results) == 5 and b.ok


def test_duplicate_stations_and_bad_entries_are_rejected(tmp_path, batch):
    rec, _cfg = batch
    with pytest.raises(InvalidInputError, match="more than once"):
        run(tmp_path, [entry("de"), {"station": f"https://www.radio-browser.info/history/{UUIDS['de']}"}])
    with pytest.raises(InvalidInputError, match="Invalid station entry"):
        run(tmp_path, [{"station": UUIDS["de"], "langauge": "de"}])   # typo'd field
    with pytest.raises(InvalidInputError):
        run(tmp_path, [entry("de")], max_llm_workers=0)
    assert rec.order("capture_start") == []


def test_accepts_strings_station_requests_and_dicts(tmp_path, batch):
    _rec, _cfg = batch
    b = run(tmp_path, [UUIDS["de"], StationRequest(station=UUIDS["fr"], language="fr"), entry("en")], llm=None)
    assert [r.language_hint for r in b.results] == [None, "fr", "en"]
    assert all(r.statuses["analyse"] == "skipped" for r in b.results)


def test_per_station_overrides_apply_only_to_that_station(tmp_path, batch):
    b = run(tmp_path, [entry("de", station_overrides={"latitude": 52.5, "longitude": 13.4}), entry("fr")],
            llm=None)
    de, fr = b.results
    assert (de.station.latitude, de.station.location_precision) == (52.5, "station")
    assert fr.station.latitude is None
