from __future__ import annotations

import wave
from datetime import datetime, timezone
from pathlib import Path

import pytest

from listen_in import radio_browser, transcription
from listen_in.models import Station

FRENCH_HISTORY = "https://www.radio-browser.info/history/31074f8a-e6f4-11e9-a96c-52543be04c81"
ENGLISH_HISTORY = "https://www.radio-browser.info/history/598c4d0e-6b06-43fb-bff4-717c591213a9"
FRENCH_UUID = "31074f8a-e6f4-11e9-a96c-52543be04c81"
ENGLISH_UUID = "598c4d0e-6b06-43fb-bff4-717c591213a9"
FIXED_NOW = datetime(2026, 11, 9, 10, 0, 0, tzinfo=timezone.utc)
FAKE_KEY = "test-api-key-SECRET-should-never-leak-0123456789"


def write_wav(path: Path, seconds: float, rate: int = 16_000, channels: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * channels * int(seconds * rate))
    return path


def station_record(uuid: str = FRENCH_UUID, **overrides) -> dict:
    rec = {
        "stationuuid": uuid,
        "name": "France Info",
        "url": "http://icecast.radiofrance.fr/franceinfo-hifi.aac",
        "url_resolved": "http://icecast.radiofrance.fr/franceinfo-hifi.aac",
        "homepage": "https://www.radiofrance.fr/franceinfo",
        "country": "France",
        "countrycode": "FR",
        "language": "french",
        "codec": "AAC",
        "bitrate": 192,
        "lastcheckok": 1,
    }
    rec.update(overrides)
    return rec


@pytest.fixture
def news_station() -> Station:
    return Station(
        stream_url="http://icecast.radiofrance.fr/franceinfo-hifi.aac",
        name="France Info",
        source="radio_browser",
        uuid=FRENCH_UUID,
    )


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Reset module caches and strip real credentials from the environment."""
    monkeypatch.setattr(radio_browser, "_server_cache", None)
    monkeypatch.setattr(radio_browser, "_last_good", None)
    for var in ("LLM_API_KEY", "ANTHROPIC_API_KEY", "LISTEN_IN_FFMPEG"):
        monkeypatch.delenv(var, raising=False)
    transcription.clear_model_cache()
    yield
    transcription.clear_model_cache()
