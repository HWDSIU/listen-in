from __future__ import annotations

import wave
from datetime import datetime, timezone
from pathlib import Path

import pytest

from listen_in import radio_browser, transcription
from listen_in.models import Station

SWAHILI_HISTORY = "https://www.radio-browser.info/history/fc939579-455a-4593-9e0d-ee9df38f5c2f"
ENGLISH_HISTORY = "https://www.radio-browser.info/history/cf0316fa-091e-4dc4-92a7-d8b0bd0cad28"
SWAHILI_UUID = "fc939579-455a-4593-9e0d-ee9df38f5c2f"
ENGLISH_UUID = "cf0316fa-091e-4dc4-92a7-d8b0bd0cad28"
FIXED_NOW = datetime(2026, 11, 9, 10, 0, 0, tzinfo=timezone.utc)
FAKE_KEY = "sk-ant-test-SECRET-should-never-leak-0123456789"


def write_wav(path: Path, seconds: float, rate: int = 16_000, channels: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * channels * int(seconds * rate))
    return path


def station_record(uuid: str = SWAHILI_UUID, **overrides) -> dict:
    rec = {
        "stationuuid": uuid,
        "name": "Radio Jambo",
        "url": "https://radiojambo-atunwadigital.streamguys1.com/radiojambo",
        "url_resolved": "https://radiojambo-atunwadigital.streamguys1.com/radiojambo",
        "homepage": "https://www.radiojambo.co.ke/",
        "country": "Kenya",
        "countrycode": "KE",
        "language": "swahili",
        "codec": "AAC",
        "bitrate": 64,
        "lastcheckok": 1,
    }
    rec.update(overrides)
    return rec


@pytest.fixture
def jambo_station() -> Station:
    return Station(
        stream_url="https://radiojambo-atunwadigital.streamguys1.com/radiojambo",
        name="Radio Jambo",
        source="radio_browser",
        uuid=SWAHILI_UUID,
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
