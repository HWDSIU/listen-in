"""Opt-in live test against the real Radio Browser API and radio streams.

Records 15 s from each supplied station (30 s total). Skipped unless:

    RUN_LIVE_RADIO_TESTS=1 pytest -m integration

Set LIVE_TRANSCRIBE=1 as well to run Whisper "tiny" on the recordings.
No LLM is called.
"""

from __future__ import annotations

import os

import pytest

from listen_in import capture_stream, resolve_station, transcribe_audio

from .conftest import ENGLISH_HISTORY, FRENCH_HISTORY

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_LIVE_RADIO_TESTS") != "1",
                       reason="set RUN_LIVE_RADIO_TESTS=1 to run live radio tests"),
]


@pytest.mark.parametrize(("ref", "lang"), [(FRENCH_HISTORY, "fr"), (ENGLISH_HISTORY, "en")])
def test_live_resolve_and_capture(tmp_path, ref, lang):
    station = resolve_station(ref)
    assert station.source == "radio_browser"
    assert station.stream_url.startswith("http")

    audio = capture_stream(station, 15, tmp_path)
    assert audio.path.exists()
    assert audio.duration_seconds >= 13.5
    assert (audio.sample_rate, audio.channels) == (16_000, 1)

    if os.environ.get("LIVE_TRANSCRIBE") == "1":
        t = transcribe_audio(audio, language=lang, model="tiny")
        assert t.language == lang
