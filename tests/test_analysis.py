from __future__ import annotations

import json
import logging

import pytest
from pydantic import SecretStr

from listen_in import analysis
from listen_in.analysis import (
    SYSTEM_PROMPT,
    LLMResponse,
    analyse_transcript,
    build_user_message,
    resolve_api_key,
)
from listen_in.exceptions import AnalysisError, AnalysisRefusedError, ConfigurationError
from listen_in.models import LLMConfig, Transcript

from .conftest import FAKE_KEY

VALID = {
    "news_detected": True,
    "broadcast_language": "fr",
    "translation_used": True,
    "summary_en": "A bulletin on fuel prices and a regional budget.",
    "stories": [
        {
            "headline": "Fuel prices rise",
            "summary": "The bulletin said pump prices rose.",
            "topics": ["economy"],
            "people": [],
            "organisations": ["Ministry of Energy"],
            "locations": ["Paris"],
            "entities": [
                {"type": "organisation", "name": "Ministry of Energy", "canonical_name": "Ministry of Energy",
                 "role": "", "uncertain": False},
                {"type": "location", "name": "Paris", "canonical_name": "Paris", "role": "", "uncertain": False},
            ],
            "key_claims": ["Prices rose by 5 cents"],
            "uncertainties": [],
            "segment_ids": [1],
        }
    ],
}


def transcript(text: str = "Il est treize heures. Le prix du carburant augmente.", lang: str = "fr") -> Transcript:
    return Transcript(text=text, language=lang, model="small", device="cpu", duration_seconds=60)


class FakeProvider:
    name = "fake"

    def __init__(self, *responses: LLMResponse):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete_json(self, *, system, messages, schema, config):
        self.calls.append({"system": system, "messages": list(messages), "schema": schema})
        return self.responses.pop(0)


def ok(text: str | dict = VALID, **kw) -> LLMResponse:
    return LLMResponse(text=text if isinstance(text, str) else json.dumps(text),
                       stop_reason="end_turn", input_tokens=100, output_tokens=50, **kw)


# ------------------------------------------------------------------ happy path
def test_valid_json_is_validated():
    p = FakeProvider(ok(request_id="req_1"))
    r = analyse_transcript(transcript(), llm_provider=p, station_name="France Info")
    assert r.analysis.news_detected is True
    assert r.analysis.stories[0].organisations == ["Ministry of Energy"]
    assert r.repaired is False
    assert r.request_ids == ["req_1"]
    assert (r.input_tokens, r.output_tokens) == (100, 50)
    assert len(p.calls) == 1


def test_prompt_requires_transcript_only_and_empty_stories():
    assert "only the transcript" in SYSTEM_PROMPT.lower()
    assert "empty stories list" in SYSTEM_PROMPT
    assert "spelt as heard" in SYSTEM_PROMPT
    msg = build_user_message(transcript(), "France Info")
    assert "Station: France Info" in msg
    assert "Transcript language (ISO 639-1): fr" in msg
    detected = transcript().model_copy(update={"detected_language": "fr", "detected_language_probability": 0.97})
    assert "fr (detected by Whisper, 97% confidence)" in build_user_message(detected)
    assert "<transcript>\nIl est treize heures." in msg


def test_no_news_returns_empty_stories():
    none = {**VALID, "news_detected": False, "stories": [], "summary_en": "Music only."}
    r = analyse_transcript(transcript(), llm_provider=FakeProvider(ok(none)))
    assert r.analysis.stories == []


def test_empty_transcript_skips_llm_call():
    p = FakeProvider()
    r = analyse_transcript(transcript(text="   "), llm_provider=p)
    assert r.analysis.news_detected is False
    assert p.calls == []


# ---------------------------------------------------------------- repair path
def test_invalid_json_gets_one_repair_retry():
    p = FakeProvider(ok("{not json"), ok())
    r = analyse_transcript(transcript(), llm_provider=p)
    assert r.repaired is True
    second = p.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert "invalid JSON" in second[-1]["content"]
    assert r.input_tokens == 200


def test_schema_invalid_json_gets_repair():
    bad = {k: v for k, v in VALID.items() if k != "summary_en"}
    p = FakeProvider(ok(bad), ok())
    r = analyse_transcript(transcript(), llm_provider=p)
    assert r.repaired is True
    assert "summary_en" in p.calls[1]["messages"][-1]["content"]


def test_still_invalid_after_repair_raises():
    p = FakeProvider(ok("nope"), ok("still nope"))
    with pytest.raises(AnalysisError, match="after one repair"):
        analyse_transcript(transcript(), llm_provider=p)
    assert len(p.calls) == 2


def test_refusal_and_truncation():
    with pytest.raises(AnalysisRefusedError):
        analyse_transcript(transcript(), llm_provider=FakeProvider(
            LLMResponse(text="", stop_reason="refusal", refusal_detail="category=None")))
    with pytest.raises(AnalysisError, match="max_tokens"):
        analyse_transcript(transcript(), llm_provider=FakeProvider(
            LLMResponse(text='{"news', stop_reason="max_tokens")))


def test_unexpected_provider_error_is_wrapped_without_key():
    class Boom:
        name = "boom"

        def complete_json(self, **kw):
            raise RuntimeError(f"connection reset (key={FAKE_KEY})")

    analysis.register_provider("boom", lambda cfg, key: Boom())
    with pytest.raises(AnalysisError) as exc:
        analyse_transcript(transcript(), api_key=FAKE_KEY, provider="boom", model="m")
    assert FAKE_KEY not in str(exc.value)


# ------------------------------------------------------------------ config/keys
def test_api_key_from_env(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "from-llm-env")
    assert resolve_api_key(LLMConfig()) == "from-llm-env"


def test_anthropic_key_env_fallback(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-anthropic-env")
    assert resolve_api_key(LLMConfig()) == "from-anthropic-env"


def test_missing_key_and_unknown_provider():
    with pytest.raises(ConfigurationError, match="No API key"):
        analyse_transcript(transcript())
    with pytest.raises(ConfigurationError, match="Unknown LLM provider"):
        analyse_transcript(transcript(), api_key="k" * 10, provider="nope")


def test_api_key_never_serialised():
    cfg = LLMConfig(api_key=SecretStr(FAKE_KEY))
    assert FAKE_KEY not in repr(cfg)
    assert FAKE_KEY not in cfg.model_dump_json()
    assert "api_key" not in cfg.model_dump()


# ------------------------------------------------- Anthropic provider (HTTP mock)
def _anthropic_with_mock(handler):
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    http = anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler))
    return analysis.AnthropicProvider(FAKE_KEY, http_client=http, max_retries=0), httpx2


def _message(text: str, stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 321, "output_tokens": 123},
    }


def test_anthropic_request_shape_and_no_key_leak(caplog):
    seen: dict = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("x-api-key") or request.headers.get("authorization", "")
        return httpx2.Response(200, json=_message(json.dumps(VALID)),
                               headers={"request-id": "req_abc"})

    provider, httpx2 = _anthropic_with_mock(handler)
    caplog.set_level(logging.DEBUG)
    r = analyse_transcript(transcript(), config=LLMConfig(api_key=SecretStr(FAKE_KEY), effort="medium"),
                           llm_provider=provider, station_name="France Info")

    body = seen["body"]
    assert body["model"] == "claude-haiku-5-5"
    assert body["system"] == SYSTEM_PROMPT
    fmt = body["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert set(fmt["schema"]["required"]) >= {"news_detected", "stories", "summary_en"}
    assert body["output_config"]["effort"] == "medium"
    assert FAKE_KEY in seen["auth"]  # sent to the API only
    assert r.request_ids == ["req_abc"]
    assert r.input_tokens == 321
    dumped = r.model_dump_json()
    assert FAKE_KEY not in dumped
    assert FAKE_KEY not in caplog.text


def test_anthropic_401_maps_to_configuration_error():
    def handler(request):
        return httpx2.Response(401, json={"type": "error", "error": {
            "type": "authentication_error", "message": "invalid x-api-key"}})

    provider, httpx2 = _anthropic_with_mock(handler)
    with pytest.raises(ConfigurationError, match="401"):
        analyse_transcript(transcript(), llm_provider=provider)


def test_anthropic_refusal_stop_reason():
    def handler(request):
        return httpx2.Response(200, json=_message("", stop_reason="refusal"))

    provider, httpx2 = _anthropic_with_mock(handler)
    with pytest.raises(AnalysisRefusedError):
        analyse_transcript(transcript(), llm_provider=provider)


def test_anthropic_effort_omitted_unless_set():
    seen: dict = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx2.Response(200, json=_message(json.dumps(VALID)))

    provider, httpx2 = _anthropic_with_mock(handler)
    analyse_transcript(transcript(), llm_provider=provider)
    assert "effort" not in seen["body"]["output_config"]


# ------------------------------------------------- entities, topics, story times
def segmented(lang: str = "fr") -> Transcript:
    from listen_in.models import TranscriptSegment

    texts = ["Il est treize heures.", "Le prix du carburant augmente.", "La météo maintenant."]
    segs = [TranscriptSegment(id=i, start=i * 4.0, end=i * 4.0 + 3.5, text=t) for i, t in enumerate(texts)]
    return Transcript(text=" ".join(texts), language=lang, segments=segs, model="small", device="cpu")


def test_user_message_numbers_segments():
    msg = build_user_message(segmented(), "France Info")
    assert "[0] Il est treize heures.\n[1] Le prix du carburant augmente." in msg


def test_prompt_covers_entities_topics_and_segments():
    for phrase in ("canonical_name", "never to add facts", "broadcasting station itself", "segment_ids", "topics"):
        assert phrase in SYSTEM_PROMPT


def test_story_times_come_from_segment_ids_and_bad_ids_are_dropped():
    story = {**VALID["stories"][0], "segment_ids": [2, 1, 99]}
    r = analyse_transcript(segmented(), llm_provider=FakeProvider(ok({**VALID, "stories": [story]})))
    s = r.analysis.stories[0]
    assert s.segment_ids == [1, 2]
    assert (s.start, s.end) == (4.0, 11.5)
    assert s.entities[0].type == "organisation" and s.topics == ["economy"]


def test_story_without_segments_has_no_times():
    story = {**VALID["stories"][0], "segment_ids": []}
    r = analyse_transcript(segmented(), llm_provider=FakeProvider(ok({**VALID, "stories": [story]})))
    assert (r.analysis.stories[0].start, r.analysis.stories[0].end) == (None, None)


def test_model_cannot_set_story_times_and_unknown_topics_are_rejected():
    from listen_in.models import NewsAnalysis

    story_schema = NewsAnalysis.model_json_schema()["$defs"]["NewsStory"]["properties"]
    assert "start" not in story_schema and "end" not in story_schema
    bad = {**VALID, "stories": [{**VALID["stories"][0], "topics": ["gossip"]}]}
    p = FakeProvider(ok(bad), ok())
    r = analyse_transcript(segmented(), llm_provider=p)
    assert r.repaired is True and "topics" in p.calls[1]["messages"][-1]["content"]


def test_story_time_ranges_split_where_segments_are_not_consecutive():
    trailed = {**VALID["stories"][0], "segment_ids": [0, 2]}       # headline, then later coverage
    continuous = {**VALID["stories"][0], "segment_ids": [1, 2]}
    r = analyse_transcript(segmented(), llm_provider=FakeProvider(ok({**VALID, "stories": [trailed, continuous]})))
    a, b = r.analysis.stories
    assert [(t.start, t.end) for t in a.time_ranges] == [(0.0, 3.5), (8.0, 11.5)]
    assert (a.start, a.end) == (0.0, 11.5)
    assert [(t.start, t.end) for t in b.time_ranges] == [(4.0, 11.5)]
