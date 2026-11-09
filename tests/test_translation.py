from __future__ import annotations

import json

import pytest

from listen_in import translate_transcript
from listen_in.analysis import LLMResponse
from listen_in.exceptions import AnalysisError
from listen_in.models import Transcript, TranscriptSegment
from listen_in.translation import SYSTEM_PROMPT

from .test_analysis import FakeProvider

FRENCH = ["Il est treize heures.", "Le prix du carburant augmente.", "À Paris, les écoles restent bloquées."]
ENGLISH = ["It is one o'clock.", "The price of fuel is rising.", "In Paris, schools remain blocked."]


def transcript(texts=FRENCH, lang: str = "fr") -> Transcript:
    segs = [TranscriptSegment(id=i, start=i * 3.0, end=i * 3.0 + 2.5, text=t) for i, t in enumerate(texts)]
    return Transcript(text=" ".join(texts), language=lang, segments=segs, model="small", device="cpu")


def reply(pairs: list[tuple[int, str]]) -> LLMResponse:
    body = {"segments": [{"id": i, "text_en": t} for i, t in pairs]}
    return LLMResponse(text=json.dumps(body), stop_reason="end_turn", input_tokens=50, output_tokens=20,
                       request_id=f"req_{pairs[0][0] if pairs else 'x'}")


def test_translates_each_segment_and_keeps_timestamps():
    p = FakeProvider(reply(list(enumerate(ENGLISH))))
    t = translate_transcript(transcript(), llm_provider=p)
    assert [s.text_en for s in t.segments] == ENGLISH
    assert [s.text for s in t.segments] == FRENCH  # original kept alongside
    assert (t.segments[1].start, t.segments[1].end) == (3.0, 5.5)
    assert t.text_en == " ".join(ENGLISH)
    assert t.source_language == "fr" and t.target_language == "en"
    assert (t.input_tokens, t.output_tokens, t.repaired) == (50, 20, False)
    user = p.calls[0]["messages"][0]["content"]
    assert "Source language (ISO 639-1): fr" in user
    assert "[0] Il est treize heures." in user and "[2] À Paris" in user
    assert p.calls[0]["system"] == SYSTEM_PROMPT


def test_prompt_rules():
    assert "exactly one entry per segment id" in SYSTEM_PROMPT
    assert "do not correct them" in SYSTEM_PROMPT
    assert "copy it unchanged" in SYSTEM_PROMPT


def test_long_transcripts_are_batched():
    p = FakeProvider(reply([(0, ENGLISH[0])]), reply([(1, ENGLISH[1])]), reply([(2, ENGLISH[2])]))
    t = translate_transcript(transcript(), llm_provider=p, max_chars_per_call=25)
    assert len(p.calls) == 3
    assert [s.text_en for s in t.segments] == ENGLISH
    assert t.request_ids == ["req_0", "req_1", "req_2"]
    assert t.input_tokens == 150


def test_missing_segment_ids_trigger_repair():
    p = FakeProvider(reply([(0, ENGLISH[0]), (2, ENGLISH[2])]), reply(list(enumerate(ENGLISH))))
    t = translate_transcript(transcript(), llm_provider=p)
    assert t.repaired is True
    repair = p.calls[1]["messages"][-1]["content"]
    assert "missing ids [1]" in repair
    assert [s.text_en for s in t.segments] == ENGLISH


def test_still_incomplete_after_repair_raises():
    bad = reply([(0, ENGLISH[0])])
    with pytest.raises(AnalysisError, match="after one repair"):
        translate_transcript(transcript(), llm_provider=FakeProvider(bad, bad))


def test_unexpected_ids_are_rejected():
    p = FakeProvider(reply([*enumerate(ENGLISH), (99, "extra")]), reply(list(enumerate(ENGLISH))))
    translate_transcript(transcript(), llm_provider=p)
    assert "unexpected ids [99]" in p.calls[1]["messages"][-1]["content"]


def test_empty_transcript_makes_no_call():
    p = FakeProvider()
    t = translate_transcript(Transcript(text="  ", language="fr", model="small", device="cpu"), llm_provider=p)
    assert t.text_en == "" and t.segments == [] and p.calls == []


def test_text_without_segments_is_one_segment():
    p = FakeProvider(reply([(0, "Hello everyone.")]))
    t = translate_transcript(
        Transcript(text="Bonjour à tous.", language="fr", model="small", device="cpu", duration_seconds=4.0),
        llm_provider=p)
    assert len(t.segments) == 1 and t.segments[0].end == 4.0 and t.text_en == "Hello everyone."


def test_translation_through_azure_provider_http():
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx2")
    from listen_in.analysis import OpenAIProvider
    from listen_in.models import LLMConfig

    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        content = json.dumps({"segments": [{"id": i, "text_en": e} for i, e in enumerate(ENGLISH)]})
        return httpx.Response(200, json={
            "id": "c", "object": "chat.completion", "created": 1, "model": "d",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    p = OpenAIProvider("k" * 20, "https://example-resource.openai.azure.com", azure=True,
                       http_client=openai.DefaultHttpxClient(transport=httpx.MockTransport(handler)), max_retries=0)
    t = translate_transcript(transcript(), config=LLMConfig(provider="azure_openai", model="my-gpt-deployment"),
                             llm_provider=p)
    assert t.text_en == " ".join(ENGLISH)
    schema = seen["body"]["response_format"]["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert any(d.get("additionalProperties") is False for d in schema.get("$defs", {}).values())
