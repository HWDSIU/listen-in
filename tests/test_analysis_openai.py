"""OpenAI and Azure OpenAI / AI Foundry providers, mocked at the HTTP layer."""

from __future__ import annotations

import json
import logging

import pytest
from pydantic import SecretStr

from listen_in.analysis import (
    SYSTEM_PROMPT,
    OpenAIProvider,
    analyse_transcript,
    get_provider,
    openai_strict_schema,
    resolve_api_key,
)
from listen_in.exceptions import AnalysisError, AnalysisRefusedError, ConfigurationError
from listen_in.models import LLMConfig, Transcript

from .conftest import FAKE_KEY
from .test_analysis import VALID, transcript

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx2")  # the OpenAI SDK's HTTP layer

AZURE_ENDPOINT = "https://example-resource.openai.azure.com"
DEPLOYMENT = "my-gpt-deployment"


def completion(content: str | None = None, *, finish_reason: str = "stop", refusal: str | None = None) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": DEPLOYMENT,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content, "refusal": refusal},
            "finish_reason": finish_reason,
        }],
        "usage": {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
    }


def make_provider(handler, *, azure: bool = True, base_url: str | None = AZURE_ENDPOINT,
                  api_version: str | None = None) -> OpenAIProvider:
    http = openai.DefaultHttpxClient(transport=httpx.MockTransport(handler))
    return OpenAIProvider(FAKE_KEY, base_url, azure=azure, api_version=api_version,
                          http_client=http, max_retries=0)


def azure_cfg(**kw) -> LLMConfig:
    return LLMConfig(provider="azure_openai", model=DEPLOYMENT, base_url=AZURE_ENDPOINT,
                     api_key=SecretStr(FAKE_KEY), **kw)


def ok_handler(seen: dict, content: str | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.setdefault("requests", []).append(request)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion(content or json.dumps(VALID)),
                              headers={"x-request-id": "req_azure_1"})
    return handler


# ------------------------------------------------------------------ Azure OpenAI
def test_azure_request_shape_and_no_key_leak(caplog):
    seen: dict = {}
    caplog.set_level(logging.DEBUG)
    r = analyse_transcript(transcript(), config=azure_cfg(), llm_provider=make_provider(ok_handler(seen)),
                           station_name="France Info")

    req = seen["requests"][0]
    assert req.url.host == "example-resource.openai.azure.com"
    assert req.url.path == f"/openai/deployments/{DEPLOYMENT}/chat/completions"
    assert req.url.params["api-version"] == "2024-12-01-preview"
    assert req.headers["api-key"] == FAKE_KEY  # sent to Azure only

    body = seen["body"]
    assert body["model"] == DEPLOYMENT
    assert body["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert body["messages"][1]["role"] == "user"
    assert "Station: France Info" in body["messages"][1]["content"]
    assert body["max_completion_tokens"] == 16_000
    assert "reasoning_effort" not in body and "max_tokens" not in body and "temperature" not in body
    fmt = body["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    story = schema["$defs"]["NewsStory"]
    assert story["additionalProperties"] is False and set(story["required"]) == set(story["properties"])

    assert r.provider == "azure_openai" and r.model == DEPLOYMENT
    assert r.analysis.stories[0].headline == "Fuel prices rise"
    assert (r.input_tokens, r.output_tokens) == (500, 200)
    assert r.request_ids == ["req_azure_1"]
    assert FAKE_KEY not in r.model_dump_json()
    assert FAKE_KEY not in caplog.text


def test_azure_custom_api_version_and_effort():
    seen: dict = {}
    analyse_transcript(transcript(), config=azure_cfg(effort="low"),
                       llm_provider=make_provider(ok_handler(seen), api_version="2025-04-01-preview"))
    assert seen["requests"][0].url.params["api-version"] == "2025-04-01-preview"
    assert seen["body"]["reasoning_effort"] == "low"


def test_azure_endpoint_from_env(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", AZURE_ENDPOINT)
    seen: dict = {}
    analyse_transcript(transcript(), config=azure_cfg(),
                       llm_provider=make_provider(ok_handler(seen), base_url=None))
    assert seen["requests"][0].url.host == "example-resource.openai.azure.com"


def test_azure_without_endpoint_is_configuration_error(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    with pytest.raises(ConfigurationError, match="endpoint"):
        OpenAIProvider(FAKE_KEY, None, azure=True)


def test_registry_builds_azure_client_from_config():
    cfg = LLMConfig(provider="azure_foundry", model=DEPLOYMENT, base_url=AZURE_ENDPOINT)
    assert cfg.provider == "azure_openai"
    p = get_provider(cfg, FAKE_KEY)
    assert isinstance(p, OpenAIProvider) and p.name == "azure_openai"
    assert str(p._client.base_url).startswith(f"{AZURE_ENDPOINT}/openai")


def test_json_object_mode_puts_schema_in_prompt():
    seen: dict = {}
    analyse_transcript(transcript(), config=azure_cfg(structured_output="json_object"),
                       llm_provider=make_provider(ok_handler(seen)))
    body = seen["body"]
    assert body["response_format"] == {"type": "json_object"}
    assert "JSON schema" in body["messages"][0]["content"]
    assert '"summary_en"' in body["messages"][0]["content"]


def test_invalid_json_gets_repair_turn_with_plain_text():
    calls: list[dict] = []
    replies = iter(["{not json", json.dumps(VALID)])

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(200, json=completion(next(replies)))

    r = analyse_transcript(transcript(), config=azure_cfg(), llm_provider=make_provider(handler))
    assert r.repaired is True
    msgs = calls[1]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "{not json"
    assert "invalid JSON" in msgs[3]["content"]


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"refusal": "I can't help with that."}, AnalysisRefusedError, "declined"),
        ({"finish_reason": "content_filter"}, AnalysisRefusedError, "declined"),
        ({"finish_reason": "length", "content": '{"news'}, AnalysisError, "max_tokens"),
    ],
)
def test_refusal_filter_and_truncation(kwargs, error, match):
    def handler(request):
        return httpx.Response(200, json=completion(**kwargs))

    with pytest.raises(error, match=match):
        analyse_transcript(transcript(), config=azure_cfg(), llm_provider=make_provider(handler))


@pytest.mark.parametrize(
    ("status", "body", "error", "match"),
    [
        (401, {"error": {"code": "401", "message": "Access denied due to invalid subscription key"}},
         ConfigurationError, "API key"),
        (404, {"error": {"code": "DeploymentNotFound", "message": "The API deployment does not exist"}},
         ConfigurationError, "deployment 'my-gpt-deployment'"),
        (400, {"error": {"code": "content_filter", "message": "The prompt was filtered"}},
         AnalysisRefusedError, "content filter"),
        (400, {"error": {"code": "invalid_request", "message": "response_format json_schema unsupported"}},
         AnalysisError, "json_object"),
        (500, {"error": {"code": "server_error", "message": "boom"}}, AnalysisError, "500"),
    ],
)
def test_http_errors_map_to_package_errors(status, body, error, match):
    def handler(request):
        return httpx.Response(status, json=body)

    with pytest.raises(error, match=match) as exc:
        analyse_transcript(transcript(), config=azure_cfg(), llm_provider=make_provider(handler))
    assert FAKE_KEY not in str(exc.value)


# --------------------------------------------------------------- OpenAI API
def test_openai_compatible_base_url_and_bearer_auth():
    seen: dict = {}
    cfg = LLMConfig(provider="openai_compatible", model="gpt-6.1", api_key=SecretStr(FAKE_KEY))
    assert cfg.provider == "openai"
    p = make_provider(ok_handler(seen), azure=False, base_url="https://llm.example.org/v1")
    r = analyse_transcript(transcript(), config=cfg, llm_provider=p)
    req = seen["requests"][0]
    assert str(req.url) == "https://llm.example.org/v1/chat/completions"
    assert req.headers["authorization"] == f"Bearer {FAKE_KEY}"
    assert r.provider == "openai"


# --------------------------------------------------------------- config / keys
def test_model_required_for_openai_providers():
    with pytest.raises(ConfigurationError, match="deployment name"):
        analyse_transcript(transcript(), provider="azure_openai", api_key=FAKE_KEY)
    with pytest.raises(ConfigurationError, match="model is required"):
        analyse_transcript(transcript(), provider="openai", api_key=FAKE_KEY)


@pytest.mark.parametrize(("provider", "env"), [("azure_openai", "AZURE_OPENAI_API_KEY"),
                                               ("openai", "OPENAI_API_KEY")])
def test_provider_specific_key_env(monkeypatch, provider, env):
    monkeypatch.setenv(env, "from-provider-env")
    assert resolve_api_key(LLMConfig(provider=provider, model="m")) == "from-provider-env"


def test_strict_schema_transform():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string", "default": "x"},
                       "b": {"type": "object", "properties": {"c": {"type": "integer"}}}},
        "required": ["b"],
    }
    out = openai_strict_schema(schema)
    assert out["required"] == ["a", "b"] and out["additionalProperties"] is False
    assert "default" not in out["properties"]["a"]
    assert out["properties"]["b"]["additionalProperties"] is False
    assert out["properties"]["b"]["required"] == ["c"]
    assert schema["required"] == ["b"]  # input untouched


def test_empty_transcript_needs_no_endpoint():
    r = analyse_transcript(Transcript(text=" ", language="fr", model="small", device="cpu"),
                           provider="azure_openai", model=DEPLOYMENT)
    assert r.analysis.news_detected is False
