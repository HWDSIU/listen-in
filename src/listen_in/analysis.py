"""Structured English news analysis of a transcript via a pluggable LLM provider.

Built-in providers:

* ``anthropic`` - Claude via the official ``anthropic`` SDK (structured outputs).
* ``openai`` - the OpenAI API or any OpenAI-compatible server (``base_url``).
* ``azure_openai`` - Azure OpenAI / Azure AI Foundry deployments.

The OpenAI-family providers use Chat Completions with a strict JSON-schema
``response_format``. Every response is validated with Pydantic; one invalid
response gets a single repair retry.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, SecretStr, ValidationError

from .exceptions import (
    AnalysisError,
    AnalysisRefusedError,
    ConfigurationError,
    OptionalDependencyError,
)
from .models import AZURE_DEFAULT_API_VERSION, AnalysisResult, LLMConfig, NewsAnalysis, Transcript
from .utils import Stopwatch, redact

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You analyse transcripts of radio broadcasts produced by automatic speech recognition \
(Whisper). Your output is read by English-speaking analysts monitoring the news.

Rules:
- Use only the transcript. Do not add background, context or facts from your own \
knowledge, and do not guess at events the transcript does not describe.
- Write every field in English. If the broadcast is in another language, translate \
faithfully and set translation_used to true.
- Keep important names of people, organisations and places, spelt as heard. When a \
name, number or passage is doubtful (likely recognition error, unclear audio), \
record that in the story's uncertainties rather than silently correcting it.
- Report claims as claims made in the broadcast ("the presenter said...", \
"according to the bulletin..."), not as established fact.
- Music, adverts, jingles, greetings, call-in chat, horoscopes and similar content \
are not news. If there is no substantive news, set news_detected to false, return an \
empty stories list, and use summary_en to describe briefly what the segment contained.
- One story per distinct news item. summary_en gives a concise overview of the whole \
segment.
- topics: one to three topics per story from the allowed list.
- entities: list every person, organisation and place from that story's people, \
organisations and locations, once each. name is the spelling as heard. canonical_name \
is the standard full form of the same entity (e.g. an acronym spelt out, a misheard \
spelling corrected); repeat name when unsure, and set uncertain to true when the name \
may be misheard or the canonical form is a guess. You may use general knowledge only to \
standardise names, never to add facts. role is the role or title as stated in the \
broadcast, or an empty string. Never list the broadcasting station itself or its \
programmes as an entity.
- segment_ids: the [n] markers of the transcript segments each story comes from."""


def build_user_message(transcript: Transcript, station_name: str | None = None) -> str:
    lang = transcript.language or "unknown"
    if transcript.detected_language == transcript.language and transcript.detected_language_probability:
        lang += f" (detected by Whisper, {transcript.detected_language_probability:.0%} confidence)"
    meta = [f"Transcript language (ISO 639-1): {lang}"]
    if station_name:
        meta.append(f"Station: {station_name}")
    if transcript.duration_seconds:
        meta.append(f"Recording length: {transcript.duration_seconds:.0f} seconds")
    segments = [s for s in transcript.segments if s.text.strip()]
    body = "\n".join(f"[{s.id}] {s.text.strip()}" for s in segments) if segments else transcript.text
    return "\n".join(meta) + f"\n\n<transcript>\n{body}\n</transcript>"


def add_story_times(analysis: NewsAnalysis, transcript: Transcript) -> NewsAnalysis:
    """Keep only segment ids that exist and set each story's start/end from them."""
    by_id = {s.id: s for s in transcript.segments}
    stories = []
    for story in analysis.stories:
        ids = sorted({i for i in story.segment_ids if i in by_id})
        stories.append(story.model_copy(update={
            "segment_ids": ids,
            "start": min(by_id[i].start for i in ids) if ids else None,
            "end": max(by_id[i].end for i in ids) if ids else None,
        }))
    return analysis.model_copy(update={"stories": stories})


# ----------------------------------------------------------------- provider API
@dataclass
class LLMResponse:
    text: str
    stop_reason: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_id: str | None = None
    refusal_detail: str | None = None
    raw_content: Any = field(default=None, repr=False)  # provider-native assistant content for repair turns


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal provider interface: return JSON text conforming to ``schema``."""

    name: str

    def complete_json(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        config: LLMConfig,
    ) -> LLMResponse: ...


class AnthropicProvider:
    """Claude via the official ``anthropic`` SDK with JSON-schema structured outputs."""

    name = "anthropic"

    def __init__(self, api_key: str, base_url: str | None = None, *,
                 timeout: float = 120.0, max_retries: int = 2, http_client: Any = None) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise OptionalDependencyError("anthropic", "llm-anthropic", "LLM analysis") from exc
        self._anthropic = anthropic
        kwargs: dict[str, Any] = {}
        if http_client is not None:  # e.g. anthropic.DefaultHttpxClient(proxy=...) or a test transport
            kwargs["http_client"] = http_client
        self._client = anthropic.Anthropic(
            api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries, **kwargs
        )

    def complete_json(self, *, system, messages, schema, config) -> LLMResponse:
        # Adapt the plain JSON schema to what Claude structured outputs accept.
        strict_schema = self._anthropic.transform_schema(schema)
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": strict_schema}}
        if config.effort:
            output_config["effort"] = config.effort
        a = self._anthropic
        try:
            resp = self._client.messages.create(  # type: ignore[call-overload]  # provider-neutral dict messages
                model=config.model,
                max_tokens=config.max_tokens,
                system=system,
                messages=messages,
                output_config=output_config,
            )
        except a.AuthenticationError as exc:
            raise ConfigurationError("Anthropic rejected the API key (401).") from exc
        except a.PermissionDeniedError as exc:
            raise ConfigurationError(f"Anthropic denied access: {exc.message}") from exc
        except a.NotFoundError as exc:
            raise ConfigurationError(f"Unknown model or endpoint: {config.model!r}") from exc
        except a.BadRequestError as exc:
            raise AnalysisError(f"Anthropic rejected the request: {exc.message}") from exc
        except a.RateLimitError as exc:
            raise AnalysisError("Anthropic rate limit hit after retries (429).") from exc
        except a.APIStatusError as exc:
            raise AnalysisError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
        except a.APIConnectionError as exc:
            raise AnalysisError(f"Could not reach the Anthropic API: {exc}") from exc

        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        refusal = None
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            refusal = (
                f"category={getattr(details, 'category', None)}; "
                f"{getattr(details, 'explanation', '') or ''}".strip()
            )
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            stop_reason=resp.stop_reason,
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            request_id=getattr(resp, "_request_id", None),
            refusal_detail=refusal,
            # SDK block objects (incl. thinking blocks) are passed back unchanged on repair.
            raw_content=list(resp.content),
        )


def openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Adapt a Pydantic JSON schema to OpenAI strict mode.

    Every object gets ``additionalProperties: false`` and lists all of its
    properties as required; ``default`` values are dropped.
    """

    def fix(node: Any) -> Any:
        if isinstance(node, dict):
            node = {k: fix(v) for k, v in node.items() if k != "default"}
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            return node
        if isinstance(node, list):
            return [fix(v) for v in node]
        return node

    return fix(json.loads(json.dumps(schema)))


class OpenAIProvider:
    """OpenAI API, OpenAI-compatible servers, or Azure OpenAI / AI Foundry (``azure=True``).

    Uses Chat Completions with a strict JSON-schema ``response_format`` (or
    ``json_object`` mode when ``LLMConfig.structured_output == "json_object"``).
    """

    name = "openai"

    def __init__(self, api_key: str, base_url: str | None = None, *, azure: bool = False,
                 api_version: str | None = None, timeout: float = 120.0, max_retries: int = 2,
                 http_client: Any = None) -> None:
        try:
            import openai
        except ImportError as exc:
            raise OptionalDependencyError(
                "openai", "llm-openai", "LLM analysis with OpenAI or Azure OpenAI"
            ) from exc
        self._openai = openai
        self._client: Any
        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout, "max_retries": max_retries}
        if http_client is not None:  # e.g. openai.DefaultHttpxClient(proxy=...) or a test transport
            kwargs["http_client"] = http_client
        if azure:
            self.name = "azure_openai"
            endpoint = base_url or os.environ.get("AZURE_OPENAI_ENDPOINT")
            if not endpoint:
                raise ConfigurationError(
                    "Azure OpenAI needs the resource endpoint: set base_url "
                    "(e.g. https://<resource>.openai.azure.com) or $AZURE_OPENAI_ENDPOINT."
                )
            self._client = openai.AzureOpenAI(
                azure_endpoint=endpoint, api_version=api_version or AZURE_DEFAULT_API_VERSION, **kwargs
            )
        else:
            self._client = openai.OpenAI(base_url=base_url, **kwargs)

    def complete_json(self, *, system, messages, schema, config) -> LLMResponse:
        if config.structured_output == "json_object":
            response_format: dict[str, Any] = {"type": "json_object"}
            system = (f"{system}\n\nReply with a single JSON object that matches this JSON schema:\n"
                      f"{json.dumps(schema)}")
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {"name": "news_analysis", "schema": openai_strict_schema(schema),
                                "strict": True},
            }
        kwargs: dict[str, Any] = {
            "model": config.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "response_format": response_format,
            "max_completion_tokens": config.max_tokens,
        }
        if config.effort:
            kwargs["reasoning_effort"] = config.effort
        o = self._openai
        where = "Azure OpenAI" if self.name == "azure_openai" else "OpenAI"
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except o.AuthenticationError as exc:
            raise ConfigurationError(f"{where} rejected the API key (401).") from exc
        except o.PermissionDeniedError as exc:
            raise ConfigurationError(f"{where} denied access: {exc.message}") from exc
        except o.NotFoundError as exc:
            what = "deployment" if self.name == "azure_openai" else "model"
            raise ConfigurationError(
                f"{where} could not find {what} {config.model!r} (404). Check the name"
                + (", endpoint and api_version." if self.name == "azure_openai" else ".")
            ) from exc
        except o.BadRequestError as exc:
            if getattr(exc, "code", None) == "content_filter":
                raise AnalysisRefusedError(f"{where} content filter blocked the request.") from exc
            hint = ""
            if "response_format" in str(exc.message) or "json_schema" in str(exc.message):
                hint = ' (if this deployment lacks JSON-schema support, set structured_output="json_object")'
            raise AnalysisError(f"{where} rejected the request: {exc.message}{hint}") from exc
        except o.RateLimitError as exc:
            raise AnalysisError(f"{where} rate limit hit after retries (429).") from exc
        except o.APIStatusError as exc:
            raise AnalysisError(f"{where} API error {exc.status_code}: {exc.message}") from exc
        except o.APIConnectionError as exc:
            raise AnalysisError(f"Could not reach {where}: {exc}") from exc

        if not resp.choices:
            raise AnalysisError(f"{where} returned no choices.")
        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        refusal = getattr(msg, "refusal", None)
        stop = {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}.get(
            choice.finish_reason or "", choice.finish_reason
        )
        if refusal:
            stop = "refusal"
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            stop_reason=stop,
            input_tokens=getattr(usage, "prompt_tokens", None),
            output_tokens=getattr(usage, "completion_tokens", None),
            request_id=getattr(resp, "_request_id", None) or getattr(resp, "id", None),
            refusal_detail=refusal or ("content filter" if choice.finish_reason == "content_filter" else None),
            raw_content=text,
        )


# ------------------------------------------------------------- provider registry
ProviderFactory = Callable[[LLMConfig, str], LLMProvider]

_PROVIDERS: dict[str, ProviderFactory] = {
    "anthropic": lambda cfg, key: AnthropicProvider(
        key, cfg.base_url, timeout=cfg.timeout_seconds, max_retries=cfg.max_retries
    ),
    "openai": lambda cfg, key: OpenAIProvider(
        key, cfg.base_url, timeout=cfg.timeout_seconds, max_retries=cfg.max_retries
    ),
    "azure_openai": lambda cfg, key: OpenAIProvider(
        key, cfg.base_url, azure=True, api_version=cfg.api_version,
        timeout=cfg.timeout_seconds, max_retries=cfg.max_retries,
    ),
}
_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register a custom provider factory ``(config, api_key) -> LLMProvider``."""
    _PROVIDERS[name] = factory


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


def coerce_llm_config(llm: LLMConfig | dict[str, Any]) -> LLMConfig:
    if isinstance(llm, LLMConfig):
        return llm
    if not isinstance(llm, dict):
        raise ConfigurationError("llm must be an LLMConfig or a dict.")
    data = {k: v for k, v in llm.items() if v is not None}
    try:
        return LLMConfig(**data)
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid llm settings: {exc.errors(include_input=False)}") from exc


def resolve_api_key(config: LLMConfig) -> str:
    if config.api_key is not None and config.api_key.get_secret_value():
        return config.api_key.get_secret_value()
    names = [config.api_key_env]
    if config.provider in _KEY_ENV:
        names.append(_KEY_ENV[config.provider])
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    raise ConfigurationError(
        f"No API key: pass api_key or set one of {', '.join(dict.fromkeys(names))}."
    )


def get_provider(config: LLMConfig, api_key: str) -> LLMProvider:
    factory = _PROVIDERS.get(config.provider)
    if factory is None:
        raise ConfigurationError(
            f"Unknown LLM provider {config.provider!r}. Available: {', '.join(available_providers())}."
        )
    return factory(config, api_key)


# ------------------------------------------------- validated structured calls
def _repair_instruction(error: str) -> str:
    return (
        "Your previous reply could not be used: " + error + "\n"
        "Reply again with only a single JSON object that matches the required schema exactly. "
        "Use only the transcript."
    )


def resolve_llm_config(
    *,
    api_key: str | None,
    provider: str,
    model: str | None,
    base_url: str | None,
    api_version: str | None,
    config: LLMConfig | dict[str, Any] | None,
) -> LLMConfig:
    """Build an :class:`LLMConfig` from keyword arguments or an explicit ``config``."""
    if config is None:
        return coerce_llm_config({
            "provider": provider, "model": model, "base_url": base_url, "api_version": api_version,
            "api_key": SecretStr(api_key) if api_key else None,
        })
    cfg = coerce_llm_config(config)
    if api_key:
        cfg = cfg.model_copy(update={"api_key": SecretStr(api_key)})
    return cfg


@dataclass
class ValidatedCall:
    parsed: Any
    input_tokens: int | None
    output_tokens: int | None
    request_ids: list[str]
    repaired: bool


def complete_validated(
    llm_provider: LLMProvider,
    *,
    system: str,
    user_message: str,
    output_model: type[BaseModel],
    config: LLMConfig,
    secrets: list[str | None],
    task: str,
    check: Callable[[Any], None] | None = None,
) -> ValidatedCall:
    """One structured LLM call validated against ``output_model``, with one repair retry.

    ``check`` may raise ``ValueError`` for problems the schema cannot express; that
    also triggers the repair turn. ``task`` names the job in error messages.
    """
    schema = output_model.model_json_schema()  # each provider adapts it to its own rules
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
    in_tok = out_tok = 0
    request_ids: list[str] = []
    last_error = ""
    for attempt in range(2):  # first try + one repair retry
        try:
            resp = llm_provider.complete_json(system=system, messages=messages, schema=schema, config=config)
        except (AnalysisError, ConfigurationError):
            raise
        except Exception as exc:  # provider bugs/unknown errors -> AnalysisError, key-safe
            raise AnalysisError(redact(f"{type(exc).__name__}: {exc}", secrets)) from exc
        in_tok += resp.input_tokens or 0
        out_tok += resp.output_tokens or 0
        if resp.request_id:
            request_ids.append(resp.request_id)
        if resp.stop_reason == "refusal":
            raise AnalysisRefusedError(f"The model declined to {task} ({resp.refusal_detail}).")
        if resp.stop_reason == "max_tokens":
            raise AnalysisError(f"Response truncated at max_tokens={config.max_tokens}; raise LLMConfig.max_tokens.")
        try:
            parsed = output_model.model_validate(json.loads(resp.text))
            if check is not None:
                check(parsed)
        except json.JSONDecodeError as exc:
            last_error = f"invalid JSON ({exc.msg})"
        except ValidationError as exc:
            last_error = f"schema validation failed: {exc.errors(include_input=False, include_url=False)}"
        except ValueError as exc:
            last_error = str(exc)
        else:
            return ValidatedCall(parsed, in_tok or None, out_tok or None, request_ids, attempt > 0)
        log.warning("Attempt %d to %s returned invalid output: %s", attempt + 1, task, last_error)
        # Append-only repair turn: keep the model's own reply, then ask for a fix.
        messages = [
            *messages,
            {"role": "assistant", "content": resp.raw_content or resp.text or "(empty)"},
            {"role": "user", "content": _repair_instruction(last_error)},
        ]
    raise AnalysisError(f"LLM output was still invalid after one repair attempt: {last_error}")


# ----------------------------------------------------------------------- analyse
def analyse_transcript(
    transcript: Transcript,
    api_key: str | None = None,
    provider: str = "anthropic",
    model: str | None = None,
    *,
    base_url: str | None = None,
    api_version: str | None = None,
    station_name: str | None = None,
    config: LLMConfig | dict[str, Any] | None = None,
    llm_provider: LLMProvider | None = None,
) -> AnalysisResult:
    """Summarise a transcript into a validated :class:`NewsAnalysis` (English).

    ``provider`` is ``"anthropic"`` (default model ``claude-haiku-5-5``), ``"openai"``
    or ``"azure_openai"`` (``model`` = deployment name, ``base_url`` = resource endpoint).
    ``api_key`` may be omitted to read ``$LLM_API_KEY`` or the provider's own variable.
    Pass ``config`` for full control, or ``llm_provider`` to inject a custom client.
    The transcript itself is never modified.
    """
    cfg = resolve_llm_config(api_key=api_key, provider=provider, model=model, base_url=base_url,
                             api_version=api_version, config=config)
    if not transcript.text.strip():
        return AnalysisResult(
            analysis=NewsAnalysis(
                news_detected=False,
                broadcast_language=transcript.language or "",
                translation_used=False,
                summary_en="The recording contained no recognisable speech.",
                stories=[],
            ),
            provider=cfg.provider,
            model=cfg.model or "",
        )

    key = None
    if llm_provider is None:
        key = resolve_api_key(cfg)
        llm_provider = get_provider(cfg, key)

    sw = Stopwatch()
    call = complete_validated(
        llm_provider,
        system=SYSTEM_PROMPT,
        user_message=build_user_message(transcript, station_name),
        output_model=NewsAnalysis,
        config=cfg,
        secrets=[key],
        task="analyse this transcript",
    )
    return AnalysisResult(
        analysis=add_story_times(call.parsed, transcript),
        provider=cfg.provider,
        model=cfg.model or "",
        input_tokens=call.input_tokens,
        output_tokens=call.output_tokens,
        repaired=call.repaired,
        request_ids=call.request_ids,
        processing_seconds=sw.seconds,
    )
