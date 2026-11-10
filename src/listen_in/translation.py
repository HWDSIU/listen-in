"""Segment-by-segment English translation of a transcript with an LLM.

Whisper's own translate mode is not used: it is weaker than an LLM and would replace
the original text. Here the original-language transcript is kept and each Whisper
segment gets an English line, so quotes can be checked against the original and
the audio timestamps. Long transcripts are translated in batches.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .analysis import (
    LLMProvider,
    complete_validated,
    get_provider,
    resolve_api_key,
    resolve_llm_config,
)
from .models import LLMConfig, Transcript, TranscriptSegment, TranslatedSegment, Translation
from .utils import Stopwatch

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You translate transcripts of radio broadcasts into English. The transcripts come from \
automatic speech recognition (Whisper) and are split into numbered segments.

Rules:
- Translate every segment faithfully and completely. Do not summarise, merge, \
reorder, add or leave out content.
- Return exactly one entry per segment id you were given, with the same id.
- Keep names of people, organisations and places as heard; do not correct them from \
your own knowledge. Keep numbers, amounts and dates as stated.
- If a segment is already in English, copy it unchanged. If it mixes languages, \
translate the non-English parts.
- If a segment is unintelligible (recognition errors), give your best literal \
reading and add [unclear]."""

#: Source characters per LLM call; keeps each response well inside output limits.
DEFAULT_MAX_CHARS_PER_CALL = 6_000


class _SegmentEnglish(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int = Field(description="The segment id, exactly as given.")
    text_en: str = Field(description="English translation of that segment.")


class _TranslationOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    segments: list[_SegmentEnglish] = Field(description="One entry per input segment, in order.")


def _batches(segments: list[TranscriptSegment], max_chars: int) -> list[list[TranscriptSegment]]:
    batches: list[list[TranscriptSegment]] = [[]]
    size = 0
    for seg in segments:
        if batches[-1] and size + len(seg.text) > max_chars:
            batches.append([])
            size = 0
        batches[-1].append(seg)
        size += len(seg.text)
    return [b for b in batches if b]


def _user_message(language: str, batch: list[TranscriptSegment]) -> str:
    lines = "\n".join(f"[{s.id}] {s.text}" for s in batch)
    return (f"Source language (ISO 639-1): {language or 'unknown'}\n"
            f"Translate segments {batch[0].id} to {batch[-1].id} into English.\n\n"
            f"<segments>\n{lines}\n</segments>")


def _id_check(expected: list[int]):
    def check(out: _TranslationOutput) -> None:
        got = [s.id for s in out.segments]
        missing = sorted(set(expected) - set(got))
        extra = sorted(set(got) - set(expected))
        if missing or extra:
            raise ValueError(
                f"Return exactly one entry for each segment id {expected[0]}-{expected[-1]}; "
                f"missing ids {missing[:20]}, unexpected ids {extra[:20]}."
            )
    return check


def translate_transcript(
    transcript: Transcript,
    api_key: str | None = None,
    provider: str = "anthropic",
    model: str | None = None,
    *,
    base_url: str | None = None,
    api_version: str | None = None,
    config: LLMConfig | dict[str, Any] | None = None,
    llm_provider: LLMProvider | None = None,
    max_chars_per_call: int = DEFAULT_MAX_CHARS_PER_CALL,
) -> Translation:
    """Translate a transcript into English, segment by segment, keeping timestamps.

    Takes the same provider settings as :func:`listen_in.analyse_transcript`. The
    original transcript is not modified; the result pairs each original segment with
    its English text. Raises :class:`~listen_in.AnalysisError` if the model's output
    still does not cover every segment after one repair retry.
    """
    cfg = resolve_llm_config(api_key=api_key, provider=provider, model=model, base_url=base_url,
                             api_version=api_version, config=config)
    segments = [s for s in transcript.segments if s.text.strip()]
    if not segments and transcript.text.strip():
        segments = [TranscriptSegment(id=0, start=0.0, end=transcript.duration_seconds or 0.0,
                                      text=transcript.text.strip())]
    if not segments:
        return Translation(source_language=transcript.language, text_en="",
                           provider=cfg.provider, model=cfg.model or "")

    key = cfg.api_key.get_secret_value() if cfg.api_key else None  # for redaction
    if llm_provider is None:
        key = resolve_api_key(cfg)
        llm_provider = get_provider(cfg, key)

    sw = Stopwatch()
    english: dict[int, str] = {}
    in_tok = out_tok = 0
    request_ids: list[str] = []
    repaired = False
    for batch in _batches(segments, max_chars_per_call):
        call = complete_validated(
            llm_provider,
            system=SYSTEM_PROMPT,
            user_message=_user_message(transcript.language, batch),
            output_model=_TranslationOutput,
            config=cfg,
            secrets=[key],
            task="translate this transcript",
            check=_id_check([s.id for s in batch]),
        )
        for item in call.parsed.segments:
            english.setdefault(item.id, item.text_en.strip())
        in_tok += call.input_tokens or 0
        out_tok += call.output_tokens or 0
        request_ids += call.request_ids
        repaired = repaired or call.repaired

    pairs = [
        TranslatedSegment(id=s.id, start=s.start, end=s.end, text=s.text, text_en=english.get(s.id))
        for s in segments
    ]
    return Translation(
        source_language=transcript.language,
        segments=pairs,
        text_en=" ".join(p.text_en for p in pairs if p.text_en),
        provider=cfg.provider,
        model=cfg.model or "",
        input_tokens=in_tok or None,
        output_tokens=out_tok or None,
        repaired=repaired,
        request_ids=request_ids,
        processing_seconds=sw.seconds,
    )
