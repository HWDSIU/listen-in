"""Typed data models (Pydantic v2) returned and accepted by the public API."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from .utils import DEFAULT_MAX_DURATION_SECONDS, utc_now

StageStatus = Literal["pending", "ok", "failed", "skipped"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------- station
class Station(_Model):
    """A resolved radio station: where to record from, plus Radio Browser metadata."""

    stream_url: str = Field(description="URL passed to ffmpeg (url_resolved preferred).")
    name: str = Field(default="", description="Station name, if known.")
    source: Literal["radio_browser", "direct_url"] = "direct_url"
    uuid: str | None = Field(default=None, description="Radio Browser station UUID.")
    url: str | None = Field(default=None, description="Radio Browser 'url' field (unresolved).")
    url_resolved: str | None = None
    homepage: str | None = None
    country: str | None = None
    countrycode: str | None = None
    subdivision: str | None = Field(
        default=None, description="State/region/province (Radio Browser 'state'), e.g. 'Berlin'."
    )
    subdivision_code: str | None = Field(
        default=None, description="ISO 3166-2 subdivision code (Radio Browser 'iso_3166_2'), e.g. 'DE-BE'."
    )
    latitude: float | None = Field(default=None, ge=-90, le=90, description="Latitude; see location_precision.")
    longitude: float | None = Field(default=None, ge=-180, le=180, description="Longitude; see location_precision.")
    location_precision: Literal["station", "subdivision", "country"] | None = Field(
        default=None,
        description="How exact latitude/longitude are: 'station' (Radio Browser or your own "
        "coordinates), 'subdivision' (supplied for a region) or 'country' (listen-in's "
        "approximate reference point for the country, usually the capital). None = no location.",
    )
    language: str | None = Field(default=None, description="Radio Browser language tag(s).")
    language_codes: list[str] = Field(
        default_factory=list, description="Language codes listed by Radio Browser ('languagecodes'), lower-case."
    )
    tags: list[str] = Field(default_factory=list, description="Radio Browser tags, e.g. ['news', 'culture'].")
    favicon: str | None = Field(default=None, description="Station logo URL listed by Radio Browser.")
    codec: str | None = None
    bitrate: int | None = None
    lastcheckok: bool | None = None
    api_server: str | None = Field(default=None, description="Radio Browser server used.")
    resolved_at: datetime = Field(default_factory=utc_now)

    @property
    def display_name(self) -> str:
        return self.name or self.stream_url


# ----------------------------------------------------------------------- capture
class CaptureOptions(_Model):
    """Options for :func:`listen_in.capture_stream`."""

    ffmpeg_path: str | None = Field(
        default=None,
        description="Explicit ffmpeg executable. Defaults to $LISTEN_IN_FFMPEG, then PATH, "
        "then the imageio-ffmpeg bundled binary if installed.",
    )
    sample_rate: int = Field(default=16_000, ge=8_000, le=48_000)
    channels: int = Field(default=1, ge=1, le=2)
    connect_timeout_seconds: float = Field(
        default=20.0, gt=0, description="Network read/connect timeout passed to ffmpeg."
    )
    process_timeout_grace_seconds: float = Field(
        default=90.0,
        gt=0,
        description="Extra wall-clock time allowed beyond the duration before ffmpeg is killed.",
    )
    max_duration_seconds: int = Field(default=DEFAULT_MAX_DURATION_SECONDS, gt=0)
    min_duration_ratio: float = Field(
        default=0.9,
        gt=0,
        le=1,
        description="Fail if the recording is shorter than this fraction of the requested duration.",
    )
    reconnect: bool = Field(default=True, description="Ask ffmpeg to reconnect dropped streams.")
    user_agent: str | None = Field(default=None, description="HTTP User-Agent sent to the stream.")
    extra_input_args: list[str] = Field(
        default_factory=list,
        description="Advanced: extra ffmpeg input options (placed before -i). Not shell-interpreted.",
    )

    @field_validator("extra_input_args")
    @classmethod
    def _no_output_override(cls, v: list[str]) -> list[str]:
        banned = {"-i", "-y", "-f", "-filter_complex"}
        bad = [a for a in v if a in banned]
        if bad:
            raise ValueError(f"extra_input_args may not contain {bad}")
        return v


class AudioCapture(_Model):
    """A completed recording."""

    path: Path | None = Field(description="The WAV file; None once deleted (keep_audio=False).")
    deleted: bool = Field(default=False, description="True if the file was deleted after transcription.")
    station: Station
    started_at: datetime
    finished_at: datetime
    requested_duration_seconds: int
    duration_seconds: float
    sample_rate: int
    channels: int
    size_bytes: int
    ffmpeg_path: str
    format: Literal["wav"] = "wav"


# ------------------------------------------------------------------ transcription
class TranscriptSegment(_Model):
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None


class Transcript(_Model):
    """Original-language Whisper transcript with timestamped segments."""

    text: str
    language: str = Field(description="Language Whisper transcribed in (ISO 639-1).")
    language_hint: str | None = Field(default=None, description="Language requested by the caller, if any.")
    detected_language: str | None = Field(
        default=None, description="Language detected by Whisper across several windows of the audio."
    )
    detected_language_probability: float | None = None
    language_mismatch: bool = Field(
        default=False,
        description="True when the hint clearly disagreed with the detected language.",
    )
    segments: list[TranscriptSegment] = Field(default_factory=list)
    model: str
    device: str
    task: Literal["transcribe"] = "transcribe"
    audio_path: Path | None = None
    duration_seconds: float | None = None
    processing_seconds: float | None = None
    created_at: datetime = Field(default_factory=utc_now)


# ----------------------------------------------------------------------- analysis
EntityType = Literal["person", "organisation", "location"]

#: Fixed topic vocabulary, so stories can be filtered and compared across stations.
Topic = Literal[
    "politics", "elections", "economy", "business", "labour", "conflict", "security",
    "crime_justice", "health", "environment", "science_technology", "education",
    "society", "migration", "culture", "religion", "sport", "weather", "transport",
    "disaster", "international", "other",
]


class Entity(BaseModel):
    """A person, organisation or place named in a story, for linking across stories."""

    model_config = ConfigDict(extra="ignore")

    type: EntityType
    name: str = Field(description="The name as heard in the broadcast (transliterated to Latin script).")
    canonical_name: str = Field(
        description="The standard full form of the name, e.g. 'MEDEF' when 'MDF' was heard, or "
        "'Antonio Guterres' for 'Antony Guterres'. Repeat name when unsure. Only standardise "
        "spelling or expand acronyms; never substitute a different entity."
    )
    role: str = Field(
        description="Role or title as stated in the broadcast, e.g. 'head of the CFDT union'. "
        "Empty string if the broadcast does not say."
    )
    uncertain: bool = Field(
        description="True if the name may be misheard or the canonical_name is a guess."
    )


class TimeRange(BaseModel):
    """A stretch of the recording, in seconds from the start."""

    start: float
    end: float


class NewsStory(BaseModel):
    """One story in the broadcast. Field names match the published analysis schema."""

    model_config = ConfigDict(extra="ignore")

    headline: str = Field(description="Short English headline.")
    summary: str = Field(description="Factual English summary based only on the transcript.")
    topics: list[Topic] = Field(description="One to three topics from the fixed list.")
    entities: list[Entity] = Field(
        description="Every person, organisation and place named in the story, once each, with its "
        "type, standard name and stated role. Exclude the broadcasting station itself."
    )
    key_claims: list[str] = Field(description="Specific factual claims made in the broadcast.")
    uncertainties: list[str] = Field(
        description="Doubtful names, figures or passages (e.g. likely transcription errors)."
    )
    segment_ids: list[int] = Field(
        description="Ids of the transcript segments ([n] markers) this story comes from."
    )
    # Filled in by listen-in, not requested from the model: the plain name lists come from
    # entities (standard names), so the two can never disagree; times come from segment_ids.
    people: SkipJsonSchema[list[str]] = Field(
        default_factory=list, description="Standard names of the people in entities."
    )
    organisations: SkipJsonSchema[list[str]] = Field(
        default_factory=list, description="Standard names of the organisations in entities."
    )
    locations: SkipJsonSchema[list[str]] = Field(
        default_factory=list, description="Standard names of the places in entities."
    )
    time_ranges: SkipJsonSchema[list[TimeRange]] = Field(
        default_factory=list,
        description="Where the story is in the recording: one range per run of consecutive segments. "
        "A story mentioned in the headlines and again later has several.",
    )
    start: SkipJsonSchema[float | None] = Field(
        default=None, description="Start of the first time range (seconds into the recording)."
    )
    end: SkipJsonSchema[float | None] = Field(
        default=None, description="End of the last time range; may span other stories in between."
    )


class NewsAnalysis(BaseModel):
    """Structured English analysis of one transcript."""

    model_config = ConfigDict(extra="ignore")

    news_detected: bool = Field(description="True only if the transcript contains substantive news.")
    broadcast_language: str = Field(description="ISO 639-1 code of the broadcast language.")
    translation_used: bool = Field(
        description="True if the model translated from the broadcast language to write this English "
        "analysis. (Whether listen-in also produced a full English transcript is "
        "transcript.translated.)"
    )
    summary_en: str = Field(description="Concise English summary based only on the transcript.")
    stories: list[NewsStory] = Field(description="Empty when no substantive news is present.")


PROVIDER_ALIASES = {
    "openai_compatible": "openai",
    "azure": "azure_openai",
    "azure_foundry": "azure_openai",
}
DEFAULT_MODELS = {"anthropic": "claude-haiku-5-5"}
AZURE_DEFAULT_API_VERSION = "2024-12-01-preview"


class LLMConfig(_Model):
    """LLM settings for analysis. ``api_key`` is a SecretStr and is never serialised.

    Providers:

    * ``anthropic`` - Claude via the Anthropic API (default model ``claude-haiku-5-5``).
    * ``openai`` - the OpenAI API, or any OpenAI-compatible server via ``base_url``.
    * ``azure_openai`` - Azure OpenAI / Azure AI Foundry. ``base_url`` is the resource
      endpoint (``https://<resource>.openai.azure.com``), ``model`` is the deployment name.
    """

    provider: str = "anthropic"
    model: str | None = Field(
        default=None,
        description="Model name; for azure_openai the deployment name. Defaults to "
        "claude-haiku-5-5 for anthropic and is required for the other providers.",
    )
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    api_key_env: str = Field(
        default="LLM_API_KEY",
        description="Environment variable read when api_key is not given. Provider-specific "
        "variables are tried next: ANTHROPIC_API_KEY, OPENAI_API_KEY or AZURE_OPENAI_API_KEY.",
    )
    base_url: str | None = Field(
        default=None,
        description="Custom API base URL; for azure_openai the resource endpoint "
        "(falls back to $AZURE_OPENAI_ENDPOINT).",
    )
    api_version: str | None = Field(
        default=None, description=f"azure_openai only; defaults to {AZURE_DEFAULT_API_VERSION}."
    )
    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = Field(
        default=None,
        description="Reasoning effort. Sent as output_config.effort (anthropic) or "
        "reasoning_effort (openai, azure_openai) only when set; leave unset for models "
        "without reasoning controls.",
    )
    structured_output: Literal["json_schema", "json_object"] = Field(
        default="json_schema",
        description="openai/azure_openai: 'json_schema' (strict structured outputs) or "
        "'json_object' for deployments that do not support JSON schemas.",
    )
    max_tokens: int = Field(default=16_000, gt=0, description="Maximum output tokens per call.")
    timeout_seconds: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=2, ge=0, description="SDK-level retries for transient errors.")

    @field_validator("provider")
    @classmethod
    def _normalise_provider(cls, v: str) -> str:
        v = v.strip().lower()
        return PROVIDER_ALIASES.get(v, v)

    @model_validator(mode="after")
    def _default_model(self) -> LLMConfig:
        if not self.model:
            if self.provider in DEFAULT_MODELS:
                self.model = DEFAULT_MODELS[self.provider]
            elif self.provider in ("openai", "azure_openai"):
                hint = " (the Azure deployment name)" if self.provider == "azure_openai" else ""
                raise ValueError(f"model is required for provider {self.provider!r}{hint}")
        return self


class TranslatedSegment(_Model):
    """One Whisper segment with its original text and English translation."""

    id: int
    start: float
    end: float
    text: str = Field(description="Original-language text.")
    text_en: str | None = Field(default=None, description="English translation (None if unavailable).")


class Translation(_Model):
    """Segment-by-segment English translation of a transcript."""

    source_language: str
    target_language: Literal["en"] = "en"
    segments: list[TranslatedSegment] = Field(default_factory=list)
    text_en: str = Field(description="The full English translation.")
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    repaired: bool = False
    request_ids: list[str] = Field(default_factory=list)
    processing_seconds: float | None = None
    created_at: datetime = Field(default_factory=utc_now)


class BilingualTranscript(_Model):
    """Original transcript and English side by side, as stored with the analysis."""

    language: str = Field(description="Language of the original transcript (ISO 639-1).")
    translated: bool = Field(
        description="True if text_en is a translation; False if the broadcast was already "
        "English (text_en is then the original) or no translation is available."
    )
    text: str = Field(description="Original-language transcript.")
    text_en: str | None = Field(default=None, description="English text (None if unavailable).")
    segments: list[TranslatedSegment] = Field(default_factory=list)


class RunStation(_Model):
    """The station fields needed to place and label a run (subset of :class:`Station`)."""

    name: str
    uuid: str | None = None
    country: str | None = None
    countrycode: str | None = None
    subdivision: str | None = None
    subdivision_code: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    location_precision: Literal["station", "subdivision", "country"] | None = None
    homepage: str | None = None


class RunContext(_Model):
    """Where and when a run happened, so an analysis file stands on its own."""

    run_id: str
    batch_id: str | None = Field(default=None, description="Shared by runs recorded together.")
    recorded_at: datetime | None = Field(default=None, description="When recording started (UTC).")
    duration_seconds: float | None = None
    language: str | None = Field(default=None, description="Language the transcript is in.")
    station: RunStation | None = None
    package_version: str | None = None


class AnalysisResult(_Model):
    """A validated :class:`NewsAnalysis` plus non-secret call metadata."""

    run: RunContext | None = Field(
        default=None, description="Run, station and time context; added by process_station."
    )
    analysis: NewsAnalysis
    transcript: BilingualTranscript | None = Field(
        default=None,
        description="Original transcript with its English translation; added by process_station.",
    )
    provider: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    repaired: bool = Field(default=False, description="True if a repair retry was needed.")
    request_ids: list[str] = Field(default_factory=list)
    processing_seconds: float | None = None
    created_at: datetime = Field(default_factory=utc_now)


# ----------------------------------------------------------------- end-to-end run
class StageResult(_Model):
    status: StageStatus = "pending"
    seconds: float | None = None
    error: str | None = Field(default=None, description="Non-secret error description.")
    detail: str | None = Field(default=None, description="Why a stage was skipped, if it was.")


class ProcessingResult(_Model):
    """Everything produced by :func:`listen_in.process_station` for one run."""

    run_id: str
    batch_id: str | None = Field(
        default=None, description="Set when the run was part of process_stations; shared by the batch."
    )
    package_version: str
    started_at: datetime
    finished_at: datetime | None = None
    station_input: str
    language_hint: str | None = Field(default=None, description="Language requested (None = detect).")
    language: str | None = Field(default=None, description="Language the transcript is in, once known.")
    requested_duration_seconds: int
    output_dir: str
    station: Station | None = None
    audio: AudioCapture | None = None
    audio_path: Path | None = None
    transcript: Transcript | None = None
    transcript_path: Path | None = None
    translation: Translation | None = None
    translation_path: Path | None = Field(default=None, description="English transcript (.en.txt).")
    analysis: AnalysisResult | None = None
    analysis_path: Path | None = None
    manifest_path: Path | None = None
    llm: dict[str, Any] | None = Field(default=None, description="Non-secret LLM settings used.")
    stages: dict[str, StageResult] = Field(
        default_factory=lambda: {
            s: StageResult() for s in ("resolve", "capture", "transcribe", "translate", "analyse", "persist")
        }
    )

    @property
    def ok(self) -> bool:
        """True when no stage failed (skipped stages such as disabled analysis are fine)."""
        return all(s.status in ("ok", "skipped") for s in self.stages.values())

    @property
    def timings(self) -> dict[str, float | None]:
        return {k: v.seconds for k, v in self.stages.items()}

    @property
    def statuses(self) -> dict[str, StageStatus]:
        return {k: v.status for k, v in self.stages.items()}


# ----------------------------------------------------------------- several stations
class StationRequest(_Model):
    """One station in :func:`listen_in.process_stations`, with its own settings."""

    station: str | Station = Field(description="Stream URL, Radio Browser UUID or history URL, or a Station.")
    language: str | None = Field(default=None, description="ISO 639-1 hint for this station (None = detect).")
    whisper_model: str | None = Field(
        default=None, description="Override the batch's Whisper model, e.g. 'turbo' for Swahili."
    )
    initial_prompt: str | None = None
    station_overrides: dict[str, Any] | None = None


class BatchResult(_Model):
    """Results of :func:`listen_in.process_stations`, one per station, in the order given."""

    batch_id: str
    started_at: datetime
    finished_at: datetime | None = None
    results: list[ProcessingResult] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when every station's run succeeded (skipped stages are fine)."""
        return all(r.ok for r in self.results)

    @property
    def statuses(self) -> dict[str, dict[str, StageStatus]]:
        return {r.run_id: r.statuses for r in self.results}
