"""Public functional API.

Individual steps can be called directly (``resolve_station``, ``capture_stream``,
``transcribe_audio``, ``translate_transcript``, ``analyse_transcript``) or chained
with ``process_station``.
No function here starts a scheduler, a background thread or a service.
"""

from __future__ import annotations

import logging
import shutil
import traceback
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import SecretStr, ValidationError

from ._version import __version__
from .analysis import LLMProvider, analyse_transcript, coerce_llm_config, get_provider, resolve_api_key
from .capture import capture_stream, probe_stream
from .exceptions import CaptureError, InvalidInputError
from .models import (
    AudioCapture,
    BatchResult,
    BilingualTranscript,
    CaptureOptions,
    LLMConfig,
    ProcessingResult,
    RunContext,
    RunStation,
    Station,
    StationRequest,
    Transcript,
    TranslatedSegment,
    Translation,
)
from .radio_browser import parse_station_reference, resolve_station
from .storage import OutputLayout, default_staging_dir, publish_file, write_json, write_text
from .transcription import (
    LanguageMismatchPolicy,
    check_transcription_ready,
    release_whisper_model,
    transcribe_audio,
)
from .translation import translate_transcript
from .utils import (
    Stopwatch,
    build_run_id,
    describe_error,
    extract_uuid,
    redact,
    slugify,
    timestamp_for_filename,
    utc_now,
    validate_duration,
)

log = logging.getLogger(__name__)

__all__ = [
    "analyse_transcript",
    "capture_stream",
    "probe_stream",
    "process_station",
    "process_stations",
    "resolve_station",
    "transcribe_audio",
    "translate_transcript",
]


@dataclass
class _Settings:
    """Settings shared by every station in a run (one station or a batch)."""

    duration: int
    layout: OutputLayout
    options: CaptureOptions
    device: str
    staging_dir: Path
    overwrite: bool
    keep_audio: bool
    on_language_mismatch: LanguageMismatchPolicy
    translate: bool
    llm_cfg: LLMConfig | None
    llm_provider: LLMProvider | None
    secrets: list[str | None]


def _prepare(
    *,
    duration_seconds: int,
    output_dir: str | Path,
    llm: LLMConfig | dict[str, Any] | None,
    device: str,
    capture_options: CaptureOptions | None,
    staging_dir: str | Path | None,
    overwrite: bool,
    keep_audio: bool,
    on_language_mismatch: LanguageMismatchPolicy,
    translate: bool,
) -> _Settings:
    """Validate shared settings and build the LLM client before anything is recorded."""
    options = capture_options or CaptureOptions()
    duration = validate_duration(duration_seconds, options.max_duration_seconds)
    layout = OutputLayout.create(output_dir)
    llm_cfg: LLMConfig | None = None
    api_key: str | None = None
    llm_provider: LLMProvider | None = None
    if llm is not None:
        llm_cfg = coerce_llm_config(llm)
        api_key = resolve_api_key(llm_cfg)
        llm_cfg = llm_cfg.model_copy(update={"api_key": SecretStr(api_key)})
        llm_provider = get_provider(llm_cfg, api_key)  # unknown provider, missing SDK or endpoint
    return _Settings(
        duration=duration, layout=layout, options=options, device=device,
        staging_dir=Path(staging_dir) if staging_dir else default_staging_dir(),
        overwrite=overwrite, keep_audio=keep_audio, on_language_mismatch=on_language_mismatch,
        translate=translate, llm_cfg=llm_cfg, llm_provider=llm_provider, secrets=[api_key],
    )


def _check_language(language: str | None) -> None:
    if language is not None and (not isinstance(language, str) or not language.strip()):
        raise InvalidInputError("language must be an ISO 639-1 code such as 'fr' or 'en', or None.")


class _StationRun:
    """One station's pipeline. Each stage records its own outcome and never raises."""

    def __init__(self, station: str | Station, *, language: str | None, whisper_model: str,
                 initial_prompt: str | None, station_overrides: dict[str, Any] | None,
                 settings: _Settings, started_at: datetime, batch_id: str | None = None) -> None:
        self.station_ref = station
        self.language = language
        self.whisper_model = whisper_model
        self.initial_prompt = initial_prompt
        self.station_overrides = station_overrides
        self.cfg = settings
        ref = station if isinstance(station, str) else station.stream_url
        self.started_at = started_at
        self.result = ProcessingResult(
            run_id=build_run_id(slugify(ref), extract_uuid(ref) or "url", started_at),
            batch_id=batch_id,
            package_version=__version__,
            started_at=started_at,
            station_input=ref,
            language_hint=language,
            requested_duration_seconds=settings.duration,
            output_dir=str(settings.layout.root),
            llm=settings.llm_cfg.model_dump(mode="json") if settings.llm_cfg else None,
        )
        self.station: Station | None = None
        self.audio: AudioCapture | None = None
        self.final: Path | None = None
        self.transcript: Transcript | None = None
        self.alive = True

    # -- bookkeeping -------------------------------------------------------
    @property
    def stages(self) -> dict[str, Any]:
        return self.result.stages

    def _ok(self, stage: str, sw: Stopwatch) -> None:
        self.stages[stage].status, self.stages[stage].seconds = "ok", sw.seconds

    def _fail(self, stage: str, exc: BaseException, sw: Stopwatch, skip: tuple[str, ...] = ()) -> None:
        self.stages[stage].status = "failed"
        self.stages[stage].seconds = sw.seconds
        self.stages[stage].error = describe_error(exc, self.cfg.secrets)
        # The traceback is formatted here and redacted, so an API key that ends up in
        # an exception message never reaches the logs.
        trace = "" if isinstance(exc, InvalidInputError) else "\n" + redact(
            "".join(traceback.format_exception(exc)).rstrip(), self.cfg.secrets)
        log.error("%s failed for %s: %s%s", stage, self.result.run_id, self.stages[stage].error, trace)
        for name in skip:
            if self.stages[name].status == "pending":
                self.stages[name].status = "skipped"
        if skip:
            self.alive = False

    # -- stages ------------------------------------------------------------
    def resolve(self) -> None:
        sw = Stopwatch()
        try:
            st = apply_station_overrides(resolve_station(self.station_ref), self.station_overrides)
            self.result.station = self.station = st
            self.result.run_id = build_run_id(slugify(st.name or "stream"), st.uuid or "url", self.started_at)
            self._ok("resolve", sw)
        except Exception as exc:  # noqa: BLE001 - isolate stages; recorded in result
            self._fail("resolve", exc, sw, skip=("capture", "transcribe", "translate", "analyse"))

    def capture(self) -> None:
        if not self.alive or self.station is None:
            return
        cfg, run_id = self.cfg, self.result.run_id
        sw = Stopwatch()
        stage_dir = cfg.staging_dir / run_id
        try:
            audio = capture_stream(self.station, cfg.duration, stage_dir, options=cfg.options,
                                   overwrite=True, filename=f"{run_id}.wav")
            if audio.path is None:  # pragma: no cover - capture_stream always returns the file
                raise CaptureError("Recording finished without a file.")
            final = publish_file(audio.path, cfg.layout.audio / audio.path.name,
                                 overwrite=cfg.overwrite, move=True)
            self.audio, self.final = audio.model_copy(update={"path": final}), final
            self.result.audio, self.result.audio_path = self.audio, final
            self._ok("capture", sw)
        except Exception as exc:  # noqa: BLE001 - isolate stages; recorded in result
            self._fail("capture", exc, sw, skip=("transcribe", "translate", "analyse"))
        finally:
            shutil.rmtree(stage_dir, ignore_errors=True)

    def transcribe(self) -> None:
        if not self.alive or self.final is None or self.audio is None:
            return
        cfg, run_id = self.cfg, self.result.run_id
        sw = Stopwatch()
        try:
            transcript = transcribe_audio(self.final, language=self.language, model=self.whisper_model,
                                          device=cfg.device, initial_prompt=self.initial_prompt,
                                          on_language_mismatch=cfg.on_language_mismatch)
            if not cfg.keep_audio:
                transcript = transcript.model_copy(update={"audio_path": None})
            tpath = cfg.layout.transcripts / f"{run_id}.json"
            transcript_file = transcript.model_dump(mode="json")
            transcript_file["audio_path"] = _relative(transcript.audio_path, cfg.layout.root)
            write_json(tpath, transcript_file, overwrite=cfg.overwrite)
            write_text(tpath.with_suffix(".txt"), transcript.text + "\n", overwrite=cfg.overwrite)
            self.transcript = transcript
            self.result.transcript, self.result.transcript_path = transcript, tpath
            self.result.language = transcript.language
            self._ok("transcribe", sw)
        except Exception as exc:  # noqa: BLE001 - isolate stages; recorded in result
            self._fail("transcribe", exc, sw, skip=("translate", "analyse"))
            if not cfg.keep_audio:
                log.warning("Transcription failed, so the recording was kept for a retry "
                            "despite keep_audio=False: %s", self.final)
            return
        if not cfg.keep_audio:  # only once the transcript is safely written
            Path(self.final).unlink(missing_ok=True)
            self.result.audio = self.audio.model_copy(update={"path": None, "deleted": True})
            self.result.audio_path = None

    def translate_and_analyse(self) -> None:
        cfg, transcript = self.cfg, self.transcript
        if not self.alive or transcript is None or self.station is None:
            return
        # translate (non-English broadcasts, LLM configured)
        skip_reason = (
            "no LLM configured" if cfg.llm_cfg is None
            else "translate=False" if not cfg.translate
            else "broadcast is already in English" if transcript.language == "en"
            else "no speech in transcript" if not transcript.text.strip()
            else None
        )
        if skip_reason:
            self.stages["translate"].status, self.stages["translate"].detail = "skipped", skip_reason
        else:
            sw = Stopwatch()
            try:
                translation = translate_transcript(transcript, config=cfg.llm_cfg, llm_provider=cfg.llm_provider)
                en_path = cfg.layout.transcripts / f"{self.result.run_id}.en.txt"
                write_text(en_path, translation.text_en + "\n", overwrite=cfg.overwrite)
                self.result.translation, self.result.translation_path = translation, en_path
                self._ok("translate", sw)
            except Exception as exc:  # noqa: BLE001 - isolate stages; recorded in result
                self._fail("translate", exc, sw)
        # analyse
        if cfg.llm_cfg is None:
            self.stages["analyse"].status, self.stages["analyse"].detail = "skipped", "no LLM configured"
            return
        sw = Stopwatch()
        try:
            analysis = analyse_transcript(transcript, config=cfg.llm_cfg, station_name=self.station.name or None,
                                          llm_provider=cfg.llm_provider)
            analysis = analysis.model_copy(update={
                "run": run_context(self.result),
                "transcript": bilingual_transcript(transcript, self.result.translation),
            })
            apath = cfg.layout.analysis / f"{self.result.run_id}.json"
            write_json(apath, analysis, overwrite=cfg.overwrite)
            self.result.analysis, self.result.analysis_path = analysis, apath
            self._ok("analyse", sw)
        except Exception as exc:  # noqa: BLE001 - isolate stages; recorded in result
            self._fail("analyse", exc, sw)

    def persist(self) -> ProcessingResult:
        return _persist(self.result, self.cfg.layout, self.cfg.overwrite, self.cfg.secrets)


def process_station(
    station: str | Station,
    language: str | None,
    duration_seconds: int = 600,
    output_dir: str | Path = "output",
    whisper_model: str = "small",
    llm: LLMConfig | dict[str, Any] | None = None,
    *,
    device: str = "auto",
    capture_options: CaptureOptions | None = None,
    staging_dir: str | Path | None = None,
    overwrite: bool = False,
    keep_audio: bool = True,
    initial_prompt: str | None = None,
    on_language_mismatch: LanguageMismatchPolicy = "use_detected",
    translate: bool = True,
    station_overrides: dict[str, Any] | None = None,
) -> ProcessingResult:
    """Resolve -> record -> transcribe -> translate -> analyse -> persist, for one station.

    For several stations recorded at the same time, use :func:`process_stations`.

    ``station_overrides`` corrects or adds station metadata after resolution, e.g.
    ``{"latitude": 48.85, "longitude": 2.27}`` for a station Radio Browser has no
    coordinates for (``location_precision`` then becomes ``"station"`` unless given).

    With an ``llm`` configured, a non-English transcript is also translated into
    English segment by segment (``translate=False`` turns this off) and the analysis
    file carries the original and English transcript side by side.

    Stage failures do not raise: they are recorded in ``result.stages`` and every
    artefact produced before the failure is kept. A JSON manifest is always written
    to ``<output_dir>/manifests/``. Set-up problems raise immediately, before any
    recording starts: invalid arguments, a missing Whisper or LLM package, an unknown
    language code, Whisper model or LLM provider, or a missing API key or Azure
    endpoint. Pass ``llm=None`` for transcription only.

    ``keep_audio=False`` deletes the recording once the transcript is saved; if
    transcription fails, the recording is kept so the run can be retried.

    ``language`` is checked against the audio before transcribing; a clearly wrong
    hint is replaced by the detected language unless ``on_language_mismatch`` says
    otherwise (see :func:`transcribe_audio`). ``whisper_model`` defaults to ``"small"``;
    use ``"turbo"`` for less widely spoken languages such as Swahili (see the README's
    "Choosing a Whisper model").
    """
    _check_language(language)
    # Fail fast: check everything that does not need the audio before recording it.
    language = check_transcription_ready(language, whisper_model, device, on_language_mismatch)
    settings = _prepare(
        duration_seconds=duration_seconds, output_dir=output_dir, llm=llm, device=device,
        capture_options=capture_options, staging_dir=staging_dir, overwrite=overwrite,
        keep_audio=keep_audio, on_language_mismatch=on_language_mismatch, translate=translate,
    )
    run = _StationRun(station, language=language, whisper_model=whisper_model, initial_prompt=initial_prompt,
                      station_overrides=station_overrides, settings=settings, started_at=utc_now())
    run.resolve()
    run.capture()
    run.transcribe()
    run.translate_and_analyse()
    return run.persist()


#: Most stations one :func:`process_stations` call records at once.
MAX_STATIONS = 5
#: Default number of stations whose translation/analysis can be with the LLM at once.
DEFAULT_LLM_WORKERS = 4


def _finish_time(run: _StationRun, default: datetime) -> datetime:
    audio = run.result.audio
    return audio.finished_at if audio is not None else default


def _station_key(ref: str | Station) -> str:
    if isinstance(ref, Station):
        return ref.uuid or ref.stream_url
    _kind, value = parse_station_reference(ref)
    return value


def process_stations(
    stations: Sequence[str | Station | StationRequest | dict[str, Any]],
    duration_seconds: int = 600,
    output_dir: str | Path = "output",
    whisper_model: str = "small",
    llm: LLMConfig | dict[str, Any] | None = None,
    *,
    device: str = "auto",
    capture_options: CaptureOptions | None = None,
    staging_dir: str | Path | None = None,
    overwrite: bool = False,
    keep_audio: bool = True,
    on_language_mismatch: LanguageMismatchPolicy = "use_detected",
    translate: bool = True,
    max_llm_workers: int = DEFAULT_LLM_WORKERS,
) -> BatchResult:
    """Record up to five stations at the same moment, then transcribe and analyse each one.

    ``stations`` takes station references, :class:`StationRequest` objects or dicts with
    the same fields, so each station can have its own ``language``, ``whisper_model``
    (e.g. ``"turbo"`` for Swahili while the rest use ``"small"``), ``initial_prompt``
    and ``station_overrides``. Everything else is shared.

    How it runs:

    * Every station's set-up is checked first; any mistake raises before recording.
    * All recordings start together, in background threads.
    * Each recording is transcribed as soon as it finishes, one at a time, so Whisper
      never competes with itself. Among recordings that are ready, one that uses the
      Whisper model already loaded goes first, and a model is released as soon as no
      remaining station needs it. Normally only one model is in memory; two are held
      only while a station that needs the first is still recording.
    * Translation and analysis run in a pool of ``max_llm_workers`` threads while the
      next recording is transcribed.

    Each station gets the same files and manifest as :func:`process_station`, and all
    of them share a ``batch_id``. A failing station never affects the others. The
    scheduling of runs (e.g. at XX:00) is left to cron, Airflow or similar.
    """
    if isinstance(stations, (str, Station)) or not isinstance(stations, Sequence):
        raise InvalidInputError("stations must be a list of stations.")
    if not 1 <= len(stations) <= MAX_STATIONS:
        raise InvalidInputError(f"process_stations takes 1 to {MAX_STATIONS} stations, got {len(stations)}.")
    if max_llm_workers < 1:
        raise InvalidInputError("max_llm_workers must be at least 1.")
    requests: list[StationRequest] = []
    for item in stations:
        try:
            if isinstance(item, StationRequest):
                requests.append(item)
            elif isinstance(item, dict):
                requests.append(StationRequest.model_validate(item))
            else:
                requests.append(StationRequest(station=item))
        except ValidationError as exc:
            raise InvalidInputError(f"Invalid station entry {item!r}: {exc.errors(include_input=False)}") from exc
    keys = [_station_key(r.station) for r in requests]
    duplicates = sorted({k for k in keys if keys.count(k) > 1})
    if duplicates:
        raise InvalidInputError(f"The same station is listed more than once: {', '.join(duplicates)}")

    # Fail fast: every station's language and Whisper model, and the LLM client.
    plans = []
    for req in requests:
        _check_language(req.language)
        model = req.whisper_model or whisper_model
        plans.append((req, check_transcription_ready(req.language, model, device, on_language_mismatch), model))
    settings = _prepare(
        duration_seconds=duration_seconds, output_dir=output_dir, llm=llm, device=device,
        capture_options=capture_options, staging_dir=staging_dir, overwrite=overwrite,
        keep_audio=keep_audio, on_language_mismatch=on_language_mismatch, translate=translate,
    )

    started_at = utc_now()
    batch = BatchResult(batch_id=f"batch_{timestamp_for_filename(started_at)}", started_at=started_at)
    runs = [
        _StationRun(req.station, language=language, whisper_model=model, initial_prompt=req.initial_prompt,
                    station_overrides=req.station_overrides, settings=settings, started_at=started_at,
                    batch_id=batch.batch_id)
        for req, language, model in plans
    ]
    for run in runs:  # quick API lookups, so that every recording can start together
        run.resolve()
    log.info("Recording %d stations for %ss (batch %s)", sum(r.alive for r in runs), settings.duration,
             batch.batch_id)

    finished: dict[_StationRun, ProcessingResult] = {}
    with ThreadPoolExecutor(max_workers=len(runs), thread_name_prefix="listen-in-capture") as capture_pool, \
         ThreadPoolExecutor(max_workers=max_llm_workers, thread_name_prefix="listen-in-llm") as llm_pool:
        captures = {capture_pool.submit(run.capture): run for run in runs}
        waiting: set[Future[None]] = set(captures)
        ready: list[_StationRun] = []
        llm_jobs: dict[Future[ProcessingResult], _StationRun] = {}
        loaded: str | None = None

        def finish(run: _StationRun) -> ProcessingResult:
            run.translate_and_analyse()
            return run.persist()

        while waiting or ready:
            done, waiting = wait(waiting, timeout=None if not ready else 0, return_when=FIRST_COMPLETED)
            for capture_done in sorted(done, key=lambda f: _finish_time(captures[f], started_at)):
                capture_done.result()  # stages record their own errors; this only surfaces bugs
                ready.append(captures[capture_done])
            if not ready:
                continue
            # Same model first, so a different model is only loaded when needed.
            run = next((r for r in ready if r.whisper_model == loaded), ready[0])
            ready.remove(run)
            if run.alive and loaded is not None and run.whisper_model != loaded:
                still_needed = {r.whisper_model for r in ready} | {captures[f].whisper_model for f in waiting}
                if loaded not in still_needed:
                    release_whisper_model(loaded)
            run.transcribe()
            if run.alive:
                loaded = run.whisper_model
            llm_jobs[llm_pool.submit(finish, run)] = run
        for job, job_run in llm_jobs.items():
            finished[job_run] = job.result()

    batch.results = [finished[run] for run in runs]
    batch.finished_at = utc_now()
    return batch


def run_context(result: ProcessingResult) -> RunContext:
    """Run, time and station context stored at the top of the analysis file."""
    st = result.station
    return RunContext(
        run_id=result.run_id,
        batch_id=result.batch_id,
        recorded_at=result.audio.started_at if result.audio else None,
        duration_seconds=result.audio.duration_seconds if result.audio else None,
        language=result.language,
        station=RunStation(**st.model_dump(include=set(RunStation.model_fields))) if st else None,
        package_version=result.package_version,
    )


#: Left out of the manifest. audio.station repeats the top-level station and
#: analysis.transcript repeats transcript + translation; output_dir and ffmpeg_path are
#: details of the machine the run happened on.
MANIFEST_EXCLUDE: dict[str, Any] = {
    "output_dir": True,
    "audio": {"station", "ffmpeg_path"},
    "analysis": {"transcript"},
}
#: LLM settings left out of the manifest: the endpoint URL can identify a private resource.
MANIFEST_LLM_EXCLUDE = ("base_url",)
_PATH_FIELDS = ("audio_path", "transcript_path", "translation_path", "analysis_path", "manifest_path")


def _relative(value: Any, root: Path) -> Any:
    """``value`` as a POSIX path relative to ``root`` when it lies inside it."""
    if not value:
        return None if value is None else value
    try:
        return Path(value).resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return str(value)


def manifest_dict(result: ProcessingResult, output_root: Path) -> dict[str, Any]:
    """The manifest JSON: nothing repeated, no machine details, paths relative to ``output_dir``."""
    data = result.model_dump(mode="json", exclude=MANIFEST_EXCLUDE)
    for key in _PATH_FIELDS:
        data[key] = _relative(data.get(key), output_root)
    if data.get("audio"):
        data["audio"]["path"] = _relative(data["audio"].get("path"), output_root)
    if data.get("transcript"):
        data["transcript"]["audio_path"] = _relative(data["transcript"].get("audio_path"), output_root)
    if data.get("llm"):
        for key in MANIFEST_LLM_EXCLUDE:
            data["llm"].pop(key, None)
    return data


def apply_station_overrides(station: Station, overrides: dict[str, Any] | None) -> Station:
    """Return ``station`` with ``overrides`` applied and validated.

    Supplying both coordinates without a ``location_precision`` marks them as exact
    (``"station"``).
    """
    if not overrides:
        return station
    data = {**station.model_dump(), **overrides}
    if ("latitude" in overrides or "longitude" in overrides) and "location_precision" not in overrides:
        has_both = data.get("latitude") is not None and data.get("longitude") is not None
        data["location_precision"] = "station" if has_both else None
    try:
        return Station.model_validate(data)
    except ValidationError as exc:
        raise InvalidInputError(f"Invalid station_overrides: {exc.errors(include_input=False)}") from exc


def bilingual_transcript(transcript: Transcript, translation: Translation | None) -> BilingualTranscript:
    """Original transcript with its English text, segment by segment.

    English broadcasts reuse the original as ``text_en`` (``translated=False``); if a
    translation is missing, ``text_en`` is ``None``.
    """
    if translation is not None:
        return BilingualTranscript(language=transcript.language, translated=True, text=transcript.text,
                                   text_en=translation.text_en, segments=translation.segments)
    english = transcript.language == "en"
    return BilingualTranscript(
        language=transcript.language,
        translated=False,
        text=transcript.text,
        text_en=transcript.text if english else None,
        segments=[
            TranslatedSegment(id=s.id, start=s.start, end=s.end, text=s.text,
                              text_en=s.text if english else None)
            for s in transcript.segments
        ],
    )


def _persist(result: ProcessingResult, layout: OutputLayout, overwrite: bool,
             secrets: list[str | None]) -> ProcessingResult:
    sw = Stopwatch()
    mpath = layout.manifests / f"{result.run_id}.json"
    result.finished_at = utc_now()
    result.manifest_path = mpath
    result.stages["persist"].status = "ok"
    try:
        write_json(mpath, manifest_dict(result, layout.root), overwrite=overwrite)
        # Record the measured write time, then refresh the (small) manifest with it.
        result.stages["persist"].seconds = sw.seconds
        write_json(mpath, manifest_dict(result, layout.root), overwrite=True)
    except Exception as exc:  # noqa: BLE001 - isolate stages; recorded in result
        result.stages["persist"].status = "failed"
        result.stages["persist"].error = describe_error(exc, secrets)
        result.manifest_path = None
        log.error("Could not write manifest %s: %s", mpath, result.stages["persist"].error)
    return result
