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
from pathlib import Path
from typing import Any

from pydantic import SecretStr, ValidationError

from ._version import __version__
from .analysis import LLMProvider, analyse_transcript, coerce_llm_config, get_provider, resolve_api_key
from .capture import capture_stream, probe_stream
from .exceptions import CaptureError, InvalidInputError
from .models import (
    BilingualTranscript,
    CaptureOptions,
    LLMConfig,
    ProcessingResult,
    RunContext,
    RunStation,
    Station,
    Transcript,
    TranslatedSegment,
    Translation,
)
from .radio_browser import resolve_station
from .storage import OutputLayout, default_staging_dir, publish_file, write_json, write_text
from .transcription import LanguageMismatchPolicy, check_transcription_ready, transcribe_audio
from .translation import translate_transcript
from .utils import (
    Stopwatch,
    build_run_id,
    describe_error,
    extract_uuid,
    redact,
    slugify,
    utc_now,
    validate_duration,
)

log = logging.getLogger(__name__)

__all__ = [
    "analyse_transcript",
    "capture_stream",
    "probe_stream",
    "process_station",
    "resolve_station",
    "transcribe_audio",
    "translate_transcript",
]


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
    options = capture_options or CaptureOptions()
    duration = validate_duration(duration_seconds, options.max_duration_seconds)
    if language is not None and (not isinstance(language, str) or not language.strip()):
        raise InvalidInputError(
            "language must be an ISO 639-1 code such as 'fr' or 'en', or None."
        )
    # Fail fast: check everything that does not need the audio before recording it.
    language = check_transcription_ready(language, whisper_model, device, on_language_mismatch)
    layout = OutputLayout.create(output_dir)

    llm_cfg: LLMConfig | None = None
    api_key: str | None = None
    llm_provider: LLMProvider | None = None
    if llm is not None:
        llm_cfg = coerce_llm_config(llm)
        api_key = resolve_api_key(llm_cfg)
        llm_cfg = llm_cfg.model_copy(update={"api_key": SecretStr(api_key)})
        llm_provider = get_provider(llm_cfg, api_key)  # unknown provider, missing SDK or endpoint
    secrets = [api_key]

    started_at = utc_now()
    ref = station if isinstance(station, str) else station.stream_url
    result = ProcessingResult(
        run_id=build_run_id(slugify(ref), extract_uuid(ref) or "url", started_at),
        package_version=__version__,
        started_at=started_at,
        station_input=ref,
        language_hint=language,
        requested_duration_seconds=duration,
        output_dir=str(layout.root),
        llm=llm_cfg.model_dump(mode="json") if llm_cfg else None,
    )
    stages = result.stages

    def fail(stage: str, exc: BaseException, sw: Stopwatch) -> None:
        stages[stage].status = "failed"
        stages[stage].seconds = sw.seconds
        stages[stage].error = describe_error(exc, secrets)
        # The traceback is formatted here and redacted, so an API key that ends up in
        # an exception message never reaches the logs.
        trace = "" if isinstance(exc, InvalidInputError) else "\n" + redact(
            "".join(traceback.format_exception(exc)).rstrip(), secrets)
        log.error("%s failed for %s: %s%s", stage, result.run_id, stages[stage].error, trace)

    def skip_from(stage_names: tuple[str, ...]) -> None:
        for s in stage_names:
            if stages[s].status == "pending":
                stages[s].status = "skipped"

    # 1. resolve -------------------------------------------------------------
    sw = Stopwatch()
    try:
        st = apply_station_overrides(resolve_station(station), station_overrides)
        result.station = st
        result.run_id = build_run_id(slugify(st.name or "stream"), st.uuid or "url", started_at)
        stages["resolve"].status, stages["resolve"].seconds = "ok", sw.seconds
    except Exception as exc:  # noqa: BLE001 - isolate stages; recorded in result
        fail("resolve", exc, sw)
        skip_from(("capture", "transcribe", "translate", "analyse"))
        return _persist(result, layout, overwrite, secrets)

    # 2. capture (to local staging, then publish) ----------------------------
    sw = Stopwatch()
    stage_dir = Path(staging_dir) if staging_dir else default_staging_dir()
    try:
        audio = capture_stream(st, duration, stage_dir / result.run_id, options=options,
                               overwrite=True, filename=f"{result.run_id}.wav")
        staged = audio.path
        if staged is None:  # pragma: no cover - capture_stream always returns the file
            raise CaptureError("Recording finished without a file.")
        final = publish_file(staged, layout.audio / staged.name, overwrite=overwrite, move=True)
        shutil.rmtree(stage_dir / result.run_id, ignore_errors=True)
        audio = audio.model_copy(update={"path": final})
        result.audio, result.audio_path = audio, final
        stages["capture"].status, stages["capture"].seconds = "ok", sw.seconds
    except Exception as exc:  # noqa: BLE001 - isolate stages; recorded in result
        fail("capture", exc, sw)
        shutil.rmtree(stage_dir / result.run_id, ignore_errors=True)
        skip_from(("transcribe", "translate", "analyse"))
        return _persist(result, layout, overwrite, secrets)

    # 3. transcribe ----------------------------------------------------------
    sw = Stopwatch()
    try:
        transcript = transcribe_audio(final, language=language, model=whisper_model,
                                      device=device, initial_prompt=initial_prompt,
                                      on_language_mismatch=on_language_mismatch)
        if not keep_audio:
            transcript = transcript.model_copy(update={"audio_path": None})
        tpath = layout.transcripts / f"{result.run_id}.json"
        transcript_file = transcript.model_dump(mode="json")
        transcript_file["audio_path"] = _relative(transcript.audio_path, layout.root)
        write_json(tpath, transcript_file, overwrite=overwrite)
        write_text(tpath.with_suffix(".txt"), transcript.text + "\n", overwrite=overwrite)
        result.transcript, result.transcript_path = transcript, tpath
        result.language = transcript.language
        stages["transcribe"].status, stages["transcribe"].seconds = "ok", sw.seconds
    except Exception as exc:  # noqa: BLE001 - isolate stages; recorded in result
        fail("transcribe", exc, sw)
        skip_from(("translate", "analyse"))
        if not keep_audio:
            log.warning("Transcription failed, so the recording was kept for a retry "
                        "despite keep_audio=False: %s", final)
        return _persist(result, layout, overwrite, secrets)
    if not keep_audio:  # only once the transcript is safely written
        Path(final).unlink(missing_ok=True)
        result.audio = audio.model_copy(update={"path": None, "deleted": True})
        result.audio_path = None

    # 4. translate (optional: non-English broadcasts, LLM configured) ---------
    skip_reason = (
        "no LLM configured" if llm_cfg is None
        else "translate=False" if not translate
        else "broadcast is already in English" if transcript.language == "en"
        else "no speech in transcript" if not transcript.text.strip()
        else None
    )
    if skip_reason:
        stages["translate"].status, stages["translate"].detail = "skipped", skip_reason
    else:
        sw = Stopwatch()
        try:
            translation = translate_transcript(transcript, config=llm_cfg, llm_provider=llm_provider)
            en_path = layout.transcripts / f"{result.run_id}.en.txt"
            write_text(en_path, translation.text_en + "\n", overwrite=overwrite)
            result.translation, result.translation_path = translation, en_path
            stages["translate"].status, stages["translate"].seconds = "ok", sw.seconds
        except Exception as exc:  # noqa: BLE001 - isolate stages; recorded in result
            fail("translate", exc, sw)

    # 5. analyse (optional) --------------------------------------------------
    if llm_cfg is None:
        stages["analyse"].status, stages["analyse"].detail = "skipped", "no LLM configured"
    else:
        sw = Stopwatch()
        try:
            analysis = analyse_transcript(transcript, config=llm_cfg, station_name=st.name or None,
                                          llm_provider=llm_provider)
            analysis = analysis.model_copy(update={
                "run": run_context(result),
                "transcript": bilingual_transcript(transcript, result.translation),
            })
            apath = layout.analysis / f"{result.run_id}.json"
            write_json(apath, analysis, overwrite=overwrite)
            result.analysis, result.analysis_path = analysis, apath
            stages["analyse"].status, stages["analyse"].seconds = "ok", sw.seconds
        except Exception as exc:  # noqa: BLE001 - isolate stages; recorded in result
            fail("analyse", exc, sw)

    return _persist(result, layout, overwrite, secrets)


def run_context(result: ProcessingResult) -> RunContext:
    """Run, time and station context stored at the top of the analysis file."""
    st = result.station
    return RunContext(
        run_id=result.run_id,
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
