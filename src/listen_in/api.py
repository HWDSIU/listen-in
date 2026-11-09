"""Public functional API.

Individual steps can be called directly (``resolve_station``, ``capture_stream``,
``transcribe_audio``, ``translate_transcript``, ``analyse_transcript``) or chained
with ``process_station``.
No function here starts a scheduler, a Spark session or a background thread.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from ._version import __version__
from .analysis import analyse_transcript, coerce_llm_config, resolve_api_key
from .capture import capture_stream, probe_stream
from .exceptions import InvalidInputError
from .models import (
    BilingualTranscript,
    CaptureOptions,
    LLMConfig,
    ProcessingResult,
    Station,
    Transcript,
    TranslatedSegment,
    Translation,
)
from .radio_browser import resolve_station
from .storage import OutputLayout, default_staging_dir, publish_file, write_json, write_text
from .transcription import LanguageMismatchPolicy, transcribe_audio
from .translation import translate_transcript
from .utils import (
    Stopwatch,
    build_run_id,
    describe_error,
    extract_uuid,
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
) -> ProcessingResult:
    """Resolve -> record -> transcribe -> translate -> analyse -> persist, for one station.

    With an ``llm`` configured, a non-English transcript is also translated into
    English segment by segment (``translate=False`` turns this off) and the analysis
    file carries the original and English transcript side by side.

    Stage failures do not raise: they are recorded in ``result.stages`` and every
    artefact produced before the failure is kept. A JSON manifest is always written
    to ``<output_dir>/manifests/``. Invalid arguments raise immediately, before any
    recording starts. Pass ``llm=None`` for transcription only.

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
    language = language.strip().lower() if language else None
    layout = OutputLayout.create(output_dir)

    llm_cfg: LLMConfig | None = None
    api_key: str | None = None
    if llm is not None:
        llm_cfg = coerce_llm_config(llm)
        api_key = resolve_api_key(llm_cfg)  # fail fast, before a long recording
        llm_cfg = llm_cfg.model_copy(update={"api_key": SecretStr(api_key)})
    secrets = [api_key]

    started_at = utc_now()
    ref = station if isinstance(station, str) else station.stream_url
    result = ProcessingResult(
        run_id=build_run_id(slugify(ref), extract_uuid(ref) or "url", started_at),
        package_version=__version__,
        started_at=started_at,
        station_input=ref,
        language=language or "auto",
        requested_duration_seconds=duration,
        output_dir=str(layout.root),
        llm=llm_cfg.model_dump(mode="json") if llm_cfg else None,
    )
    stages = result.stages

    def fail(stage: str, exc: BaseException, sw: Stopwatch) -> None:
        stages[stage].status = "failed"
        stages[stage].seconds = sw.seconds
        stages[stage].error = describe_error(exc, secrets)
        log.error("%s failed for %s: %s", stage, result.run_id, stages[stage].error,
                  exc_info=not isinstance(exc, InvalidInputError))

    def skip_from(stage_names: tuple[str, ...]) -> None:
        for s in stage_names:
            if stages[s].status == "pending":
                stages[s].status = "skipped"

    # 1. resolve -------------------------------------------------------------
    sw = Stopwatch()
    try:
        st = resolve_station(station)
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
        tpath = layout.transcripts / f"{result.run_id}.json"
        write_json(tpath, transcript, overwrite=overwrite)
        write_text(tpath.with_suffix(".txt"), transcript.text + "\n", overwrite=overwrite)
        result.transcript, result.transcript_path = transcript, tpath
        stages["transcribe"].status, stages["transcribe"].seconds = "ok", sw.seconds
    except Exception as exc:  # noqa: BLE001 - isolate stages; recorded in result
        fail("transcribe", exc, sw)
        skip_from(("translate", "analyse"))
        return _persist(result, layout, overwrite, secrets)
    finally:
        if not keep_audio and result.audio_path:
            Path(result.audio_path).unlink(missing_ok=True)
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
            translation = translate_transcript(transcript, config=llm_cfg)
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
            analysis = analyse_transcript(transcript, config=llm_cfg, station_name=st.name or None)
            analysis = analysis.model_copy(
                update={"transcript": bilingual_transcript(transcript, result.translation)}
            )
            apath = layout.analysis / f"{result.run_id}.json"
            write_json(apath, analysis, overwrite=overwrite)
            result.analysis, result.analysis_path = analysis, apath
            stages["analyse"].status, stages["analyse"].seconds = "ok", sw.seconds
        except Exception as exc:  # noqa: BLE001 - isolate stages; recorded in result
            fail("analyse", exc, sw)

    return _persist(result, layout, overwrite, secrets)


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
        write_json(mpath, result, overwrite=overwrite)
        # Record the measured write time, then refresh the (small) manifest with it.
        result.stages["persist"].seconds = sw.seconds
        write_json(mpath, result, overwrite=True)
    except Exception as exc:  # noqa: BLE001 - isolate stages; recorded in result
        result.stages["persist"].status = "failed"
        result.stages["persist"].error = describe_error(exc, secrets)
        result.manifest_path = None
        log.error("Could not write manifest %s: %s", mpath, result.stages["persist"].error)
    return result
