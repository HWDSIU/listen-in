"""Public functional API.

Individual steps can be called directly (``resolve_station``, ``capture_stream``,
``transcribe_audio``, ``analyse_transcript``) or chained with ``process_station``.
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
from .models import CaptureOptions, LLMConfig, ProcessingResult, Station
from .radio_browser import resolve_station
from .storage import OutputLayout, default_staging_dir, publish_file, write_json, write_text
from .transcription import transcribe_audio
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
) -> ProcessingResult:
    """Resolve -> record -> transcribe -> (optionally) analyse -> persist, for one station.

    Stage failures do not raise: they are recorded in ``result.stages`` and every
    artefact produced before the failure is kept. A JSON manifest is always written
    to ``<output_dir>/manifests/``. Invalid arguments raise immediately, before any
    recording starts. Pass ``llm=None`` for transcription only.
    """
    options = capture_options or CaptureOptions()
    duration = validate_duration(duration_seconds, options.max_duration_seconds)
    if language is not None and (not isinstance(language, str) or not language.strip()):
        raise InvalidInputError("language must be an ISO 639-1 code such as 'fr' or 'en', or None.")
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
        skip_from(("capture", "transcribe", "analyse"))
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
        skip_from(("transcribe", "analyse"))
        return _persist(result, layout, overwrite, secrets)

    # 3. transcribe ----------------------------------------------------------
    sw = Stopwatch()
    try:
        transcript = transcribe_audio(final, language=language, model=whisper_model,
                                      device=device, initial_prompt=initial_prompt)
        tpath = layout.transcripts / f"{result.run_id}.json"
        write_json(tpath, transcript, overwrite=overwrite)
        write_text(tpath.with_suffix(".txt"), transcript.text + "\n", overwrite=overwrite)
        result.transcript, result.transcript_path = transcript, tpath
        stages["transcribe"].status, stages["transcribe"].seconds = "ok", sw.seconds
    except Exception as exc:  # noqa: BLE001 - isolate stages; recorded in result
        fail("transcribe", exc, sw)
        skip_from(("analyse",))
        return _persist(result, layout, overwrite, secrets)
    finally:
        if not keep_audio and result.audio_path:
            Path(result.audio_path).unlink(missing_ok=True)
            result.audio_path = None

    # 4. analyse (optional) --------------------------------------------------
    if llm_cfg is None:
        stages["analyse"].status = "skipped"
    else:
        sw = Stopwatch()
        try:
            analysis = analyse_transcript(transcript, config=llm_cfg, station_name=st.name or None)
            apath = layout.analysis / f"{result.run_id}.json"
            write_json(apath, analysis, overwrite=overwrite)
            result.analysis, result.analysis_path = analysis, apath
            stages["analyse"].status, stages["analyse"].seconds = "ok", sw.seconds
        except Exception as exc:  # noqa: BLE001 - isolate stages; recorded in result
            fail("analyse", exc, sw)

    return _persist(result, layout, overwrite, secrets)


def _persist(result: ProcessingResult, layout: OutputLayout, overwrite: bool,
             secrets: list[str | None]) -> ProcessingResult:
    sw = Stopwatch()
    mpath = layout.manifests / f"{result.run_id}.json"
    result.finished_at = utc_now()
    result.manifest_path = mpath
    result.stages["persist"].status = "ok"
    try:
        result.stages["persist"].seconds = sw.seconds
        write_json(mpath, result, overwrite=overwrite)
    except Exception as exc:  # noqa: BLE001 - isolate stages; recorded in result
        result.stages["persist"].status = "failed"
        result.stages["persist"].error = describe_error(exc, secrets)
        result.manifest_path = None
        log.error("Could not write manifest %s: %s", mpath, result.stages["persist"].error)
    return result
