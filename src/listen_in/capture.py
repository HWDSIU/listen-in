"""Record a stream to mono 16 kHz WAV with ffmpeg (invoked without a shell)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import wave
from pathlib import Path

from .exceptions import (
    CaptureError,
    FFmpegNotFoundError,
    InvalidInputError,
    OutputExistsError,
    StreamUnavailableError,
)
from .models import AudioCapture, CaptureOptions, Station
from .radio_browser import resolve_station
from .utils import (
    USER_AGENT,
    build_run_id,
    slugify,
    utc_now,
    validate_duration,
    validate_output_dir,
    validate_stream_url,
)

log = logging.getLogger(__name__)

FFMPEG_ENV = "LISTEN_IN_FFMPEG"


# ------------------------------------------------------------------ locate ffmpeg
def _check_executable(candidate: str) -> str | None:
    resolved = shutil.which(candidate)
    if resolved:
        return resolved
    p = Path(candidate).expanduser()
    if p.is_file() and os.access(p, os.X_OK):
        return str(p)
    return None


def find_ffmpeg(explicit: str | None = None) -> str:
    """Return a usable ffmpeg executable path or raise :class:`FFmpegNotFoundError`.

    Search order: ``explicit`` argument, ``$LISTEN_IN_FFMPEG``, ``ffmpeg`` on PATH,
    then the static binary from the optional ``imageio-ffmpeg`` package.
    """
    for label, candidate in (("ffmpeg_path", explicit), (FFMPEG_ENV, os.environ.get(FFMPEG_ENV))):
        if candidate:
            found = _check_executable(candidate)
            if not found:
                raise FFmpegNotFoundError(f"{label}={candidate!r} is not an executable file.")
            return found
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg  # optional extra: listen-in[ffmpeg]

        bundled = imageio_ffmpeg.get_ffmpeg_exe()
        if bundled and _check_executable(bundled):
            return bundled
    except (ImportError, RuntimeError):
        pass
    raise FFmpegNotFoundError()


def find_ffprobe(ffmpeg_path: str) -> str | None:
    """ffprobe next to ffmpeg, else on PATH. Optional: probing falls back to ffmpeg."""
    sibling = Path(ffmpeg_path).with_name("ffprobe" + (".exe" if os.name == "nt" else ""))
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return str(sibling)
    return shutil.which("ffprobe")


# ---------------------------------------------------------------- command builders
def _input_args(url: str, options: CaptureOptions) -> list[str]:
    timeout_us = str(int(options.connect_timeout_seconds * 1_000_000))
    args = ["-user_agent", options.user_agent or USER_AGENT, "-rw_timeout", timeout_us]
    if options.reconnect:
        args += [
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_on_network_error", "1",
            "-reconnect_delay_max", "5",
        ]
    return [*args, *options.extra_input_args, "-i", validate_stream_url(url)]


def build_capture_command(
    ffmpeg: str, url: str, output_path: Path, duration_seconds: int, options: CaptureOptions
) -> list[str]:
    """Argument list for recording. Every value is a separate argv item (no shell)."""
    return [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error",
        *_input_args(url, options),
        "-t", str(int(duration_seconds)),
        "-vn", "-map_metadata", "-1",
        "-ac", str(options.channels),
        "-ar", str(options.sample_rate),
        "-c:a", "pcm_s16le",
        "-f", "wav",
        "-y", str(output_path),
    ]


def build_probe_command(ffmpeg: str, ffprobe: str | None, url: str, options: CaptureOptions) -> list[str]:
    if ffprobe:
        return [
            ffprobe, "-hide_banner", "-v", "error",
            *_input_args(url, options),
            "-show_entries", "stream=codec_name,sample_rate,channels,bit_rate",
            "-of", "json",
        ]
    return [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "info",
        *_input_args(url, options),
        "-t", "3", "-vn", "-f", "null", "-",
    ]


def _tail(text: str, n: int = 800) -> str:
    text = (text or "").strip()
    return text[-n:] if len(text) > n else text


# ------------------------------------------------------------------------ probe
def probe_stream(station: str | Station, options: CaptureOptions | None = None) -> dict:
    """Open the stream briefly and report codec details. Raises StreamUnavailableError."""
    options = options or CaptureOptions()
    st = resolve_station(station)
    ffmpeg = find_ffmpeg(options.ffmpeg_path)
    ffprobe = find_ffprobe(ffmpeg)
    cmd = build_probe_command(ffmpeg, ffprobe, st.stream_url, options)
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
            timeout=options.connect_timeout_seconds + 30, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise StreamUnavailableError(f"Timed out probing {st.stream_url}") from exc
    if p.returncode != 0:
        raise StreamUnavailableError(f"Could not open {st.stream_url}: {_tail(p.stderr)}")
    if ffprobe:
        streams = json.loads(p.stdout or "{}").get("streams", [])
        info = streams[0] if streams else {}
        return {"tool": "ffprobe", **info}
    lines = [ln.strip() for ln in p.stderr.splitlines() if "Audio:" in ln]
    return {"tool": "ffmpeg", "stream": lines[0] if lines else ""}


# ----------------------------------------------------------------------- capture
def _wav_info(path: Path) -> tuple[float, int, int]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        return w.getnframes() / float(rate), rate, w.getnchannels()


def capture_stream(
    station: str | Station,
    duration_seconds: int = 60,
    output_dir: str | Path = "output/audio",
    *,
    options: CaptureOptions | None = None,
    overwrite: bool = False,
    filename: str | None = None,
) -> AudioCapture:
    """Record ``duration_seconds`` of a stream to a mono 16 kHz WAV file.

    Audio is written to ``<name>.partial.wav`` and atomically renamed on success.
    An existing file is never replaced unless ``overwrite=True``.
    """
    options = options or CaptureOptions()
    duration = validate_duration(duration_seconds, options.max_duration_seconds)
    out_dir = validate_output_dir(output_dir)
    st = resolve_station(station)
    ffmpeg = find_ffmpeg(options.ffmpeg_path)

    started_at = utc_now()
    if filename is None:
        stem = build_run_id(slugify(st.display_name if st.name else "stream"),
                            st.uuid or "url", started_at)
        filename = f"{stem}.wav"
    if Path(filename).name != filename or not filename.endswith(".wav"):
        raise InvalidInputError("filename must be a bare file name ending in .wav")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    if out_path.exists() and not overwrite:
        raise OutputExistsError(f"{out_path} already exists (pass overwrite=True to replace it).")
    partial = out_path.with_name(out_path.stem + ".partial.wav")
    partial.unlink(missing_ok=True)

    cmd = build_capture_command(ffmpeg, st.stream_url, partial, duration, options)
    timeout = duration + options.connect_timeout_seconds + options.process_timeout_grace_seconds
    log.info("Recording %ss from %s", duration, st.display_name)
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        partial.unlink(missing_ok=True)
        raise CaptureError(f"ffmpeg did not finish within {timeout:.0f}s; recording abandoned.") from exc
    except OSError as exc:
        raise FFmpegNotFoundError(f"Could not execute {ffmpeg}: {exc}") from exc

    if p.returncode != 0 or not partial.exists() or partial.stat().st_size <= 44:
        partial.unlink(missing_ok=True)
        raise StreamUnavailableError(
            f"ffmpeg exited with code {p.returncode} recording {st.stream_url}: {_tail(p.stderr)}"
        )
    try:
        seconds, rate, channels = _wav_info(partial)
    except (wave.Error, EOFError) as exc:
        partial.unlink(missing_ok=True)
        raise CaptureError(f"Recording is not a readable WAV file: {exc}") from exc
    if seconds < duration * options.min_duration_ratio:
        partial.unlink(missing_ok=True)
        raise CaptureError(
            f"Recording too short: {seconds:.1f}s of {duration}s requested "
            "(the stream probably dropped)."
        )
    os.replace(partial, out_path)
    return AudioCapture(
        path=out_path,
        station=st,
        started_at=started_at,
        finished_at=utc_now(),
        requested_duration_seconds=duration,
        duration_seconds=round(seconds, 3),
        sample_rate=rate,
        channels=channels,
        size_bytes=out_path.stat().st_size,
        ffmpeg_path=ffmpeg,
    )
