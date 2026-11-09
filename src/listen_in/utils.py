"""Small helpers shared across modules: validation, naming, timing, redaction."""

from __future__ import annotations

import re
import time
import unicodedata
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from ._version import __version__
from .exceptions import InvalidInputError

USER_AGENT = f"listen-in/{__version__} (+https://github.com/HWDSIU/listen-in)"

UUID_RE = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE
)
RADIO_BROWSER_HOSTS = ("radio-browser.info",)
ALLOWED_STREAM_SCHEMES = ("http", "https")

#: Hard ceiling on a single recording, overridable via ``CaptureOptions.max_duration_seconds``.
DEFAULT_MAX_DURATION_SECONDS = 3 * 60 * 60


# ------------------------------------------------------------------- validation
def extract_uuid(value: str) -> str | None:
    """Return the first station UUID found in ``value`` (lower-cased), or ``None``."""
    m = UUID_RE.search(value)
    return m.group(0).lower() if m else None


def is_radio_browser_url(value: str) -> bool:
    host = (urlparse(value).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in RADIO_BROWSER_HOSTS)


def validate_stream_url(url: str) -> str:
    """Accept only absolute http(s) URLs with a host; reject everything else.

    ffmpeg understands many protocols (``file:``, ``concat:``, ``pipe:``...). Allowing
    only http(s) keeps a user-supplied "stream URL" from reading local files.
    """
    if not isinstance(url, str) or not url.strip():
        raise InvalidInputError("Stream URL must be a non-empty string.")
    url = url.strip()
    if any(c in url for c in "\r\n\t\x00"):
        raise InvalidInputError("Stream URL contains control characters.")
    parsed = urlparse(url)
    if parsed.scheme.lower() not in ALLOWED_STREAM_SCHEMES:
        raise InvalidInputError(
            f"Stream URL must use http or https (got {parsed.scheme or 'no scheme'!r})."
        )
    if not parsed.hostname:
        raise InvalidInputError(f"Stream URL has no host: {url!r}")
    return url


def validate_duration(seconds: float, maximum: float = DEFAULT_MAX_DURATION_SECONDS) -> int:
    try:
        value = int(seconds)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(f"duration_seconds must be a number, got {seconds!r}") from exc
    if value != seconds:
        raise InvalidInputError(f"duration_seconds must be a whole number, got {seconds!r}")
    if value <= 0:
        raise InvalidInputError("duration_seconds must be positive.")
    if value > maximum:
        raise InvalidInputError(
            f"duration_seconds={value} exceeds the configured maximum of {int(maximum)} s "
            "(raise CaptureOptions.max_duration_seconds to allow longer recordings)."
        )
    return value


def validate_output_dir(path: str | Path) -> Path:
    """Normalise an output directory: a local path or a mounted storage path."""
    if isinstance(path, str):
        if not path.strip():
            raise InvalidInputError("output_dir must not be empty.")
        if "\x00" in path:
            raise InvalidInputError("output_dir contains a NUL byte.")
        if path.startswith("dbfs:/"):
            # Storage URI with a local FUSE mount: use the mount, which Python file APIs can open.
            path = "/dbfs/" + path[len("dbfs:/") :].lstrip("/")
        if "://" in path:
            raise InvalidInputError(
                f"output_dir must be a filesystem path, not a URI: {path!r}. "
                "For cloud storage, use the path where it is mounted on this machine."
            )
    p = Path(path).expanduser()
    if p.exists() and not p.is_dir():
        raise InvalidInputError(f"output_dir exists and is not a directory: {p}")
    return p


# ----------------------------------------------------------------------- naming
def slugify(value: str, max_length: int = 40) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return (value[:max_length].rstrip("-")) or "station"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_for_filename(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_run_id(station_slug: str, station_key: str, started_at: datetime) -> str:
    """Deterministic run identifier: ``<slug>_<key8>_<YYYYmmddTHHMMSSZ>``."""
    key = re.sub(r"[^a-z0-9]", "", station_key.lower())[:8] or "url"
    return f"{station_slug}_{key}_{timestamp_for_filename(started_at)}"


# ----------------------------------------------------------------------- timing
class Stopwatch:
    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    @property
    def seconds(self) -> float:
        return round(time.perf_counter() - self._t0, 3)


# -------------------------------------------------------------------- redaction
def redact(text: str, secrets: Iterable[str | None]) -> str:
    """Replace any occurrence of the given secrets in ``text`` with ``[REDACTED]``."""
    for s in secrets:
        if s and len(s) >= 6:
            text = text.replace(s, "[REDACTED]")
    return text


def describe_error(exc: BaseException, secrets: Iterable[str | None] = ()) -> str:
    """One-line, secret-free description of an exception for results and manifests."""
    msg = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    out = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
    return redact(out, secrets)[:1000]
