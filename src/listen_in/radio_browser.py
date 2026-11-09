"""Resolve station references (stream URL, Radio Browser UUID or history URL) to a Station.

Radio Browser (https://www.radio-browser.info) is a community directory with several
API mirrors. History pages are identifiers, not streams: we extract the UUID and ask
the API for the station record, preferring ``url_resolved`` over ``url``. HTML pages
are never scraped.
"""

from __future__ import annotations

import logging
import random
import socket
import threading
from collections.abc import Sequence

import httpx

from .exceptions import (
    InvalidInputError,
    RadioBrowserUnavailableError,
    StationNotFoundError,
    StationResolutionError,
)
from .models import Station
from .utils import (
    USER_AGENT,
    UUID_RE,
    extract_uuid,
    is_radio_browser_url,
    validate_stream_url,
)

log = logging.getLogger(__name__)

DISCOVERY_HOST = "all.api.radio-browser.info"
FALLBACK_SERVERS: tuple[str, ...] = (
    "de1.api.radio-browser.info",
    "de2.api.radio-browser.info",
    "fi1.api.radio-browser.info",
    "nl1.api.radio-browser.info",
    "at1.api.radio-browser.info",
)

_lock = threading.Lock()
_server_cache: list[str] | None = None
_last_good: str | None = None


# -------------------------------------------------------------- server discovery
def _dns_servers() -> list[str]:
    """Radio Browser's documented method: resolve all.api, then reverse-DNS each IP."""
    names: set[str] = set()
    for *_, sockaddr in socket.getaddrinfo(DISCOVERY_HOST, 443, proto=socket.IPPROTO_TCP):
        try:
            host = socket.gethostbyaddr(str(sockaddr[0]))[0].lower()
        except OSError:
            continue
        if host.endswith(".api.radio-browser.info"):
            names.add(host)
    return sorted(names)


def _json_servers(client: httpx.Client) -> list[str]:
    r = client.get(f"https://{DISCOVERY_HOST}/json/servers")
    r.raise_for_status()
    return sorted({str(s["name"]).lower() for s in r.json() if s.get("name")})


def discover_servers(client: httpx.Client | None = None, *, refresh: bool = False) -> list[str]:
    """Return Radio Browser API hostnames to try, best first. Cached per process.

    Order: DNS discovery (shuffled, as Radio Browser recommends for load spreading),
    then the JSON server list, then a static fallback list.
    """
    global _server_cache
    with _lock:
        if _server_cache is not None and not refresh:
            return _ordered(_server_cache)

    found: list[str] = []
    try:
        found = _dns_servers()
    except OSError as exc:
        log.debug("Radio Browser DNS discovery failed: %s", exc)
    if not found:
        own = client is None
        c = client or _make_client(timeout=8.0)
        try:
            found = _json_servers(c)
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            log.debug("Radio Browser JSON discovery failed: %s", exc)
        finally:
            if own:
                c.close()
    random.shuffle(found)
    servers = list(dict.fromkeys([*found, *FALLBACK_SERVERS]))
    with _lock:
        _server_cache = servers
    return _ordered(servers)


def _ordered(servers: list[str]) -> list[str]:
    if _last_good and _last_good in servers:
        return [_last_good, *[s for s in servers if s != _last_good]]
    return list(servers)


def _remember_good(host: str) -> None:
    global _last_good
    with _lock:
        _last_good = host


def _make_client(timeout: float) -> httpx.Client:
    return httpx.Client(
        timeout=timeout, headers={"User-Agent": USER_AGENT}, follow_redirects=True
    )


# -------------------------------------------------------------------- resolution
def parse_station_reference(ref: str) -> tuple[str, str]:
    """Classify a reference. Returns ``("uuid", uuid)`` or ``("url", stream_url)``."""
    if not isinstance(ref, str) or not ref.strip():
        raise InvalidInputError("Station reference must be a non-empty string.")
    ref = ref.strip()
    if UUID_RE.fullmatch(ref):
        return "uuid", ref.lower()
    if "://" in ref:
        if is_radio_browser_url(ref):
            uuid = extract_uuid(ref)
            if not uuid:
                raise InvalidInputError(
                    f"Radio Browser URL does not contain a station UUID: {ref!r}"
                )
            return "uuid", uuid
        return "url", validate_stream_url(ref)
    raise InvalidInputError(
        f"Not a stream URL, Radio Browser UUID or Radio Browser history URL: {ref!r}"
    )


def resolve_station(
    station: str | Station,
    *,
    client: httpx.Client | None = None,
    servers: Sequence[str] | None = None,
    timeout: float = 10.0,
) -> Station:
    """Resolve a station reference to a :class:`Station`.

    Accepts a direct http(s) stream URL, a Radio Browser station UUID, or a Radio
    Browser history URL (``https://www.radio-browser.info/history/<uuid>``).
    Radio Browser lookups retry on the next API server if one fails.
    """
    if isinstance(station, Station):
        return station
    kind, value = parse_station_reference(station)
    if kind == "url":
        return Station(stream_url=value, source="direct_url")
    return _lookup_uuid(value, client=client, servers=servers, timeout=timeout)


def _lookup_uuid(
    uuid: str,
    *,
    client: httpx.Client | None,
    servers: Sequence[str] | None,
    timeout: float,
) -> Station:
    own = client is None
    c = client or _make_client(timeout)
    try:
        hosts = list(servers) if servers else discover_servers(c)
        errors: list[str] = []
        for host in hosts:
            try:
                r = c.get(f"https://{host}/json/stations/byuuid/{uuid}")
                r.raise_for_status()
                data = r.json()
            except (httpx.HTTPError, ValueError) as exc:
                log.info("Radio Browser server %s failed: %s", host, exc)
                errors.append(f"{host}: {type(exc).__name__}")
                continue
            if not isinstance(data, list):
                errors.append(f"{host}: unexpected response")
                continue
            _remember_good(host)
            if not data:
                raise StationNotFoundError(f"No Radio Browser station with UUID {uuid}.")
            return _station_from_record(data[0], host)
        raise RadioBrowserUnavailableError(
            "Could not reach any Radio Browser API server (" + "; ".join(errors) + "). "
            "Check outbound HTTPS access to *.api.radio-browser.info."
        )
    finally:
        if own:
            c.close()


def _text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _split_list(value: object, *, lower: bool = False) -> list[str]:
    """Comma-separated Radio Browser field -> de-duplicated, trimmed list."""
    items = [x.strip() for x in str(value or "").split(",")]
    items = [x.lower() if lower else x for x in items if x]
    return list(dict.fromkeys(items))


def _coordinate(value: object, limit: float) -> float | None:
    """Parse a latitude/longitude; None when missing, non-numeric or out of range."""
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if -limit <= number <= limit else None


def _station_from_record(rec: dict, host: str) -> Station:
    raw = (rec.get("url_resolved") or "").strip() or (rec.get("url") or "").strip()
    if not raw:
        raise StationResolutionError(
            f"Radio Browser station {rec.get('stationuuid')} has no stream URL."
        )
    try:
        stream_url = validate_stream_url(raw)
    except InvalidInputError as exc:
        raise StationResolutionError(f"Radio Browser returned an unusable stream URL: {exc}") from exc
    bitrate = rec.get("bitrate")
    return Station(
        stream_url=stream_url,
        name=(rec.get("name") or "").strip(),
        source="radio_browser",
        uuid=(rec.get("stationuuid") or "").lower() or None,
        url=rec.get("url") or None,
        url_resolved=rec.get("url_resolved") or None,
        homepage=rec.get("homepage") or None,
        country=rec.get("country") or None,
        countrycode=rec.get("countrycode") or None,
        subdivision=_text(rec.get("state")),
        subdivision_code=_text(rec.get("iso_3166_2")),
        latitude=_coordinate(rec.get("geo_lat"), 90),
        longitude=_coordinate(rec.get("geo_long"), 180),
        language=rec.get("language") or None,
        language_codes=_split_list(rec.get("languagecodes"), lower=True),
        tags=_split_list(rec.get("tags")),
        favicon=_text(rec.get("favicon")),
        codec=rec.get("codec") or None,
        bitrate=int(bitrate) if isinstance(bitrate, (int, float)) and bitrate else None,
        lastcheckok=bool(rec["lastcheckok"]) if "lastcheckok" in rec else None,
        api_server=host,
    )
