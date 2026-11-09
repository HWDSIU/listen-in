from __future__ import annotations

import httpx
import pytest

from listen_in import radio_browser
from listen_in.exceptions import (
    InvalidInputError,
    RadioBrowserUnavailableError,
    StationNotFoundError,
)
from listen_in.radio_browser import parse_station_reference, resolve_station

from .conftest import (
    ENGLISH_HISTORY,
    ENGLISH_UUID,
    FRENCH_HISTORY,
    FRENCH_UUID,
    station_record,
)


def make_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": "test"})


def no_network(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
    raise AssertionError(f"unexpected network call to {request.url}")


# ------------------------------------------------------------------ parsing
@pytest.mark.parametrize(
    ("ref", "uuid"),
    [(FRENCH_HISTORY, FRENCH_UUID), (ENGLISH_HISTORY, ENGLISH_UUID)],
)
def test_uuid_extracted_from_supplied_history_urls(ref, uuid):
    assert parse_station_reference(ref) == ("uuid", uuid)


def test_bare_uuid_is_normalised_to_lowercase():
    assert parse_station_reference(FRENCH_UUID.upper()) == ("uuid", FRENCH_UUID)


def test_direct_stream_url_needs_no_network():
    url = "https://stream.live.vc.bbcmedia.co.uk/bbc_world_service"
    st = resolve_station(url, client=make_client(no_network))
    assert st.stream_url == url
    assert st.source == "direct_url"
    assert st.uuid is None


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "not a station",
        "31074f8a-e6f4-11e9-a96c",  # truncated UUID
        "https://www.radio-browser.info/history/not-a-uuid",
        "file:///etc/passwd",
        "ftp://example.com/stream",
        "concat:/etc/passwd|/etc/hosts",
        "https://",
    ],
)
def test_invalid_references_rejected(bad):
    with pytest.raises(InvalidInputError):
        resolve_station(bad, client=make_client(no_network))


# --------------------------------------------------------------- resolution
def test_resolves_history_url_and_sends_user_agent():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["ua"] = request.headers.get("user-agent")
        return httpx.Response(200, json=[station_record()])

    st = resolve_station(FRENCH_HISTORY, client=make_client(handler), servers=["s1.example"])
    assert seen["path"] == f"/json/stations/byuuid/{FRENCH_UUID}"
    assert seen["ua"]
    assert st.name == "France Info"
    assert st.source == "radio_browser"
    assert st.uuid == FRENCH_UUID
    assert st.api_server == "s1.example"
    assert st.bitrate == 192
    assert st.lastcheckok is True


def test_prefers_url_resolved_over_url():
    rec = station_record(url="http://example.com/playlist.pls", url_resolved="https://cdn.example.com/live")
    client = make_client(lambda r: httpx.Response(200, json=[rec]))
    st = resolve_station(FRENCH_UUID, client=client, servers=["s1"])
    assert st.stream_url == "https://cdn.example.com/live"
    assert st.url == "http://example.com/playlist.pls"


def test_falls_back_to_url_when_url_resolved_empty():
    rec = station_record(url="https://example.com/live", url_resolved="")
    client = make_client(lambda r: httpx.Response(200, json=[rec]))
    assert resolve_station(FRENCH_UUID, client=client, servers=["s1"]).stream_url == "https://example.com/live"


def test_unusable_stream_url_from_api_is_rejected():
    rec = station_record(url="file:///etc/passwd", url_resolved="")
    client = make_client(lambda r: httpx.Response(200, json=[rec]))
    with pytest.raises(radio_browser.StationResolutionError):
        resolve_station(FRENCH_UUID, client=client, servers=["s1"])


def test_server_failover_on_http_error_and_connection_error():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)
        if request.url.host == "down.example":
            raise httpx.ConnectError("boom", request=request)
        if request.url.host == "broken.example":
            return httpx.Response(503)
        return httpx.Response(200, json=[station_record()])

    st = resolve_station(
        FRENCH_UUID,
        client=make_client(handler),
        servers=["down.example", "broken.example", "good.example"],
    )
    assert calls == ["down.example", "broken.example", "good.example"]
    assert st.api_server == "good.example"


def test_last_good_server_tried_first_next_time(monkeypatch):
    monkeypatch.setattr(radio_browser, "_server_cache", ["a.example", "b.example"])
    calls = []

    def handler(request):
        calls.append(request.url.host)
        if request.url.host == "a.example":
            return httpx.Response(500)
        return httpx.Response(200, json=[station_record()])

    resolve_station(FRENCH_UUID, client=make_client(handler))
    resolve_station(FRENCH_UUID, client=make_client(handler))
    assert calls == ["a.example", "b.example", "b.example"]


def test_missing_station_raises_not_found():
    client = make_client(lambda r: httpx.Response(200, json=[]))
    with pytest.raises(StationNotFoundError):
        resolve_station(FRENCH_UUID, client=client, servers=["s1", "s2"])


def test_all_servers_down_raises_unavailable():
    client = make_client(lambda r: httpx.Response(502))
    with pytest.raises(RadioBrowserUnavailableError, match="s1: HTTPStatusError"):
        resolve_station(FRENCH_UUID, client=client, servers=["s1", "s2"])


# ---------------------------------------------------------------- discovery
def test_discovery_prefers_dns_then_appends_fallbacks(monkeypatch):
    monkeypatch.setattr(radio_browser, "_dns_servers", lambda: ["de1.api.radio-browser.info"])
    servers = radio_browser.discover_servers(make_client(no_network))
    assert servers[0] == "de1.api.radio-browser.info"
    assert set(radio_browser.FALLBACK_SERVERS) <= set(servers)
    assert len(servers) == len(set(servers))


def test_discovery_uses_json_list_when_dns_fails(monkeypatch):
    def dns_fail():
        raise OSError("no dns")

    monkeypatch.setattr(radio_browser, "_dns_servers", dns_fail)
    client = make_client(
        lambda r: httpx.Response(200, json=[{"name": "xx1.api.radio-browser.info"}])
    )
    servers = radio_browser.discover_servers(client)
    assert "xx1.api.radio-browser.info" in servers


def test_discovery_falls_back_to_static_list(monkeypatch):
    monkeypatch.setattr(radio_browser, "_dns_servers", lambda: [])
    client = make_client(lambda r: httpx.Response(500))
    assert set(radio_browser.discover_servers(client)) == set(radio_browser.FALLBACK_SERVERS)


# ------------------------------------------------------------- extra metadata
def test_captures_subdivision_geo_tags_and_language_codes():
    rec = station_record(
        state="Berlin", iso_3166_2="DE-BE", geo_lat=52.4802578, geo_long=13.3347639,
        languagecodes="DE", tags="culture,news,public service,information",
        favicon="https://example.org/icon.png",
    )
    st = resolve_station(FRENCH_UUID, client=make_client(lambda r: httpx.Response(200, json=[rec])), servers=["s1"])
    assert (st.subdivision, st.subdivision_code) == ("Berlin", "DE-BE")
    assert (st.latitude, st.longitude) == (52.4802578, 13.3347639)
    assert st.language_codes == ["de"]
    assert st.tags == ["culture", "news", "public service", "information"]
    assert st.favicon == "https://example.org/icon.png"
    dumped = st.model_dump(mode="json")  # flows into manifests unchanged
    assert dumped["subdivision_code"] == "DE-BE" and dumped["latitude"] == 52.4802578


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, (None, None, None, None, [], [])),  # fields absent
        ({"state": "", "iso_3166_2": " ", "geo_lat": None, "geo_long": None, "tags": "", "languagecodes": ""},
         (None, None, None, None, [], [])),
        ({"geo_lat": "51.5", "geo_long": "-0.12"}, (None, None, 51.5, -0.12, [], [])),  # numeric strings
        ({"geo_lat": 200, "geo_long": "east"}, (None, None, None, None, [], [])),  # out of range / junk
        ({"tags": " news , news,talk ,", "languagecodes": "DE, en ,de"},
         (None, None, None, None, ["news", "talk"], ["de", "en"])),
    ],
)
def test_extra_metadata_is_optional_and_cleaned(overrides, expected):
    rec = station_record(countrycode="", **overrides)  # no country: isolates cleaning from the fallback
    st = resolve_station(FRENCH_UUID, client=make_client(lambda r: httpx.Response(200, json=[rec])), servers=["s1"])
    assert (st.subdivision, st.subdivision_code, st.latitude, st.longitude, st.tags, st.language_codes) == expected


# ------------------------------------------------------------ location precision
def _resolve(rec):
    return resolve_station(FRENCH_UUID, client=make_client(lambda r: httpx.Response(200, json=[rec])), servers=["s1"])


def test_station_coordinates_have_station_precision():
    st = _resolve(station_record(geo_lat=48.85, geo_long=2.27))
    assert (st.latitude, st.longitude, st.location_precision) == (48.85, 2.27, "station")


def test_missing_coordinates_fall_back_to_country_reference_point():
    st = _resolve(station_record(countrycode="FR"))
    assert (st.latitude, st.longitude, st.location_precision) == (48.86, 2.35, "country")


def test_half_coordinates_count_as_missing():
    st = _resolve(station_record(countrycode="FR", geo_lat=48.85))
    assert st.location_precision == "country" and st.latitude == 48.86


def test_unknown_country_has_no_location():
    st = _resolve(station_record(countrycode="ZZ"))
    assert (st.latitude, st.longitude, st.location_precision) == (None, None, None)


def test_country_reference_table_is_sane():
    from listen_in.geo import COUNTRY_REFERENCE_POINTS, country_reference_point

    assert len(COUNTRY_REFERENCE_POINTS) > 200
    for code, (lat, lon) in COUNTRY_REFERENCE_POINTS.items():
        assert len(code) == 2 and code.isupper()
        assert -90 <= lat <= 90 and -180 <= lon <= 180, code
    assert country_reference_point("gb") == (51.51, -0.13)
    assert country_reference_point(None) is None
