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
