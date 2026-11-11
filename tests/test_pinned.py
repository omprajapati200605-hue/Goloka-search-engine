"""Connection-time SSRF protection. Uses a REAL local TCP server and a REAL httpx client (no respx)."""
import asyncio

import httpcore
import httpx
import pytest

from app.config import Settings
from app.crawler.fetcher import SafeFetcher
from app.crawler.pinned import PinnedBackend, PinnedConnectBlocked, make_pinned_transport

PUBLIC = "93.184.216.34"


@pytest.fixture
async def local_server():
    seen: list[str] = []

    async def handle(reader, writer):
        head = (await reader.readuntil(b"\r\n\r\n")).decode()
        seen.append(head)
        body = b"<html><title>pinned</title><p>hello</p></html>"
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n"
                     b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    yield port, seen
    server.close()
    await server.wait_closed()


async def test_connects_to_the_resolved_ip_and_keeps_the_hostname(local_server):
    port, seen = local_server

    async def resolver(host):
        return ["127.0.0.1"]

    transport = make_pinned_transport(resolver, public_check=lambda ip: True)       # test-only: allow loopback
    async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
        r = await client.get(f"http://pinned.test:{port}/page")      # "pinned.test" does not exist in real DNS
    assert r.status_code == 200 and "hello" in r.text
    assert f"Host: pinned.test:{port}" in seen[0]                    # hostname preserved for virtual hosting / TLS


@pytest.mark.parametrize("answer", [["127.0.0.1"], ["10.0.0.5"], ["169.254.169.254"], [PUBLIC, "192.168.1.10"], ["::1"]])
async def test_non_public_answers_are_refused_at_connect_time(answer):
    async def resolver(host):
        return answer

    async with httpx.AsyncClient(transport=make_pinned_transport(resolver), trust_env=False) as client:
        with pytest.raises(httpx.ConnectError) as info:
            await client.get("http://evil.test/")
    assert isinstance(info.value.__cause__, PinnedConnectBlocked) and "non-public" in str(info.value.__cause__)


async def test_ip_literal_and_resolver_failure_and_empty_answer_are_refused():
    async def boom(host):
        raise OSError("dns down")

    async def empty(host):
        return []

    for resolver, url in ((boom, "http://x.test/"), (empty, "http://x.test/"), (boom, "http://10.1.2.3/")):
        async with httpx.AsyncClient(transport=make_pinned_transport(resolver), trust_env=False) as client:
            with pytest.raises(httpx.ConnectError):
                await client.get(url)


async def test_dns_rebinding_is_stopped_by_the_fetcher():
    """Pre-check sees a public address; by the time the socket is opened the name points at loopback."""
    calls = []

    async def rebinding(host):
        calls.append(host)
        return [PUBLIC] if len(calls) == 1 else ["127.0.0.1"]

    s = Settings(_env_file=None, crawl_per_domain_delay_seconds=0.0, crawl_timeout_seconds=2.0)
    fetcher = SafeFetcher(s, resolver=rebinding)
    try:
        res = await fetcher.fetch("http://rebind.test/")
    finally:
        await fetcher.aclose()
    assert len(calls) >= 2                                            # the connection re-resolved and re-checked
    assert res.error_kind == "ssrf_blocked" and "127.0.0.1" in res.error


async def test_fetcher_default_client_is_pinned_and_ignores_proxy_env(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid:3128")
    fetcher = SafeFetcher(Settings(_env_file=None))
    try:
        backend = fetcher._client._transport._pool._network_backend
        assert isinstance(backend, PinnedBackend)
        assert fetcher._client._mounts == {}                          # no env-proxy transports
    finally:
        await fetcher.aclose()


def test_unsupported_httpx_fails_closed(monkeypatch):
    class Odd:
        _pool = None

    monkeypatch.setattr("app.crawler.pinned.httpx.AsyncHTTPTransport", Odd)
    with pytest.raises(RuntimeError, match="not supported"):
        make_pinned_transport()


async def test_backend_rejects_unix_sockets():
    backend = PinnedBackend(httpcore.AnyIOBackend())
    with pytest.raises(PinnedConnectBlocked):
        await backend.connect_unix_socket("/tmp/x.sock")
