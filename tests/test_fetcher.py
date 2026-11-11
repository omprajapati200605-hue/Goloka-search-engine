import asyncio
import time

import httpx
import pytest

from app.crawler.fetcher import DomainRateLimiter, SafeFetcher

HTML = "<html><head><title>T</title></head><body><p>hello</p></body></html>"


@pytest.fixture
async def fetcher(settings, resolver):
    f = SafeFetcher(settings, resolver=resolver)
    yield f
    await f.aclose()


async def test_fetch_success(fetcher, mock):
    mock.get("https://site.test/").respond(200, html=HTML)
    r = await fetcher.fetch("https://site.test/")
    assert r.ok and r.status_code == 200 and "hello" in r.text
    assert r.content_type == "text/html" and r.final_url == "https://site.test/"


async def test_fetch_follows_redirects_and_records_chain(fetcher, mock):
    mock.get("https://site.test/a").respond(301, headers={"Location": "/b"})
    mock.get("https://site.test/b").respond(302, headers={"Location": "https://site.test/c"})
    mock.get("https://site.test/c").respond(200, html=HTML)
    r = await fetcher.fetch("https://site.test/a")
    assert r.ok and r.final_url == "https://site.test/c"
    assert r.redirects == ["https://site.test/a", "https://site.test/b"]


async def test_redirect_to_private_ip_literal_blocked(fetcher, mock):
    mock.get("https://site.test/").respond(302, headers={"Location": "http://169.254.169.254/latest/meta-data/"})
    meta = mock.get("http://169.254.169.254/latest/meta-data/").respond(200, html="secret")
    r = await fetcher.fetch("https://site.test/")
    assert r.error_kind == "ssrf_blocked" and not meta.called


async def test_redirect_to_host_resolving_private_blocked(fetcher, mock):
    mock.get("https://site.test/").respond(302, headers={"Location": "https://private.test/admin"})
    internal = mock.get("https://private.test/admin").respond(200, html="secret")
    r = await fetcher.fetch("https://site.test/")
    assert r.error_kind == "ssrf_blocked" and not internal.called


async def test_initial_url_resolving_private_blocked(fetcher, mock):
    route = mock.get("https://loopback.test/").respond(200, html=HTML)
    r = await fetcher.fetch("https://loopback.test/")
    assert r.error_kind == "ssrf_blocked" and not route.called


async def test_redirect_to_non_http_scheme_rejected(fetcher, mock):
    mock.get("https://site.test/").respond(302, headers={"Location": "file:///etc/passwd"})
    assert (await fetcher.fetch("https://site.test/")).error_kind == "invalid_url"


async def test_too_many_redirects(fetcher, mock):
    for i in range(10):
        mock.get(f"https://site.test/{i}").respond(302, headers={"Location": f"/{i + 1}"})
    assert (await fetcher.fetch("https://site.test/0")).error_kind == "too_many_redirects"


async def test_redirect_loop(fetcher, mock):
    mock.get("https://site.test/a").respond(302, headers={"Location": "/b"})
    mock.get("https://site.test/b").respond(302, headers={"Location": "/a"})
    assert (await fetcher.fetch("https://site.test/a")).error_kind == "redirect_loop"


@pytest.mark.parametrize("status,kind", [
    (404, "not_found"), (410, "not_found"), (401, "restricted"), (403, "restricted"),
    (429, "rate_limited"), (500, "http_error"), (503, "http_error"),
])
async def test_http_errors_classified_and_never_retried(fetcher, mock, status, kind):
    route = mock.get("https://site.test/x").respond(status, html="nope")
    r = await fetcher.fetch("https://site.test/x")
    assert r.error_kind == kind and r.status_code == status
    assert route.call_count == 1


async def test_timeout(fetcher, mock):
    mock.get("https://site.test/slow").mock(side_effect=httpx.ReadTimeout("slow"))
    assert (await fetcher.fetch("https://site.test/slow")).error_kind == "timeout"


async def test_network_error(fetcher, mock):
    mock.get("https://site.test/down").mock(side_effect=httpx.ConnectError("refused"))
    assert (await fetcher.fetch("https://site.test/down")).error_kind == "network_error"


async def test_non_html_rejected(fetcher, mock):
    mock.get("https://site.test/f").respond(200, content=b"%PDF", headers={"Content-Type": "application/pdf"})
    assert (await fetcher.fetch("https://site.test/f")).error_kind == "non_html"


async def test_body_too_large_streaming(fetcher, mock):
    mock.get("https://site.test/big").respond(200, content=b"x" * 60_000, headers={"Content-Type": "text/html"})
    assert (await fetcher.fetch("https://site.test/big")).error_kind == "too_large"


async def test_content_length_too_large(fetcher, mock):
    mock.get("https://site.test/big").respond(
        200, content=b"x", headers={"Content-Type": "text/html", "Content-Length": "999999"})
    assert (await fetcher.fetch("https://site.test/big")).error_kind == "too_large"


async def test_charset_decoding(fetcher, mock):
    mock.get("https://site.test/l1").respond(
        200, content="café".encode("latin-1"), headers={"Content-Type": "text/html; charset=latin-1"})
    assert (await fetcher.fetch("https://site.test/l1")).text == "café"


async def test_policy_block_prevents_request(fetcher, mock):
    route = mock.get("https://site.test/x").respond(200, html=HTML)

    class Deny:
        async def check(self, url):
            return "nope"

        async def delay(self, url):
            return 0.0

    r = await fetcher.fetch("https://site.test/x", policy=Deny())
    assert r.error_kind == "blocked" and not route.called


async def test_sends_configured_user_agent(fetcher, mock):
    route = mock.get("https://site.test/").respond(200, html=HTML)
    await fetcher.fetch("https://site.test/")
    assert route.calls.last.request.headers["user-agent"].startswith("GOLOKABot/0.1")


async def test_rate_limiter_spaces_requests_per_domain():
    limiter = DomainRateLimiter()
    start = time.monotonic()
    await asyncio.gather(*(limiter.wait("a.test", 0.2) for _ in range(3)))
    assert time.monotonic() - start >= 0.38
    start = time.monotonic()
    await asyncio.gather(*(limiter.wait(f"h{i}.test", 0.2) for i in range(3)))  # different hosts: parallel
    assert time.monotonic() - start < 0.15
