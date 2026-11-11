import pytest

from app.crawler.crawler import Crawler
from app.crawler.links import extract_links
from app.crawler.url_utils import SSRFError, URLValidationError


def page(title, links=(), extra=""):
    anchors = "".join(f'<a href="{h}">link</a>' for h in links)
    return f"<html><head><title>{title}</title>{extra}</head><body><h1>{title}</h1>{anchors}</body></html>"


def site(mock, pages: dict[str, str], host="https://site.test", robots=None):
    routes = {}
    if robots is None:
        mock.get(f"{host}/robots.txt").respond(404)
    else:
        mock.get(f"{host}/robots.txt").respond(200, text=robots)
    for path, html in pages.items():
        routes[path] = mock.get(f"{host}{path}").respond(200, html=html)
    return routes


@pytest.fixture
def run(settings, resolver):
    async def _run(seed, pages_seen=None, **kwargs):
        collected = pages_seen if pages_seen is not None else []
        crawler = Crawler(settings, resolver=resolver, on_page=collected.append)
        try:
            return await crawler.crawl(seed, **kwargs)
        finally:
            await crawler.aclose()
    return _run


# ---- link extraction -----------------------------------------------------------------------
def test_extract_links_resolves_filters_and_dedupes():
    html = """<a href="/a">1</a><a href="/a#x">2</a><a href="b?utm_source=z">3</a>
      <a href="mailto:x@y.z">m</a><a href="javascript:void(0)">j</a><a href="#top">t</a>
      <a href="https://other.test/q" rel="nofollow">nf</a><a href="https://other.test/ok">ok</a>"""
    assert extract_links(html, "https://site.test/dir/") == [
        "https://site.test/a", "https://site.test/dir/b", "https://other.test/ok"]


def test_extract_links_respects_meta_nofollow_and_base_tag():
    assert extract_links('<meta name="robots" content="nofollow"><a href="/a">x</a>', "https://s.test/") == []
    assert extract_links('<base href="https://cdn.test/base/"><a href="x">x</a>', "https://s.test/") == [
        "https://cdn.test/base/x"]


# ---- discovery, depth, limits --------------------------------------------------------------
async def test_crawl_discovers_pages_within_depth(mock, run):
    site(mock, {
        "/": page("Home", ["/a", "/b"]),
        "/a": page("A", ["/c"]),
        "/b": page("B"),
        "/c": page("C", ["/d"]),
        "/d": page("D"),
    })
    got = []
    summary = await run("https://site.test/", got, max_depth=2, max_pages=50)
    assert sorted(p.url for p in got) == [
        "https://site.test/", "https://site.test/a", "https://site.test/b", "https://site.test/c"]
    assert {p.depth for p in got} == {0, 1, 2}
    assert summary.pages_crawled == 4 and summary.finished_at and summary.started_at


async def test_depth_zero_crawls_only_seed(mock, run):
    routes = site(mock, {"/": page("Home", ["/a"]), "/a": page("A")})
    got = []
    await run("https://site.test/", got, max_depth=0)
    assert [p.url for p in got] == ["https://site.test/"] and not routes["/a"].called


async def test_max_pages_limit(mock, run):
    site(mock, {"/": page("Home", [f"/p{i}" for i in range(10)]), **{f"/p{i}": page(f"P{i}") for i in range(10)}})
    summary = await run("https://site.test/", max_pages=4, max_depth=3)
    assert len(summary.outcomes) == 4


async def test_hard_limits_clamp_requests(mock, run, settings):
    site(mock, {"/": page("Home")})
    summary = await run("https://site.test/", max_pages=10**6, max_depth=10**6)
    assert summary.max_pages == settings.crawl_hard_max_pages
    assert summary.max_depth == settings.crawl_hard_max_depth


async def test_invalid_limits_rejected(run):
    with pytest.raises(ValueError):
        await run("https://site.test/", max_pages=0)
    with pytest.raises(ValueError):
        await run("https://site.test/", max_depth=-1)


# ---- scope ---------------------------------------------------------------------------------
async def test_stays_on_seed_domain_by_default(mock, run):
    site(mock, {"/": page("Home", ["https://other.test/x", "/a"]), "/a": page("A")})
    other = mock.get("https://other.test/x").respond(200, html=page("X"))
    got = []
    await run("https://site.test/", got)
    assert not other.called and len(got) == 2


async def test_allowlist_permits_extra_domains(mock, run):
    site(mock, {"/": page("Home", ["https://other.test/x"])})
    site(mock, {"/x": page("X")}, host="https://other.test")
    got = []
    await run("https://site.test/", got, allowed_domains=["other.test"])
    assert sorted(p.url for p in got) == ["https://other.test/x", "https://site.test/"]


async def test_invalid_allowlist_domain_rejected(mock, run):
    with pytest.raises(ValueError):
        await run("https://site.test/", allowed_domains=["https://bad/path"])


async def test_redirect_to_unlisted_domain_is_blocked(mock, run):
    mock.get("https://site.test/robots.txt").respond(404)
    mock.get("https://site.test/").respond(302, headers={"Location": "https://evil.test/"})
    evil = mock.get("https://evil.test/").respond(200, html=page("Evil"))
    got = []
    summary = await run("https://site.test/", got)
    assert not evil.called and not got and summary.outcomes[0].status == "blocked"


# ---- duplicates ----------------------------------------------------------------------------
async def test_duplicate_urls_fetched_once(mock, run):
    routes = site(mock, {"/": page("Home", ["/a", "/a#frag", "/a?utm_source=x", "/a/../a"]), "/a": page("A")})
    await run("https://site.test/")
    assert routes["/a"].call_count == 1


async def test_duplicate_content_detected(mock, run):
    same = page("Same", [])
    site(mock, {"/": page("Home", ["/x", "/y"]), "/x": same, "/y": same})
    got = []
    summary = await run("https://site.test/", got)
    assert len(got) == 2  # home + one of x/y
    assert summary.count("duplicate") == 1


async def test_redirect_target_already_seen_is_duplicate(mock, run):
    mock.get("https://site.test/robots.txt").respond(404)
    mock.get("https://site.test/").respond(200, html=page("Home", ["/old", "/new"]))
    mock.get("https://site.test/old").respond(301, headers={"Location": "/new"})
    new = mock.get("https://site.test/new").respond(200, html=page("New"))
    summary = await run("https://site.test/")
    assert new.call_count <= 2 and summary.count("fetched") == 2


# ---- robots, login, restricted, noindex ----------------------------------------------------
async def test_robots_disallow_is_respected(mock, run):
    routes = site(mock, {"/": page("Home", ["/private/x", "/ok"]), "/private/x": page("P"), "/ok": page("OK")},
                  robots="User-agent: *\nDisallow: /private/")
    got = []
    summary = await run("https://site.test/", got)
    assert not routes["/private/x"].called
    assert sorted(p.url for p in got) == ["https://site.test/", "https://site.test/ok"]
    assert summary.count("blocked") == 1


async def test_robots_unreachable_means_nothing_crawled(mock, run):
    mock.get("https://site.test/robots.txt").respond(500)
    home = mock.get("https://site.test/").respond(200, html=page("Home"))
    summary = await run("https://site.test/")
    assert not home.called and summary.pages_crawled == 0 and summary.count("blocked") == 1


async def test_login_paths_not_requested(mock, run):
    routes = site(mock, {"/": page("Home", ["/login", "/account/settings", "/ok"]),
                         "/login": page("L"), "/account/settings": page("S"), "/ok": page("OK")})
    await run("https://site.test/")
    assert not routes["/login"].called and not routes["/account/settings"].called and routes["/ok"].called


async def test_password_form_page_not_stored(mock, run):
    login_like = page("Sign in", extra="").replace("</body>", '<form><input type="password" name="p"></form></body>')
    site(mock, {"/": page("Home", ["/portal"]), "/portal": login_like})
    got = []
    summary = await run("https://site.test/", got)
    assert [p.url for p in got] == ["https://site.test/"] and summary.count("restricted") == 1


async def test_401_403_recorded_as_restricted_not_retried(mock, run):
    mock.get("https://site.test/robots.txt").respond(404)
    mock.get("https://site.test/").respond(200, html=page("Home", ["/members", "/forbidden"]))
    r1 = mock.get("https://site.test/members").respond(401)
    r2 = mock.get("https://site.test/forbidden").respond(403)
    summary = await run("https://site.test/")
    assert summary.count("restricted") == 2 and r1.call_count == 1 and r2.call_count == 1


async def test_noindex_pages_not_delivered_but_links_followed(mock, run):
    site(mock, {"/": page("Home", ["/n"]),
                "/n": page("N", ["/after"], extra='<meta name="robots" content="noindex">'),
                "/after": page("After")})
    got = []
    await run("https://site.test/", got)
    assert sorted(p.url for p in got) == ["https://site.test/", "https://site.test/after"]


async def test_non_html_links_not_requested(mock, run):
    routes = site(mock, {"/": page("Home", ["/doc.pdf", "/img.png"])})
    pdf = mock.get("https://site.test/doc.pdf").respond(200, content=b"%PDF")
    await run("https://site.test/")
    assert not pdf.called and len(routes) == 1


# ---- errors --------------------------------------------------------------------------------
async def test_http_errors_are_recorded_and_crawl_continues(mock, run):
    mock.get("https://site.test/robots.txt").respond(404)
    mock.get("https://site.test/").respond(200, html=page("Home", ["/gone", "/boom", "/ok"]))
    mock.get("https://site.test/gone").respond(404)
    mock.get("https://site.test/boom").respond(500)
    mock.get("https://site.test/ok").respond(200, html=page("OK"))
    got = []
    summary = await run("https://site.test/", got)
    assert len(got) == 2 and summary.pages_failed == 2
    assert {o.http_status for o in summary.outcomes if o.status == "error"} == {404, 500}


async def test_on_page_exception_does_not_stop_crawl(mock, run, settings, resolver):
    site(mock, {"/": page("Home", ["/a"]), "/a": page("A")})

    def boom(_page):
        raise RuntimeError("db down")

    crawler = Crawler(settings, resolver=resolver, on_page=boom)
    summary = await crawler.crawl("https://site.test/")
    await crawler.aclose()
    assert summary.pages_failed == 2 and all("on_page failed" in o.detail for o in summary.outcomes)


async def test_async_callbacks_supported(mock, run, settings, resolver):
    site(mock, {"/": page("Home")})
    seen = []

    async def on_page(p):
        seen.append(p.url)

    crawler = Crawler(settings, resolver=resolver, on_page=on_page)
    await crawler.crawl("https://site.test/")
    await crawler.aclose()
    assert seen == ["https://site.test/"]


# ---- seed validation -----------------------------------------------------------------------
@pytest.mark.parametrize("seed", ["ftp://site.test/", "not a url", "http://127.0.0.1/", "http://localhost:8080/"])
async def test_bad_seeds_rejected(run, seed):
    with pytest.raises(URLValidationError):
        await run(seed)


async def test_seed_resolving_to_private_ip_rejected(run):
    with pytest.raises(SSRFError):
        await run("https://private.test/")
