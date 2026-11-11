from app.crawler.fetcher import FetchResult
from app.crawler.robots import RobotsPolicy, agent_token, parse_robots

ROBOTS = """
# comment
User-agent: *
Disallow: /private/
Disallow: /*.pdf$
Allow: /private/public-page
Crawl-delay: 2

User-agent: GOLOKABot
Disallow: /secret
Crawl-delay: 5
"""


def allowed(text, path, token="otherbot"):
    return parse_robots(text, token).is_allowed(path)


def test_agent_token():
    assert agent_token("GOLOKABot/0.1 (+https://example.com/bot)") == "golokabot"


def test_wildcard_group_rules():
    assert not allowed(ROBOTS, "/private/x")
    assert allowed(ROBOTS, "/public")
    assert not allowed(ROBOTS, "/docs/file.pdf")
    assert allowed(ROBOTS, "/docs/file.pdf?x=1")      # '$' anchors the end
    assert allowed(ROBOTS, "/private/public-page")    # longer Allow beats shorter Disallow


def test_specific_group_overrides_wildcard():
    rules = parse_robots(ROBOTS, "golokabot")
    assert not rules.is_allowed("/secret/page")
    assert rules.is_allowed("/private/x")   # wildcard group ignored for named agent
    assert rules.crawl_delay == 5


def test_empty_disallow_allows_all_and_missing_rules():
    assert allowed("User-agent: *\nDisallow:", "/anything")
    assert allowed("", "/anything")


def test_disallow_all():
    assert not allowed("User-agent: *\nDisallow: /", "/")


def test_multiple_agents_share_group():
    text = "User-agent: a\nUser-agent: golokabot\nDisallow: /x\n"
    assert not parse_robots(text, "golokabot").is_allowed("/x")


def test_allow_wins_tie():
    assert allowed("User-agent: *\nDisallow: /p\nAllow: /p", "/p")


# ---- policy / caching / failure modes ------------------------------------------------------
def make_policy(result: FetchResult):
    calls = []

    async def fetch(url):
        calls.append(url)
        return result

    return RobotsPolicy(fetch, "GOLOKABot/0.1"), calls


def res(status=None, text="", kind=None):
    return FetchResult(url="u", final_url="u", status_code=status, text=text, error_kind=kind)


async def test_policy_obeys_and_caches():
    policy, calls = make_policy(res(200, "User-agent: *\nDisallow: /no\nCrawl-delay: 3"))
    assert not await policy.allowed("https://a.test/no/page")
    assert await policy.allowed("https://a.test/yes")
    assert await policy.crawl_delay("https://a.test/yes") == 3
    assert calls == ["https://a.test/robots.txt"]  # fetched once per origin


async def test_policy_404_allows_all():
    policy, _ = make_policy(res(404, kind="not_found"))
    assert await policy.allowed("https://a.test/anything")


async def test_policy_server_error_disallows_all():
    for kind, status in [("http_error", 500), ("timeout", None), ("network_error", None), ("ssrf_blocked", None)]:
        policy, _ = make_policy(res(status, kind=kind))
        assert not await policy.allowed("https://a.test/")


async def test_policy_403_disallows_all():
    policy, _ = make_policy(res(403, kind="restricted"))
    assert not await policy.allowed("https://a.test/")


async def test_policy_caches_per_origin():
    policy, calls = make_policy(res(404, kind="not_found"))
    await policy.allowed("https://a.test/")
    await policy.allowed("https://b.test/")
    await policy.allowed("http://a.test/")
    assert len(calls) == 3


async def test_crawl_delay_is_capped():
    policy, _ = make_policy(res(200, "User-agent: *\nCrawl-delay: 9999"))
    assert await policy.crawl_delay("https://a.test/") == 30.0
