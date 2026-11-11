import pytest
import respx

from app.config import Settings

PUBLIC_IP = "93.184.216.34"
RESOLVE_TABLE = {
    "private.test": ["10.0.0.5"],
    "metadata.test": ["169.254.169.254"],
    "loopback.test": ["127.0.0.1"],
    "mixed.test": [PUBLIC_IP, "192.168.1.10"],
}


async def fake_resolver(host: str) -> list[str]:
    """Deterministic DNS: every *.test host is public unless listed in RESOLVE_TABLE."""
    return RESOLVE_TABLE.get(host, [PUBLIC_IP])


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        crawl_per_domain_delay_seconds=0.0,
        crawl_concurrency=3,
        crawl_max_page_bytes=50_000,
        crawl_max_redirects=3,
        crawl_timeout_seconds=2.0,
        user_agent="GOLOKABot/0.1 (+https://example.test/bot)",
    )


@pytest.fixture
def resolver():
    return fake_resolver


@pytest.fixture
def mock():
    """Mocked network: any request without a registered route raises, so tests can never hit the web."""
    with respx.mock(assert_all_called=False) as router:
        yield router
