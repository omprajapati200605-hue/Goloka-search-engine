import pytest

from app.crawler.url_utils import (
    DNSResolutionError,
    SSRFError,
    URLValidationError,
    assert_public_host,
    domain_allowed,
    is_public_ip,
    is_restricted_path,
    looks_like_html_url,
    normalize_domain,
    normalize_url,
    validate_url,
)


# ---- normalization -------------------------------------------------------------------------
def test_normalize_basic():
    assert normalize_url("HTTPS://Example.COM:443/a/b?z=1&a=2#frag") == "https://example.com/a/b?a=2&z=1"


def test_normalize_adds_root_path_and_drops_default_port():
    assert normalize_url("http://example.com:80") == "http://example.com/"
    assert normalize_url("http://example.com:8080") == "http://example.com:8080/"


def test_normalize_removes_tracking_params():
    assert normalize_url("https://a.com/p?utm_source=x&id=7&fbclid=abc") == "https://a.com/p?id=7"


def test_normalize_relative_with_base():
    assert normalize_url("../x?y=1", "https://a.com/d/e/f") == "https://a.com/d/x?y=1"
    assert normalize_url("//b.com/z", "https://a.com/") == "https://b.com/z"


def test_normalize_idna_host():
    assert normalize_url("https://bücher.example/").startswith("https://xn--bcher-kva.example/")


def test_normalize_rejects_relative_without_base():
    with pytest.raises(URLValidationError):
        normalize_url("/just/a/path")


# ---- validation ----------------------------------------------------------------------------
@pytest.mark.parametrize("url", [
    "https://example.com", "http://example.com/a?b=1", "https://sub.example.co.uk:8443/x",
])
def test_validate_accepts_public_urls(url):
    assert validate_url(url).startswith("http")


@pytest.mark.parametrize("url", [
    "", "   ", "ftp://example.com/", "file:///etc/passwd", "javascript:alert(1)", "example.com",
    "https://user:pw@example.com/", "https://example.com:22/", "https://", "http://exa mple.com",
    "https://example.com/" + "a" * 3000,
])
def test_validate_rejects_bad_urls(url):
    with pytest.raises(URLValidationError):
        validate_url(url)


@pytest.mark.parametrize("url", [
    "http://localhost/", "http://127.0.0.1/", "http://[::1]/", "http://10.0.0.1/", "http://192.168.1.1/",
    "http://172.16.0.1/", "http://169.254.169.254/latest/meta-data/", "http://100.64.0.1/",
    "http://0.0.0.0/", "http://[::ffff:127.0.0.1]/", "http://2130706433/", "http://intranet/",
    "http://printer.local/", "http://service.internal/", "http://foo.localhost/",
])
def test_validate_blocks_ssrf_targets(url):
    with pytest.raises(SSRFError):
        validate_url(url)


def test_ssrf_error_is_validation_error():
    assert issubclass(SSRFError, URLValidationError)


@pytest.mark.parametrize("ip,expected", [
    ("93.184.216.34", True), ("8.8.8.8", True), ("2606:4700:4700::1111", True),
    ("10.1.2.3", False), ("127.0.0.1", False), ("169.254.1.1", False), ("100.64.0.1", False),
    ("::1", False), ("fe80::1", False), ("fc00::1", False), ("224.0.0.1", False),
    ("::ffff:10.0.0.1", False), ("not-an-ip", False),
])
def test_is_public_ip(ip, expected):
    assert is_public_ip(ip) is expected


# ---- DNS-level SSRF checks -----------------------------------------------------------------
async def test_assert_public_host_ok(resolver):
    await assert_public_host("site.test", resolver)


@pytest.mark.parametrize("host", ["private.test", "metadata.test", "loopback.test", "mixed.test"])
async def test_assert_public_host_blocks_private_resolution(host, resolver):
    with pytest.raises(SSRFError):
        await assert_public_host(host, resolver)


async def test_assert_public_host_literal_private_ip(resolver):
    with pytest.raises(SSRFError):
        await assert_public_host("10.0.0.1", resolver)


async def test_assert_public_host_empty_resolution():
    async def none(_host):
        return []
    with pytest.raises(DNSResolutionError):
        await assert_public_host("x.test", none)


# ---- scope helpers -------------------------------------------------------------------------
def test_domain_allowed():
    allowed = {"example.com"}
    assert domain_allowed("example.com", allowed)
    assert domain_allowed("blog.example.com", allowed)
    assert not domain_allowed("evil-example.com", allowed)
    assert not domain_allowed("example.com.evil.net", allowed)


def test_normalize_domain():
    assert normalize_domain(" Example.COM. ") == "example.com"
    for bad in ["", "https://a.com", "a.com/path", "localhost", "a b.com"]:
        with pytest.raises(ValueError):
            normalize_domain(bad)


def test_html_url_and_restricted_path_heuristics():
    assert looks_like_html_url("https://a.com/page") and looks_like_html_url("https://a.com/p.html")
    assert not looks_like_html_url("https://a.com/file.PDF")
    assert is_restricted_path("https://a.com/login") and is_restricted_path("https://a.com/x/admin/y")
    assert not is_restricted_path("https://a.com/blog/authors")
