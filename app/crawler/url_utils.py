"""URL normalization, validation and SSRF protection.

Security model
--------------
* Only http/https URLs, no embedded credentials, only a small set of ports.
* Hostnames that are obviously internal (localhost, *.local, single-label names) are rejected.
* Literal IPs must be globally routable (no private, loopback, link-local, CGNAT, reserved).
* Hostnames are resolved and EVERY resolved address must be public (assert_public_host).
  This runs again for every redirect hop (see fetcher.py).
"""
from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from typing import Awaitable, Callable, Iterable
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit

MAX_URL_LENGTH = 2048
DEFAULT_ALLOWED_PORTS = frozenset({80, 443, 8080, 8443})

_TRACKING_PARAMS = {"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "igshid", "ref_src"}

_BLOCKED_HOST_SUFFIXES = (
    ".localhost", ".local", ".internal", ".localdomain", ".lan", ".home.arpa", ".intranet", ".corp",
)

NON_HTML_EXTENSIONS = frozenset({
    "pdf", "jpg", "jpeg", "png", "gif", "svg", "webp", "ico", "bmp", "css", "js", "json", "xml",
    "zip", "gz", "tar", "rar", "7z", "mp3", "mp4", "avi", "mov", "wmv", "webm", "woff", "woff2",
    "ttf", "eot", "exe", "dmg", "msi", "apk", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "csv",
})

# Path segments that indicate login-protected / session-specific areas. Exact segment match only.
RESTRICTED_SEGMENTS = frozenset({
    "login", "signin", "sign-in", "log-in", "logout", "signout", "sign-out", "register", "signup",
    "sign-up", "my-account", "account", "admin", "wp-admin", "wp-login.php", "checkout", "cart",
    "oauth", "sso",
})


_HOSTNAME_RE = re.compile(r"[a-z0-9_]([a-z0-9_-]{0,61}[a-z0-9_])?(\.[a-z0-9_]([a-z0-9_-]{0,61}[a-z0-9_])?)*")


class URLValidationError(ValueError):
    """The URL is malformed or not allowed."""


class SSRFError(URLValidationError):
    """The URL points at (or resolves to) a non-public network location."""


class DNSResolutionError(Exception):
    """The hostname could not be resolved."""


Resolver = Callable[[str], Awaitable[list[str]]]


# --------------------------------------------------------------------------- normalization
def normalize_url(url: str, base: str | None = None) -> str:
    """Return a canonical absolute URL: lowercase host, no fragment, default port dropped,
    tracking parameters removed, query parameters sorted. Raises URLValidationError."""
    try:
        raw = urljoin(base, url.strip()) if base else url.strip()
        parts = urlsplit(raw)
        scheme = parts.scheme.lower()
        host = parts.hostname
        port = parts.port
    except ValueError as exc:
        raise URLValidationError(f"Malformed URL: {exc}") from exc
    if not scheme or not host:
        raise URLValidationError("URL must be absolute and contain a host")

    host = host.lower().rstrip(".")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise URLValidationError("Invalid hostname encoding") from exc

    hostpart = f"[{host}]" if ":" in host else host
    default_port = {"http": 80, "https": 443}.get(scheme)
    netloc = hostpart if (port is None or port == default_port) else f"{hostpart}:{port}"

    pairs = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS and not k.lower().startswith("utm_")
    ]
    query = urlencode(sorted(pairs))
    path = parts.path or "/"
    return f"{scheme}://{netloc}{path}" + (f"?{query}" if query else "")


# --------------------------------------------------------------------------- IP checks
def is_public_ip(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    return addr.is_global and not addr.is_multicast


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def _blocked_hostname(host: str) -> bool:
    if host == "localhost" or host.endswith(_BLOCKED_HOST_SUFFIXES):
        return True
    return "." not in host and not _is_ip_literal(host)  # single-label names are internal


# --------------------------------------------------------------------------- validation
def validate_url(url: str, *, allowed_ports: Iterable[int] = DEFAULT_ALLOWED_PORTS) -> str:
    """Validate a URL and return its normalized form. Raises URLValidationError / SSRFError.

    This is a *syntactic* check plus literal-IP check. DNS-based checks happen in
    assert_public_host(), which must be awaited before connecting."""
    if not isinstance(url, str) or not url.strip():
        raise URLValidationError("URL is empty")
    url = url.strip()
    if len(url) > MAX_URL_LENGTH:
        raise URLValidationError("URL is too long")
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise URLValidationError(f"Malformed URL: {exc}") from exc

    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        raise URLValidationError("Only http and https URLs are allowed")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise URLValidationError("URLs with embedded credentials are not allowed")
    if not parts.hostname:
        raise URLValidationError("URL has no host")
    effective_port = port or (443 if scheme == "https" else 80)
    if effective_port not in set(allowed_ports):
        raise URLValidationError(f"Port {effective_port} is not allowed")

    normalized = normalize_url(url)
    host = urlsplit(normalized).hostname or ""
    if _is_ip_literal(host):
        if not is_public_ip(host):
            raise SSRFError(f"Non-public IP address is not allowed: {host}")
    elif not _HOSTNAME_RE.fullmatch(host):
        raise URLValidationError(f"Invalid hostname: {host!r}")
    elif _blocked_hostname(host):
        raise SSRFError(f"Internal hostname is not allowed: {host}")
    return normalized


async def system_resolver(host: str) -> list[str]:
    """Resolve a hostname to all of its IP addresses using the system resolver."""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise DNSResolutionError(f"Cannot resolve {host}: {exc}") from exc
    return sorted({info[4][0] for info in infos})


async def assert_public_host(host: str, resolver: Resolver = system_resolver) -> None:
    """Raise SSRFError unless `host` is a public IP or resolves ONLY to public IPs."""
    if _is_ip_literal(host):
        if not is_public_ip(host):
            raise SSRFError(f"Non-public IP address is not allowed: {host}")
        return
    ips = await resolver(host)
    if not ips:
        raise DNSResolutionError(f"{host} did not resolve to any address")
    bad = [ip for ip in ips if not is_public_ip(ip)]
    if bad:
        raise SSRFError(f"{host} resolves to a non-public address ({bad[0]})")


# --------------------------------------------------------------------------- scope helpers
def normalize_domain(domain: str) -> str:
    d = domain.strip().lower().rstrip(".")
    if not d or "/" in d or ":" in d or " " in d or "." not in d:
        raise ValueError(f"Invalid domain: {domain!r}")
    return d.encode("idna").decode("ascii")


def domain_allowed(host: str, allowed: Iterable[str]) -> bool:
    """True if host equals an allowed domain or is a subdomain of one."""
    host = host.lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in allowed)


def looks_like_html_url(url: str) -> bool:
    path = urlsplit(url).path
    last = path.rsplit("/", 1)[-1]
    if "." not in last:
        return True
    return last.rsplit(".", 1)[-1].lower() not in NON_HTML_EXTENSIONS


def is_restricted_path(url: str) -> bool:
    segments = [s.lower() for s in urlsplit(url).path.split("/") if s]
    return any(s in RESTRICTED_SEGMENTS for s in segments)


_WS = re.compile(r"\s+")


def collapse_whitespace(text: str) -> str:
    return _WS.sub(" ", text).strip()
