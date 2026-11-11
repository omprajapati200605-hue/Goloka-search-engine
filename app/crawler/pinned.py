"""Connection-time SSRF protection (closes the DNS-rebinding gap).

Checking a hostname and then letting the HTTP client resolve it again leaves a window in which a hostile DNS
server can answer "public" for the check and "10.0.0.5" for the connection. Here the connection itself is
made by `PinnedBackend`: it resolves the host once, requires EVERY returned address to be public, and connects
to one of those exact IP literals. TLS still validates the certificate against the original hostname (httpcore
takes the server name from the request URL, not from the address it dials).

This replaces the network backend inside httpx's transport via a private attribute (`_pool._network_backend`).
If a future httpx/httpcore release removes it, `make_pinned_transport` raises instead of silently running
without the protection (tests/test_pinned.py also pins this).
"""
from __future__ import annotations

import ipaddress
from typing import Callable, Iterable

import httpcore
import httpx

from app.crawler.url_utils import Resolver, is_public_ip, system_resolver


class PinnedConnectBlocked(httpcore.ConnectError):
    """Raised when a connection would go to a non-public address (or the name does not resolve)."""


class PinnedBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, inner: httpcore.AsyncNetworkBackend, resolver: Resolver = system_resolver,
                 public_check: Callable[[str], bool] = is_public_ip) -> None:
        self._inner = inner
        self._resolve = resolver
        self._is_public = public_check

    async def connect_tcp(self, host: str, port: int, timeout: float | None = None,
                          local_address: str | None = None, socket_options: Iterable | None = None):
        try:
            ipaddress.ip_address(host)
            ips = [host]
        except ValueError:
            try:
                ips = list(await self._resolve(host))
            except Exception as exc:  # noqa: BLE001 - any resolver failure means we do not connect
                raise PinnedConnectBlocked(f"could not resolve {host}: {exc}") from exc
        if not ips:
            raise PinnedConnectBlocked(f"{host} did not resolve to any address")
        bad = [ip for ip in ips if not self._is_public(ip)]
        if bad:
            raise PinnedConnectBlocked(f"{host} resolves to a non-public address ({bad[0]}); connection refused")
        last: Exception | None = None
        for ip in ips:                         # only validated literals are ever dialled
            try:
                return await self._inner.connect_tcp(ip, port, timeout=timeout, local_address=local_address,
                                                     socket_options=socket_options)
            except httpcore.ConnectError as exc:
                last = exc
        assert last is not None
        raise last

    async def connect_unix_socket(self, path, timeout=None, socket_options=None):
        raise PinnedConnectBlocked("unix sockets are not allowed")

    async def sleep(self, seconds: float) -> None:
        await self._inner.sleep(seconds)


def make_pinned_transport(resolver: Resolver = system_resolver,
                          public_check: Callable[[str], bool] = is_public_ip) -> httpx.AsyncHTTPTransport:
    transport = httpx.AsyncHTTPTransport()
    pool = getattr(transport, "_pool", None)
    inner = getattr(pool, "_network_backend", None)
    if inner is None:
        raise RuntimeError("this httpx/httpcore version is not supported: cannot pin connections to validated IPs")
    pool._network_backend = PinnedBackend(inner, resolver, public_check)
    return transport
