"""robots.txt parsing (RFC 9309 semantics) and per-origin caching.

The stdlib urllib.robotparser ignores `*` / `$` wildcards in paths, which would make us crawl
pages that the site disallowed, so a small compliant parser is implemented here.

Fetch-failure policy (conservative):
  * 200            -> parse and obey
  * 404 / 410 / other 4xx (except 401/403) -> no robots.txt, everything allowed
  * 401 / 403      -> treated as restricted, everything disallowed
  * 5xx, timeout, network or SSRF error -> everything disallowed
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable
from urllib.parse import urlsplit

MAX_CRAWL_DELAY = 30.0


@dataclass
class RobotsRules:
    rules: list[tuple[bool, re.Pattern[str], int]] = field(default_factory=list)  # (allow, regex, specificity)
    crawl_delay: float | None = None
    disallow_all: bool = False

    def is_allowed(self, path_and_query: str) -> bool:
        if self.disallow_all:
            return False
        best: tuple[int, bool] | None = None
        for allow, pattern, specificity in self.rules:
            if pattern.match(path_and_query):
                candidate = (specificity, allow)  # longest match wins; Allow wins ties
                if best is None or candidate > best:
                    best = candidate
        return True if best is None else best[1]


def _compile(path: str) -> tuple[re.Pattern[str], int]:
    anchored = path.endswith("$")
    if anchored:
        path = path[:-1]
    regex = ".*".join(re.escape(part) for part in path.split("*"))
    return re.compile(regex + ("$" if anchored else "")), len(path)


def parse_robots(text: str, agent_token: str) -> RobotsRules:
    token = agent_token.lower()
    groups: list[tuple[set[str], list[tuple[bool, str]], float | None]] = []
    agents: set[str] = set()
    rules: list[tuple[bool, str]] = []
    delay: float | None = None
    in_rules = False

    def flush() -> None:
        if agents:
            groups.append((set(agents), list(rules), delay))

    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if ":" not in line:
            continue
        key, value = (p.strip() for p in line.split(":", 1))
        key = key.lower()
        if key == "user-agent":
            if in_rules:
                flush()
                agents, rules, delay, in_rules = set(), [], None, False
            agents.add(value.lower())
        elif key in ("allow", "disallow"):
            in_rules = True
            if value:  # empty Disallow means "allow everything"
                rules.append((key == "allow", value))
        elif key == "crawl-delay":
            in_rules = True
            try:
                delay = max(0.0, float(value))
            except ValueError:
                pass
    flush()

    specific = [g for g in groups if token in g[0]]
    chosen = specific or [g for g in groups if "*" in g[0]]
    result = RobotsRules()
    for _agents, g_rules, g_delay in chosen:
        for allow, path in g_rules:
            pattern, spec = _compile(path)
            result.rules.append((allow, pattern, spec))
        if g_delay is not None:
            result.crawl_delay = g_delay
    return result


def agent_token(user_agent: str) -> str:
    """'GOLOKABot/0.1 (+https://...)' -> 'golokabot'"""
    return user_agent.split("/", 1)[0].split(" ", 1)[0].strip().lower()


class RobotsPolicy:
    """Fetches, caches and applies robots.txt per origin."""

    def __init__(self, fetch: Callable[[str], Awaitable["FetchResultLike"]], user_agent: str) -> None:
        self._fetch = fetch
        self._token = agent_token(user_agent)
        self._cache: dict[str, RobotsRules] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _rules_for(self, url: str) -> RobotsRules:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin in self._cache:
            return self._cache[origin]
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            if origin in self._cache:
                return self._cache[origin]
            result = await self._fetch(f"{origin}/robots.txt")
            if result.error_kind is None and result.status_code == 200:
                rules = parse_robots(result.text, self._token)
            elif result.status_code is not None and 400 <= result.status_code < 500 \
                    and result.status_code not in (401, 403):
                rules = RobotsRules()  # no robots.txt -> allow all
            else:
                rules = RobotsRules(disallow_all=True)
            self._cache[origin] = rules
            return rules

    async def allowed(self, url: str) -> bool:
        rules = await self._rules_for(url)
        parts = urlsplit(url)
        target = (parts.path or "/") + (f"?{parts.query}" if parts.query else "")
        return rules.is_allowed(target)

    async def crawl_delay(self, url: str) -> float:
        rules = await self._rules_for(url)
        return min(rules.crawl_delay or 0.0, MAX_CRAWL_DELAY)


class FetchResultLike:  # structural type only, for annotations
    error_kind: str | None
    status_code: int | None
    text: str
