"""Result snippets: the passage of a document that best shows why it matched the query.

Strategy
  1. Find query-term occurrences in the original text (matching is accent/case-insensitive and uses the
     same tokenizer as the index).
  2. Pick the window of `max_chars` that contains the most distinct query terms (then most matches).
  3. Cut on word boundaries and mark truncation with "…".
  4. If no query term occurs in the text (e.g. a purely semantic match) use `fallback` (typically the best
     semantic chunk), otherwise the start of the document.
"""
from __future__ import annotations

from app.search.tokenizer import MAX_TOKEN_LENGTH, _WORD, analyze_query, normalize

ELLIPSIS = "…"
_MAX_ANCHORS = 200


def truncate(text: str, max_chars: int) -> str:
    flat = " ".join((text or "").split())
    if len(flat) <= max_chars:
        return flat
    cut = flat.rfind(" ", 0, max_chars)
    return flat[: cut if cut > max_chars // 2 else max_chars].rstrip() + ELLIPSIS


def build_snippet(text: str | None, query: str, *, max_chars: int = 240, fallback: str | None = None) -> str:
    flat = " ".join((text or "").split())
    terms = set(analyze_query(query))
    spans: list[tuple[int, int, str]] = []
    if terms:
        for m in _WORD.finditer(flat):
            word = m.group()
            if len(word) <= MAX_TOKEN_LENGTH and (norm := normalize(word)) in terms:
                spans.append((m.start(), m.end(), norm))

    if not spans:
        return truncate(fallback, max_chars) if fallback and fallback.strip() else truncate(flat, max_chars)
    if len(flat) <= max_chars:
        return flat

    best: tuple[tuple[int, int, int], int] | None = None
    for start, _end, _term in spans[:_MAX_ANCHORS]:
        win_start = max(0, start - max_chars // 4)
        win_end = win_start + max_chars
        inside = [s for s in spans if win_start <= s[0] and s[1] <= win_end]
        key = (len({s[2] for s in inside}), len(inside), -win_start)
        if best is None or key > best[0]:
            best = (key, win_start)
    win_start = best[1]
    win_end = min(len(flat), win_start + max_chars)

    # tidy the edges to word boundaries, but never lose every match
    tidy_start, tidy_end = win_start, win_end
    if tidy_start > 0 and flat[tidy_start - 1] != " ":
        nxt = flat.find(" ", tidy_start)
        tidy_start = nxt + 1 if nxt != -1 else tidy_start
    if tidy_end < len(flat) and flat[tidy_end] != " ":
        prv = flat.rfind(" ", tidy_start, tidy_end)
        tidy_end = prv if prv > tidy_start else tidy_end
    if any(tidy_start <= s[0] and s[1] <= tidy_end for s in spans):
        win_start, win_end = tidy_start, tidy_end

    snippet = flat[win_start:win_end].strip()
    return ("" if win_start == 0 else ELLIPSIS) + snippet + ("" if win_end >= len(flat) else ELLIPSIS)
