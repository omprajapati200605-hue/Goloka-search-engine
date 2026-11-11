"""Text analysis for keyword search: tokenization, stop-word handling, query processing.

The same analysis MUST be applied to documents and queries. Bump TOKENIZER_VERSION whenever the
rules below change; persisted keyword indexes built with another version are discarded and rebuilt.

Design notes
  * Unicode aware: NFKC normalization + case folding. Accents are kept ("café" != "cafe") because
    stripping combining marks would destroy Indic scripts such as Devanagari.
  * A token is a run of letters/digits (plus combining marks of Indic/Thai/Hebrew/Arabic/Latin),
    so "node.js" -> ["node", "js"], "snake_case" -> ["snake", "case"], "खोज इंजन" -> 2 tokens.
  * Stop-words are English only; other languages are tokenized but not filtered.
  * No stemming (exact word forms are matched; semantic search covers inflections).
"""
from __future__ import annotations

import re
import unicodedata

TOKENIZER_VERSION = 1
MAX_TOKEN_LENGTH = 64
MAX_QUERY_TERMS = 64

STOPWORDS: frozenset[str] = frozenset("""
a about above after again against all also am an and any are aren as at be because been before being
below between both but by can cannot could couldn d did didn do does doesn doing don down during each
few for from further had hadn has hasn have haven having he her here hers herself him himself his how
i if in into is isn it its itself just ll m me more most mustn my myself no nor not now of off on once
only or other our ours ourselves out over own re s same shan she should shouldn so some such t than
that the their theirs them themselves then there these they this those through to too under until up
ve very was wasn we were weren what when where which while who whom why will with won would wouldn y
you your yours yourself yourselves
""".split())

# letters/digits (no underscore) OR combining marks used by common scripts (excluding danda punctuation)
_WORD = re.compile(
    r"(?:[^\W_]|[̀-֑ͯ-ׇً-ٰٟऀ-ॣ०-९ॱ-෿฀-๿])+"
)


def normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").casefold()


def tokenize(text: str | None, *, remove_stopwords: bool = False) -> list[str]:
    """Split text into normalized tokens (duplicates kept: term frequency matters)."""
    if not text:
        return []
    tokens = [t for t in _WORD.findall(normalize(text)) if len(t) <= MAX_TOKEN_LENGTH]
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def analyze_document(text: str | None) -> list[str]:
    """Tokens that are stored in the index for a document field (stop-words removed)."""
    return tokenize(text, remove_stopwords=True)


def analyze_query(query: str | None, *, max_terms: int = MAX_QUERY_TERMS) -> list[str]:
    """Query processing: tokenize, drop stop-words, de-duplicate (order kept), cap the length.

    If the query consists only of stop-words ("to be or not to be") they are kept, so the query
    still produces terms instead of silently matching nothing."""
    tokens = tokenize(query)
    filtered = [t for t in tokens if t not in STOPWORDS] or tokens
    return list(dict.fromkeys(filtered))[:max_terms]
