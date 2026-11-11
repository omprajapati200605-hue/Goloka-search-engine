"""Split document text into overlapping, sentence-aligned chunks for embedding.

Embedding models see a limited window (all-MiniLM-L6-v2: ~256 tokens), and one vector per long page
blurs its topics together. Chunks of ~`size_words` words (default 200) each get their own vector;
search results then point at the best-matching chunk.

Rules
  * Text is split into lines, then sentences (. ! ? and the Devanagari danda), so chunks do not cut
    sentences in half unless a single sentence is longer than a whole chunk (then it is split by words).
  * Chunks are packed greedily up to `size_words`. Size is measured as max(words, characters / 6), so
    text without spaces (or huge "words") still produces bounded chunks.
  * Consecutive chunks overlap by up to `overlap_words` (whole sentences when they fit, otherwise the
    last words of the previous chunk), so context at a boundary is not lost.
"""
from __future__ import annotations

import re

CHUNKER_VERSION = 2
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")
_CHARS_PER_WORD = 6


def _weight(text: str) -> int:
    """Approximate size in 'words': the larger of the word count and characters / 6."""
    return max(len(text.split()), -(-len(text) // _CHARS_PER_WORD), 1)


def _split_long(sentence: str, step: int) -> list[str]:
    """Break an over-long sentence into pieces of at most `step` weight (words, or character windows
    for giant unspaced words)."""
    pieces: list[str] = []
    current: list[str] = []
    current_weight = 0
    window = step * _CHARS_PER_WORD
    for word in sentence.split():
        for part in (word[i:i + window] for i in range(0, len(word), window)):
            w = _weight(part)
            if current and current_weight + w > step:
                pieces.append(" ".join(current))
                current, current_weight = [], 0
            current.append(part)
            current_weight += w
    if current:
        pieces.append(" ".join(current))
    return pieces


def _units(text: str, size_words: int, overlap_words: int) -> list[str]:
    # pieces of a split sentence are (size - overlap) wide so the carried overlap always fits beside them
    step = max(1, size_words - overlap_words)
    units: list[str] = []
    for line in text.splitlines():
        for sentence in _SENTENCE_SPLIT.split(line.strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            if _weight(sentence) > size_words:
                units.extend(_split_long(sentence, step))
            else:
                units.append(sentence)
    return units


def split_into_chunks(text: str | None, *, size_words: int = 200, overlap_words: int = 40) -> list[str]:
    if size_words < 1 or overlap_words < 0 or overlap_words >= size_words:
        raise ValueError("require size_words >= 1 and 0 <= overlap_words < size_words")
    if not text or not text.strip():
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_weight = 0
    for unit in _units(text, size_words, overlap_words):
        n = _weight(unit)
        if current and current_weight + n > size_words:
            chunks.append(" ".join(current))
            carry: list[str] = []
            carried = 0
            for previous in reversed(current):              # carry whole trailing sentences
                pw = _weight(previous)
                if carried + pw > overlap_words:
                    break
                carry.insert(0, previous)
                carried += pw
            if not carry and overlap_words > 0:              # sentence too long: carry its last words
                tail = " ".join(current[-1].split()[-overlap_words:])
                carry, carried = [tail], _weight(tail)
            while carry and carried + n > size_words:        # make room for the new sentence
                carried -= _weight(carry.pop(0))
            current, current_weight = carry, carried
        current.append(unit)
        current_weight += n
    if current:
        chunks.append(" ".join(current))
    return chunks
