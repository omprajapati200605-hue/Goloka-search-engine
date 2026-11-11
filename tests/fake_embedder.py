"""Deterministic stand-in for a sentence-transformers model, used by tests only.

It hashes words into a fixed-size vector (feature hashing) and L2-normalises it, so cosine similarity
reflects shared vocabulary. An optional `concepts` map folds different words onto one concept, which
lets tests check retrieval of documents that share MEANING but no literal word with the query.
This is NOT a real language model; the real model is only exercised on the user's machine.
"""
from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np

from app.search.tokenizer import tokenize


class HashingEmbedder:
    def __init__(self, dim: int = 256, name: str = "hashing-test", concepts: dict[str, str] | None = None) -> None:
        self.name = name
        self._dim = dim
        self._concepts = concepts or {}
        self.embedded_texts: list[str] = []   # every text ever embedded (to assert "not re-embedded")

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            self.embedded_texts.append(text)
            for token in tokenize(text, remove_stopwords=True):
                token = self._concepts.get(token, token)
                out[row, int(hashlib.sha1(token.encode()).hexdigest()[:8], 16) % self._dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1.0, norms)
