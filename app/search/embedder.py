"""Embedding models.

`Embedder` is the small interface the vector index depends on, so tests (and future models) can
plug in without sentence-transformers. `SentenceTransformerEmbedder` is the production
implementation; the model is loaded lazily on first use (the first run downloads ~90 MB for
all-MiniLM-L6-v2 from huggingface.co and caches it).
"""
from __future__ import annotations

import threading
from typing import Protocol, Sequence

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingError(RuntimeError):
    """The embedding model could not be loaded or failed to encode."""


class Embedder(Protocol):
    name: str

    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dimension), rows L2-normalised."""


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, *, batch_size: int = 32, device: str | None = None) -> None:
        self.name = model_name
        self._batch_size = batch_size
        self._device = device
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise EmbeddingError(
                        "sentence-transformers is not installed. Run: pip install sentence-transformers") from exc
                try:
                    self._model = SentenceTransformer(self.name, device=self._device)
                except Exception as exc:  # noqa: BLE001 - network, disk, bad model name ...
                    raise EmbeddingError(
                        f"Could not load embedding model {self.name!r}: {exc}. The first run downloads the "
                        "model from huggingface.co, so an internet connection is required once.") from exc
            return self._model

    @property
    def dimension(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        model = self._load()
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        try:
            vectors = model.encode(
                list(texts), batch_size=self._batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001
            raise EmbeddingError(f"Embedding failed: {exc}") from exc
        return np.ascontiguousarray(vectors, dtype=np.float32)
