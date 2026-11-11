"""The sentence-transformers wrapper, tested with a fake `sentence_transformers` module.

The real model needs a ~90 MB download, so it is covered by an opt-in test:
    set GOLOKA_RUN_MODEL_TESTS=1  (Windows: `set GOLOKA_RUN_MODEL_TESTS=1`)  then  python -m pytest tests/test_embedder.py
"""
import os
import sys
import types

import numpy as np
import pytest

from app.search.embedder import DEFAULT_MODEL, EmbeddingError, SentenceTransformerEmbedder


class FakeModel:
    instances = 0

    def __init__(self, name, device=None):
        FakeModel.instances += 1
        self.name, self.device, self.calls = name, device, []

    def get_sentence_embedding_dimension(self):
        return 8

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.ones((len(texts), 8), dtype=np.float64) / np.sqrt(8)


@pytest.fixture
def fake_st(monkeypatch):
    FakeModel.instances = 0
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


def test_default_model_name():
    assert SentenceTransformerEmbedder().name == DEFAULT_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_model_is_loaded_lazily_and_once(fake_st):
    emb = SentenceTransformerEmbedder("m")
    assert FakeModel.instances == 0
    emb.embed(["a"])
    emb.embed(["b"])
    _ = emb.dimension
    assert FakeModel.instances == 1


def test_embed_requests_normalised_float32_vectors(fake_st):
    emb = SentenceTransformerEmbedder("m", batch_size=16)
    out = emb.embed(["x", "y", "z"])
    assert out.shape == (3, 8) and out.dtype == np.float32 and out.flags["C_CONTIGUOUS"]
    _texts, kwargs = emb._model.calls[0]
    assert kwargs["normalize_embeddings"] is True and kwargs["batch_size"] == 16 and kwargs["convert_to_numpy"] is True


def test_empty_input_gives_empty_matrix(fake_st):
    out = SentenceTransformerEmbedder("m").embed([])
    assert out.shape == (0, 8) and out.dtype == np.float32


def test_dimension(fake_st):
    assert SentenceTransformerEmbedder("m").dimension == 8


def test_missing_package_gives_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)   # makes `import` raise ImportError
    with pytest.raises(EmbeddingError, match="pip install sentence-transformers"):
        SentenceTransformerEmbedder("m").embed(["a"])


def test_model_download_failure_gives_helpful_error(monkeypatch):
    module = types.ModuleType("sentence_transformers")

    def boom(*a, **k):
        raise OSError("connection refused")

    module.SentenceTransformer = boom
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    with pytest.raises(EmbeddingError, match="internet connection"):
        SentenceTransformerEmbedder("m").embed(["a"])


def test_encode_failure_wrapped(fake_st, monkeypatch):
    emb = SentenceTransformerEmbedder("m")
    emb._load()
    monkeypatch.setattr(emb._model, "encode", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("oom")))
    with pytest.raises(EmbeddingError, match="oom"):
        emb.embed(["a"])


@pytest.mark.skipif(os.environ.get("GOLOKA_RUN_MODEL_TESTS") != "1", reason="downloads the real model (~90 MB)")
def test_real_model_end_to_end():
    emb = SentenceTransformerEmbedder()
    assert emb.dimension == 384
    vecs = emb.embed(["a fast automobile", "a quick car", "boiling pasta in salted water"])
    assert vecs.shape == (3, 384) and np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-4)
    assert float(vecs[0] @ vecs[1]) > float(vecs[0] @ vecs[2]) + 0.2     # meaning, not just words
