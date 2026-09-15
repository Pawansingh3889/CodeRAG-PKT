"""answer()'s retriever selection: default is "hybrid", and each explicit
choice calls the right underlying function, not a different one. No LLM or
embedding calls: generate/retrieve* are monkeypatched to fakes."""

import numpy as np
import pytest

from ragpkt import pipeline
from ragpkt.chunking import Chunk
from ragpkt.vectorstore import VectorStore


def make_chunk(id_: str, text: str) -> Chunk:
    return Chunk(id=id_, path="f.py", kind="function", name=id_, start_line=1, end_line=1, text=text)


def make_store() -> VectorStore:
    chunks = [make_chunk("a", "def generate_cached(): return kv_cache_logic()"),
              make_chunk("b", "def unrelated(): pass")]
    return VectorStore(chunks, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))


def test_default_retriever_is_hybrid(monkeypatch):
    store = make_store()
    monkeypatch.setattr(pipeline, "retrieve_hybrid", lambda *a, **k: [(store.chunks[0], 1.0)])
    monkeypatch.setattr(pipeline, "generate", lambda messages: "fake answer")

    result = pipeline.answer("q", store)  # no retriever= passed
    assert result["retrieved"][0][1] == "a"


def test_retriever_hybrid_does_not_call_baseline(monkeypatch):
    store = make_store()
    called = {}
    monkeypatch.setattr(pipeline, "retrieve_hybrid", lambda *a, **k: [(store.chunks[0], 1.0)])
    monkeypatch.setattr(pipeline, "retrieve", lambda *a, **k: called.setdefault("baseline", True) or [])
    monkeypatch.setattr(pipeline, "generate", lambda messages: "fake answer")

    pipeline.answer("q", store, retriever="hybrid")
    assert "baseline" not in called


def test_retriever_baseline_calls_baseline(monkeypatch):
    store = make_store()
    monkeypatch.setattr(pipeline, "retrieve", lambda *a, **k: [(store.chunks[1], 1.0)])
    monkeypatch.setattr(pipeline, "generate", lambda messages: "fake answer")

    result = pipeline.answer("q", store, retriever="baseline")
    assert result["retrieved"][0][1] == "b"


def test_retriever_mmr_calls_mmr(monkeypatch):
    store = make_store()
    monkeypatch.setattr(pipeline, "retrieve_mmr", lambda *a, **k: [(store.chunks[1], 1.0)])
    monkeypatch.setattr(pipeline, "generate", lambda messages: "fake answer")

    result = pipeline.answer("q", store, retriever="mmr")
    assert result["retrieved"][0][1] == "b"


def test_unknown_retriever_raises():
    store = make_store()
    with pytest.raises(ValueError):
        pipeline.answer("q", store, retriever="nonexistent")
