"""answer(use_hybrid=True) uses retrieve_hybrid, not baseline/mmr. No LLM
or embedding calls: generate/embed_texts are monkeypatched to fakes."""

import numpy as np

from ragpkt import pipeline
from ragpkt.chunking import Chunk
from ragpkt.vectorstore import VectorStore


def make_chunk(id_: str, text: str) -> Chunk:
    return Chunk(id=id_, path="f.py", kind="function", name=id_, start_line=1, end_line=1, text=text)


def test_answer_with_hybrid_calls_retrieve_hybrid(monkeypatch):
    chunks = [
        make_chunk("a", "def generate_cached(): return kv_cache_logic()"),
        make_chunk("b", "def unrelated(): pass"),
    ]
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store = VectorStore(chunks, vectors)

    monkeypatch.setattr(pipeline, "retrieve_hybrid", lambda *a, **k: [(chunks[0], 1.0)])
    called = {}

    def fake_retrieve(*a, **k):
        called["baseline_was_called"] = True
        return [(chunks[1], 1.0)]

    monkeypatch.setattr(pipeline, "retrieve", fake_retrieve)
    monkeypatch.setattr(pipeline, "generate", lambda messages: "fake answer")

    result = pipeline.answer("how does generate_cached work", store, use_hybrid=True)

    assert "baseline_was_called" not in called
    assert result["retrieved"][0][1] == "a"  # chunk name, from the hybrid result, not baseline's


def test_answer_hybrid_wins_over_mmr_when_both_set(monkeypatch):
    chunks = [make_chunk("a", "x"), make_chunk("b", "y")]
    store = VectorStore(chunks, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))

    monkeypatch.setattr(pipeline, "retrieve_hybrid", lambda *a, **k: [(chunks[0], 1.0)])
    monkeypatch.setattr(pipeline, "retrieve_mmr", lambda *a, **k: [(chunks[1], 1.0)])
    monkeypatch.setattr(pipeline, "generate", lambda messages: "fake answer")

    result = pipeline.answer("q", store, use_hybrid=True, use_mmr=True)
    assert result["retrieved"][0][1] == "a"
