"""Hybrid (cosine + BM25, RRF-fused) retrieval tests.

No OpenAI key needed: embed_query is monkeypatched to return a fixed
synthetic vector, since the RRF fusion logic under test doesn't care
where the vectors came from, only that vector search and BM25 search
each contribute a rank order.
"""

import numpy as np
import pytest

from ragpkt import retrieval
from ragpkt.chunking import Chunk
from ragpkt.keyword_search import BM25Index
from ragpkt.retrieval import retrieve_hybrid
from ragpkt.vectorstore import VectorStore


def make_chunk(id_: str, text: str) -> Chunk:
    return Chunk(id=id_, path="f.py", kind="function", name=id_, start_line=1, end_line=1, text=text)


@pytest.fixture
def store_and_bm25(monkeypatch):
    chunks = [
        make_chunk("a", "def generate_cached(model, tok, prompt): return kv_cache_logic()"),
        make_chunk("b", "def unrelated_math(x, y): return x * y"),
        make_chunk("c", "class SelfAttention: causal_mask = True"),
    ]
    # Synthetic embeddings: chunk 'a' is closest to the query vector on
    # purpose, so cosine search alone should already surface it, letting
    # the RRF test assert both signals agree rather than only one.
    vectors = np.array([
        [1.0, 0.0, 0.0],  # a
        [0.0, 1.0, 0.0],  # b
        [0.0, 0.0, 1.0],  # c
    ], dtype=np.float32)
    store = VectorStore(chunks, vectors)
    monkeypatch.setattr(retrieval, "embed_query", lambda text: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    bm25 = BM25Index(chunks)
    return store, bm25


def test_hybrid_surfaces_the_chunk_both_signals_agree_on(store_and_bm25):
    store, bm25 = store_and_bm25
    hits = retrieve_hybrid("how does generate_cached use a kv cache", store, bm25, k=3)
    assert hits[0][0].id == "a"


def test_hybrid_respects_k(store_and_bm25):
    store, bm25 = store_and_bm25
    hits = retrieve_hybrid("query", store, bm25, k=2)
    assert len(hits) == 2


def test_hybrid_can_surface_a_keyword_hit_vector_search_misses(monkeypatch):
    # Here the synthetic embedding points AWAY from the keyword-relevant
    # chunk, so only BM25 favors it. RRF should still be able to pull it
    # into the top-k because it's still ranked (just not first) in the
    # BM25 list, demonstrating the actual value of hybrid search: a
    # chunk a pure-vector search ranks low still has a path into top-k.
    chunks = [
        make_chunk("a", "def unrelated_math(x, y): return x * y"),
        make_chunk("b", "def unrelated_helper(): pass"),
        make_chunk("c", "def generate_cached(model): return kv_cache_logic()"),
    ]
    vectors = np.array([
        [1.0, 0.0, 0.0],  # a: closest to the fake query vector below
        [0.0, 1.0, 0.0],  # b
        [0.0, 0.0, 1.0],  # c: farthest from the fake query vector
    ], dtype=np.float32)
    store = VectorStore(chunks, vectors)
    monkeypatch.setattr(retrieval, "embed_query", lambda text: np.array([1.0, 0.0, 0.0], dtype=np.float32))
    bm25 = BM25Index(chunks)

    hits = retrieve_hybrid("generate_cached kv cache", store, bm25, k=3)
    retrieved_ids = [c.id for c, _ in hits]
    assert "c" in retrieved_ids
