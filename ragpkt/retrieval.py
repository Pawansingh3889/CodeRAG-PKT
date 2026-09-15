"""
Milestone 4: retrieval.

Plain top-k cosine search is the baseline. Its failure mode shows up on
repos with duplication or near-duplicate chunks (e.g. three functions
that all wrap the same API call): the top-k list ends up full of near-
identical chunks, wasting context budget instead of covering the
question from different angles.

MMR (Maximal Marginal Relevance) fixes that by picking chunks greedily,
trading off "relevant to the query" against "different from what's
already picked" — a classic IR technique, ~15 lines of numpy once you
already have a vector store.
"""

from __future__ import annotations

import numpy as np

from ragpkt.chunking import Chunk
from ragpkt.embeddings import embed_query
from ragpkt.vectorstore import VectorStore


def retrieve(query: str, store: VectorStore, k: int = 5) -> list[tuple[Chunk, float]]:
    """Baseline retrieval: top-k by cosine similarity."""
    qvec = embed_query(query)
    return store.search(qvec, k=k)


def retrieve_mmr(
    query: str,
    store: VectorStore,
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
) -> list[tuple[Chunk, float]]:
    """MMR retrieval: diversify the top-k instead of taking pure top-k.

    1. Pull the `fetch_k` most relevant candidates (cheap, coarse pass).
    2. Greedily pick `k` of them, each time favoring a candidate that's
       both relevant to the query (`lambda_mult`) AND far from what's
       already been selected (`1 - lambda_mult`).

    lambda_mult=1.0 reduces to plain top-k; lower values push harder for
    diversity across the picked chunks.
    """
    qvec = embed_query(query)
    candidates = store.search(qvec, k=min(fetch_k, len(store)))
    if not candidates:
        return []

    cand_chunks = [c for c, _ in candidates]
    cand_idx = [store.chunks.index(c) for c in cand_chunks]  # positions in store.vectors
    cand_vecs = store.vectors[cand_idx]
    q = qvec / max(np.linalg.norm(qvec), 1e-8)
    relevance = cand_vecs @ q  # cosine sim to query, same as candidates' scores

    selected: list[int] = []
    remaining = list(range(len(cand_chunks)))

    while remaining and len(selected) < k:
        if not selected:
            # first pick: purely most relevant
            best = max(remaining, key=lambda i: relevance[i])
        else:
            selected_vecs = cand_vecs[selected]
            best = None
            best_score = -1e9
            for i in remaining:
                redundancy = float(np.max(cand_vecs[i] @ selected_vecs.T))
                mmr_score = lambda_mult * relevance[i] - (1 - lambda_mult) * redundancy
                if mmr_score > best_score:
                    best_score, best = mmr_score, i
        selected.append(best)
        remaining.remove(best)

    return [(cand_chunks[i], float(relevance[i])) for i in selected]
