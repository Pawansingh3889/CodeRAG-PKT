"""
Retrieval quality metrics: precision@k and recall@k against a hand-labeled
set of (question -> expected chunk ids). Separated from eval/eval_retrieval.py
so the metric math itself is unit-testable without needing a real index or
an OpenAI key: a fixture with fake chunk ids exercises the exact same code
path as a live run.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalScore:
    precision: float
    recall: float
    hit: bool  # at least one expected chunk was retrieved at all


def precision_recall_at_k(retrieved_ids: list[str], expected_ids: list[str]) -> RetrievalScore:
    """retrieved_ids: chunk ids returned by a retriever, already truncated to k.
    expected_ids: the chunk id(s) considered a correct answer for this question.

    precision: fraction of the retrieved chunks that were actually relevant.
    recall: fraction of the relevant chunks that were retrieved.
    Both are computed over sets, so a duplicate id in `retrieved_ids` (which
    shouldn't happen, but isn't assumed away) doesn't double-count.
    """
    if not expected_ids:
        raise ValueError("expected_ids must be non-empty; an unlabeled question isn't scoreable")

    retrieved_set = set(retrieved_ids)
    expected_set = set(expected_ids)
    hits = retrieved_set & expected_set

    precision = len(hits) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(hits) / len(expected_set)
    return RetrievalScore(precision=precision, recall=recall, hit=bool(hits))


def average_scores(scores: list[RetrievalScore]) -> dict[str, float]:
    """Mean precision/recall/hit-rate across a list of per-question scores."""
    if not scores:
        return {"precision": 0.0, "recall": 0.0, "hit_rate": 0.0}
    n = len(scores)
    return {
        "precision": sum(s.precision for s in scores) / n,
        "recall": sum(s.recall for s in scores) / n,
        "hit_rate": sum(1 for s in scores if s.hit) / n,
    }
