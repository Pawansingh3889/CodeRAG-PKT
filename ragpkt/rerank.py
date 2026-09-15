"""
Milestone 7: LLM re-ranking.

Embedding similarity is a proxy for relevance, not the real thing — it
finds chunks that are *topically close* to the query, which isn't always
the same as *useful for answering it*. A cheap fix: ask the chat model
itself to score each retrieved chunk's relevance (1-10) against the
actual question, then keep only the top-scoring ones before building the
final prompt. Costs one extra (small) LLM call per query, in exchange for
noticeably fewer irrelevant chunks eating context budget.
"""

from __future__ import annotations

import re

from ragpkt.chunking import Chunk
from ragpkt.generate import generate

_RERANK_PROMPT = """\
Question: {question}

Chunk from {path} ({name}):
{text}

On a scale of 1-10, how relevant is this chunk to answering the question?
Reply with ONLY the number, nothing else."""


def _score_chunk(question: str, chunk: Chunk) -> int:
    messages = [{
        "role": "user",
        "content": _RERANK_PROMPT.format(
            question=question, path=chunk.path,
            name=chunk.name or "module", text=chunk.text[:1500],
        ),
    }]
    reply = generate(messages, temperature=0.0)
    match = re.search(r"\d+", reply)
    return int(match.group()) if match else 5  # default to "unsure/medium" on a bad parse


def rerank(question: str, candidates: list[tuple[Chunk, float]], keep: int = 5) -> list[tuple[Chunk, float]]:
    """Re-score each (chunk, embedding_score) pair with an LLM relevance
    judgment and return the top `keep`, sorted by that judgment."""
    scored = [(chunk, _score_chunk(question, chunk)) for chunk, _ in candidates]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [(chunk, float(score)) for chunk, score in scored[:keep]]
