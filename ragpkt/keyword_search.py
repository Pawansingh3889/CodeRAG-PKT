"""
Milestone 10: keyword search (BM25), for hybrid retrieval.

Embedding search finds chunks that are semantically close to a query, but
can miss an exact identifier match: a query naming `RESUME_FROM` literally
should find the chunk containing that literal string, even when the
embedding space doesn't rank it first (a common failure mode on code,
where two functions can be semantically similar but only one has the
name you actually typed). BM25 is the classical fix: a TF-IDF variant
with term-frequency saturation and document-length normalization,
~50 lines of plain Python, no dependency, matching the rest of this
repo's "no vector DB, no hidden library" premise.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from ragpkt.chunking import Chunk

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(text: str) -> list[str]:
    """Lowercased identifier/word tokens. Underscore-separated names split
    naturally on the regex boundary; camelCase is left whole rather than
    guessed apart, since splitting it wrong would hurt exact-name matches
    more than leaving it alone helps partial ones."""
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """A minimal Okapi BM25 index over a fixed list of Chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("BM25Index needs at least one chunk")
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self._doc_tokens = [tokenize(c.text) for c in chunks]
        self._doc_len = [len(toks) for toks in self._doc_tokens]
        self._avg_doc_len = sum(self._doc_len) / len(self._doc_len)
        self._term_counts = [Counter(toks) for toks in self._doc_tokens]
        self._idf = self._compute_idf(self._doc_tokens)

    def _compute_idf(self, doc_tokens: list[list[str]]) -> dict[str, float]:
        doc_freq: dict[str, int] = {}
        for toks in doc_tokens:
            for term in set(toks):
                doc_freq[term] = doc_freq.get(term, 0) + 1
        n_docs = len(doc_tokens)
        # +1 keeps idf non-negative even for a term in every document,
        # unlike the classic (unclamped) BM25 idf formula.
        return {
            term: math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
            for term, df in doc_freq.items()
        }

    def score(self, query: str) -> list[float]:
        """BM25 score of every chunk against `query`, same order as self.chunks."""
        q_terms = tokenize(query)
        scores = [0.0] * len(self.chunks)
        for i, term_counts in enumerate(self._term_counts):
            doc_len = self._doc_len[i]
            length_norm = 1 - self.b + self.b * doc_len / max(self._avg_doc_len, 1e-8)
            total = 0.0
            for term in q_terms:
                tf = term_counts.get(term)
                if not tf:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = tf + self.k1 * length_norm
                total += idf * (tf * (self.k1 + 1)) / max(denom, 1e-8)
            scores[i] = total
        return scores

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self.score(query)
        ranked = sorted(range(len(self.chunks)), key=lambda i: -scores[i])[:k]
        return [(self.chunks[i], scores[i]) for i in ranked]
