"""
Milestone 3: vector store.

No FAISS, no Chroma, no Pinecone. For a repo-sized corpus (hundreds to a
few thousand chunks) a brute-force cosine-similarity search over a plain
numpy matrix is both simple to read and fast enough (a few milliseconds
per query) — so that's what this is: matrix multiply + argsort, saved to
disk as a .npy array plus a metadata sidecar.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ragpkt.chunking import Chunk


class VectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        assert len(chunks) == vectors.shape[0], "chunks/vectors length mismatch"
        self.chunks = chunks
        # L2-normalize once at build time so retrieval is a single dot product.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        self.vectors = vectors / norms

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[Chunk, float]]:
        """Top-k chunks by cosine similarity to the query vector."""
        q = query_vector / max(np.linalg.norm(query_vector), 1e-8)
        scores = self.vectors @ q  # (N,) — cosine similarity since both sides are unit vectors
        top_idx = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "vectors.npy", self.vectors)
        meta = [asdict(c) for c in self.chunks]
        (out_dir / "chunks.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, in_dir: Path) -> "VectorStore":
        vectors = np.load(in_dir / "vectors.npy")
        meta = json.loads((in_dir / "chunks.json").read_text())
        chunks = [Chunk.from_dict(d) for d in meta]
        store = cls.__new__(cls)
        store.chunks = chunks
        store.vectors = vectors  # already normalized when saved
        return store
