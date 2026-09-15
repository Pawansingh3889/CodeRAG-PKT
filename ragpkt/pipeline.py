"""
Milestone 8: end-to-end pipeline.

Wires chunking -> embeddings -> vector store -> retrieval -> (optional
re-rank) -> prompt -> generation into one call, so index.py and ask.py
stay thin CLIs and eval/run_eval.py can drive the whole thing
programmatically for the prompt-technique ablation.
"""

from __future__ import annotations

from pathlib import Path

from ragpkt.chunking import Chunk, chunk_repo
from ragpkt.embeddings import embed_texts
from ragpkt.prompts import BUILDERS
from ragpkt.generate import generate
from ragpkt.rerank import rerank as rerank_chunks
from ragpkt.retrieval import retrieve, retrieve_mmr
from ragpkt.vectorstore import VectorStore

INDEX_DIR = Path(".ragpkt/index")


def build_index(repo_root: Path, out_dir: Path = INDEX_DIR) -> VectorStore:
    chunks: list[Chunk] = chunk_repo(repo_root)
    if not chunks:
        raise ValueError(f"No chunkable files found under {repo_root}")
    vectors = embed_texts([c.text for c in chunks])
    store = VectorStore(chunks, vectors)
    store.save(out_dir)
    return store


def load_index(in_dir: Path = INDEX_DIR) -> VectorStore:
    if not (in_dir / "vectors.npy").exists():
        raise FileNotFoundError(
            f"No index at {in_dir}. Run `python index.py --repo <path>` first."
        )
    return VectorStore.load(in_dir)


def answer(
    question: str,
    store: VectorStore,
    technique: str = "zero_shot",
    k: int = 5,
    use_mmr: bool = False,
    use_rerank: bool = False,
) -> dict:
    """Run the full pipeline for one question. Returns the answer plus the
    intermediate state (retrieved chunks, prompt) so callers/eval can
    inspect what happened, not just the final string."""
    if technique not in BUILDERS:
        raise ValueError(f"Unknown technique {technique!r}, choose from {list(BUILDERS)}")

    fetch_k = k * 3 if use_rerank else k
    candidates = retrieve_mmr(question, store, k=fetch_k) if use_mmr else retrieve(question, store, k=fetch_k)

    if use_rerank:
        candidates = rerank_chunks(question, candidates, keep=k)

    chunks = [c for c, _ in candidates]
    messages = BUILDERS[technique](question, chunks)
    reply = generate(messages)

    return {
        "question": question,
        "technique": technique,
        "retrieved": [(c.path, c.name, round(score, 3)) for c, score in candidates],
        "answer": reply,
    }
