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
from ragpkt.keyword_search import BM25Index
from ragpkt.prompts import BUILDERS
from ragpkt.generate import generate
from ragpkt.rerank import rerank as rerank_chunks
from ragpkt.retrieval import retrieve, retrieve_hybrid, retrieve_mmr
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


def load_index(in_dir: Path | str = INDEX_DIR) -> VectorStore:
    # Accepts a plain str, not just a Path: ask.py's argparse `--index`
    # default is `str(INDEX_DIR)`, and passing that straight through used
    # to crash on `in_dir / "vectors.npy"` (str has no `/` operator).
    # Found by actually running `python ask.py`, not by reading the code.
    in_dir = Path(in_dir)
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
    use_hybrid: bool = False,
    use_rerank: bool = False,
) -> dict:
    """Run the full pipeline for one question. Returns the answer plus the
    intermediate state (retrieved chunks, prompt) so callers/eval can
    inspect what happened, not just the final string.

    use_hybrid wins if both use_hybrid and use_mmr are set: MMR diversifies
    a single ranked list, hybrid fuses two, mixing both isn't a coherent
    third strategy, so this picks one rather than silently doing something
    unspecified.
    """
    if technique not in BUILDERS:
        raise ValueError(f"Unknown technique {technique!r}, choose from {list(BUILDERS)}")

    fetch_k = k * 3 if use_rerank else k
    if use_hybrid:
        bm25 = BM25Index(store.chunks)
        candidates = retrieve_hybrid(question, store, bm25, k=fetch_k)
    elif use_mmr:
        candidates = retrieve_mmr(question, store, k=fetch_k)
    else:
        candidates = retrieve(question, store, k=fetch_k)

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
