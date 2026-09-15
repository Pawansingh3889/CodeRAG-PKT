#!/usr/bin/env python3
"""
Milestone 12: chunk-size ablation.

Tests whether the text-chunking window size (ragpkt/chunking.py's
chunk_text_file) actually affects retrieval quality, the same way
eval/run_eval.py already ablates prompting technique. Reuses the
precision@k/recall@k metric from eval/eval_retrieval.py, applied to a
small grid of (window_lines, overlap) settings.

Honest scope note, checked before writing this rather than assumed: on
the default AskPKT target, only README.md is chunked by the text-window
path (chunk_text_file); every .py file is AST-chunked by function/class,
which doesn't have a "size" to tune at all. So this ablation's effect
is real but narrow here, it's exercising the mechanism correctly, not
claiming a bigger effect than one 75-line README can produce. Point it
at a repo with more prose (docs, markdown notes) to see a larger swing.

This re-embeds the whole repo once per grid point, real OpenAI API
calls each time, not free. With the default 3-point grid that's 3 full
re-indexes of the target repo.

Usage:
    python eval/eval_chunking.py --repo ../AskPKT
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from ragpkt.chunking import chunk_repo
from ragpkt.embeddings import embed_texts
from ragpkt.metrics import average_scores, precision_recall_at_k
from ragpkt.retrieval import retrieve
from ragpkt.vectorstore import VectorStore

load_dotenv()

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
REPORT_PATH = Path(__file__).parent / "chunking_report.md"
K = 5

# (window_lines, overlap) grid. Kept small on purpose: each point is a
# full re-embed of the target repo.
GRID = [(20, 3), (40, 6), (80, 12)]


def build_temp_store(repo_root: Path, window_lines: int, overlap: int) -> VectorStore:
    chunks = chunk_repo(repo_root, text_window_lines=window_lines, text_overlap=overlap)
    with tempfile.TemporaryDirectory() as cache_dir:
        # Point the embedding cache at a scratch dir per grid point, so a
        # (window, overlap) setting's chunks don't collide in the cache
        # with a different setting's chunks that happen to share text.
        import ragpkt.embeddings as embeddings_module
        original_cache = embeddings_module.CACHE_DIR
        embeddings_module.CACHE_DIR = Path(cache_dir)
        try:
            vectors = embed_texts([c.text for c in chunks])
        finally:
            embeddings_module.CACHE_DIR = original_cache
    return VectorStore(chunks, vectors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="Path to the repo to ablate chunking on")
    args = parser.parse_args()
    repo_root = Path(args.repo).resolve()

    cases = json.loads(QUESTIONS_PATH.read_text())
    lines = [
        "# Chunk-size ablation (window_lines, overlap)\n",
        f"Repo: {repo_root}. Grid: {GRID}. k={K}.\n",
        ("Note: this repo's target is mostly .py (AST-chunked, size-invariant); "
         "only text/markdown files are affected by this grid. See script docstring.\n"),
    ]

    results = []
    for window_lines, overlap in GRID:
        print(f"Chunking + embedding with window_lines={window_lines}, overlap={overlap} ...")
        store = build_temp_store(repo_root, window_lines, overlap)
        scores = []
        for case in cases:
            hits = retrieve(case["question"], store, k=K)
            retrieved_ids = [c.id for c, _ in hits]
            scores.append(precision_recall_at_k(retrieved_ids, case["expect_chunk_ids"]))
        avg = average_scores(scores)
        results.append((window_lines, overlap, len(store), avg))

    lines.append("| window_lines | overlap | chunks | precision@k | recall@k | hit rate |")
    lines.append("|---|---|---|---|---|---|")
    for window_lines, overlap, n_chunks, avg in results:
        lines.append(
            f"| {window_lines} | {overlap} | {n_chunks} | "
            f"{avg['precision']:.2f} | {avg['recall']:.2f} | {avg['hit_rate']:.2f} |"
        )

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
