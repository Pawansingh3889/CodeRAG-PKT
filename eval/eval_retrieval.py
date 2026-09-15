#!/usr/bin/env python3
"""
Milestone 11: retrieval quality eval (precision@k / recall@k).

eval/run_eval.py scores the *generated answer* by keyword recall. This
script scores *retrieval itself*: for each question in questions.json,
does the retriever actually surface the chunk that should answer it,
before an LLM ever sees it? A generation eval can look fine by accident
(the model answers correctly from general knowledge, or a wrong chunk
still contains the right keywords); this one can't, it only checks
whether the right chunk made it into context.

Compares three retrievers on the same fixed question set: baseline
top-k cosine, MMR, and hybrid (cosine + BM25 keyword, fused with RRF).

Usage (after `python index.py --repo <path>`, needs a real OpenAI key
in .env, embeddings aren't free):
    python eval/eval_retrieval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from ragpkt.keyword_search import BM25Index
from ragpkt.metrics import average_scores, precision_recall_at_k
from ragpkt.pipeline import load_index
from ragpkt.retrieval import retrieve, retrieve_hybrid, retrieve_mmr

load_dotenv()

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
REPORT_PATH = Path(__file__).parent / "retrieval_report.md"
K = 5


def main() -> None:
    store = load_index()
    bm25 = BM25Index(store.chunks)
    cases = json.loads(QUESTIONS_PATH.read_text())

    retrievers = {
        "baseline (cosine top-k)": lambda q: retrieve(q, store, k=K),
        "mmr": lambda q: retrieve_mmr(q, store, k=K),
        "hybrid (cosine + BM25, RRF)": lambda q: retrieve_hybrid(q, store, bm25, k=K),
    }

    lines = [
        "# Retrieval eval: precision@k / recall@k\n",
        f"Index: {len(store)} chunks. {len(cases)} questions, k={K}.\n",
    ]
    all_scores: dict[str, list] = {name: [] for name in retrievers}

    for case in cases:
        q = case["question"]
        expected = case["expect_chunk_ids"]
        lines.append(f"## {q}\n")
        lines.append(f"expected: `{', '.join(expected)}`\n")
        for name, retriever in retrievers.items():
            hits = retriever(q)
            retrieved_ids = [c.id for c, _ in hits]
            score = precision_recall_at_k(retrieved_ids, expected)
            all_scores[name].append(score)
            marker = "hit" if score.hit else "miss"
            lines.append(
                f"- **{name}** ({marker}): precision={score.precision:.2f} "
                f"recall={score.recall:.2f} — retrieved `{', '.join(retrieved_ids)}`"
            )
        lines.append("")

    lines.append("## Summary\n")
    lines.append("| retriever | precision@k | recall@k | hit rate |")
    lines.append("|---|---|---|---|")
    for name, scores in all_scores.items():
        avg = average_scores(scores)
        lines.append(f"| {name} | {avg['precision']:.2f} | {avg['recall']:.2f} | {avg['hit_rate']:.2f} |")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
