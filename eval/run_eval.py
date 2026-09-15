#!/usr/bin/env python3
"""
Milestone 9: ablation eval.

Runs every question in questions.json through all three prompting
techniques (zero-shot / few-shot / chain-of-thought), scores each answer
by simple keyword recall against a hand-picked expectation, and writes a
side-by-side markdown report. This isn't a rigorous eval harness — it's
a small, honest, reproducible demonstration that the prompt-engineering
choices in prompts.py actually change answer quality, with the raw
answers included so a reader can judge for themselves rather than trust
a single aggregate number.

Usage (from the repo root, after `python index.py --repo <path>`):
    python eval/run_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Run as `python eval/run_eval.py` from the repo root, so the repo root
# (not eval/) needs to be on sys.path for `ragpkt` to be importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from ragpkt.pipeline import load_index, answer

load_dotenv()

TECHNIQUES = ["zero_shot", "few_shot", "chain_of_thought"]
QUESTIONS_PATH = Path(__file__).parent / "questions.json"
REPORT_PATH = Path(__file__).parent / "report.md"


def keyword_recall(text: str, keywords: list[str]) -> float:
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    return hits / len(keywords) if keywords else 0.0


def main() -> None:
    store = load_index()
    cases = json.loads(QUESTIONS_PATH.read_text())

    lines = ["# Prompt-technique ablation\n",
             f"Index: {len(store)} chunks. {len(cases)} questions x {len(TECHNIQUES)} techniques.\n"]
    totals = {t: [] for t in TECHNIQUES}

    for case in cases:
        q, keywords = case["question"], case["expect_keywords"]
        lines.append(f"## {q}\n")
        for tech in TECHNIQUES:
            result = answer(q, store, technique=tech)
            score = keyword_recall(result["answer"], keywords)
            totals[tech].append(score)
            lines.append(f"**{tech}** (keyword recall: {score:.2f})\n")
            lines.append(f"> {result['answer'].replace(chr(10), chr(10) + '> ')}\n")
        lines.append("")

    lines.append("## Summary\n")
    lines.append("| technique | avg keyword recall |")
    lines.append("|---|---|")
    for tech in TECHNIQUES:
        avg = sum(totals[tech]) / len(totals[tech]) if totals[tech] else 0.0
        lines.append(f"| {tech} | {avg:.2f} |")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
