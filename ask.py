#!/usr/bin/env python3
"""Ask a question against a previously built index.

Usage:
    python ask.py "How does the tokenizer train?"
    python ask.py "..." --technique chain_of_thought --mmr --rerank
"""
from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from ragpkt.pipeline import INDEX_DIR, answer, load_index

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--index", default=str(INDEX_DIR))
    parser.add_argument("--technique", default="zero_shot", choices=["zero_shot", "few_shot", "chain_of_thought"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--mmr", action="store_true", help="Use MMR retrieval for diversity")
    parser.add_argument("--rerank", action="store_true", help="LLM-rerank candidates before answering")
    parser.add_argument("--show-context", action="store_true", help="Print which chunks were retrieved")
    args = parser.parse_args()

    store = load_index(args.index)
    result = answer(
        args.question, store,
        technique=args.technique, k=args.k,
        use_mmr=args.mmr, use_rerank=args.rerank,
    )

    if args.show_context:
        print("Retrieved chunks:")
        print(json.dumps(result["retrieved"], indent=2))
        print()

    print(result["answer"])


if __name__ == "__main__":
    main()
