#!/usr/bin/env python3
"""Build a vector index over a target repo.

Usage:
    python index.py --repo ../AskPKT
"""
from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from ragpkt.pipeline import INDEX_DIR, build_index

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Path to the repo to index")
    parser.add_argument("--out", default=str(INDEX_DIR), help="Where to save the index")
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    if not repo_root.is_dir():
        raise SystemExit(f"Not a directory: {repo_root}")

    print(f"Chunking + embedding {repo_root} ...")
    store = build_index(repo_root, out_dir=Path(args.out))
    print(f"Indexed {len(store)} chunks -> {args.out}")


if __name__ == "__main__":
    main()
