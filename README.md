# CodeRAG-PKT — RAG + prompt engineering, built from scratch

A retrieval-augmented Q&A system over a codebase, built the same way as
[AskPKT](https://github.com/Pawansingh3889/AskPKT): every core piece hand-written
so each RAG/prompt-engineering concept maps to real, readable code — no
LangChain, no LlamaIndex, no vector DB. Generation uses the OpenAI API
(a 1.2M-parameter from-scratch model like AskPKT has no instruction-following
ability to build prompt engineering on top of); everything around it —
chunking, embeddings cache, vector search, retrieval, re-ranking, prompting —
is hand-built.

Default target: indexes and answers questions about the
[AskPKT](https://github.com/Pawansingh3889/AskPKT) repo itself.

## Setup
```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then paste your OpenAI API key into .env
```

## Usage
```
python index.py --repo ../AskPKT
python ask.py "How does the tokenizer train?"        # default retriever is hybrid, see below
python ask.py "Does attention use a causal mask?" --technique chain_of_thought --retriever mmr --rerank --show-context
python eval/run_eval.py
python eval/eval_retrieval.py            # precision@k/recall@k: baseline vs. mmr vs. hybrid
python eval/eval_chunking.py --repo ../AskPKT   # chunk-size ablation

# no OpenAI key needed for these:
pip install -r requirements-dev.txt
pytest
```

## Milestones
- [x] 1. Code-aware chunking (AST-split `.py`, per-cell `.ipynb`, sliding-window `.md`/`.txt`) — `ragpkt/chunking.py`
- [x] 2. Embeddings with a disk cache (re-indexing an unchanged repo costs nothing) — `ragpkt/embeddings.py`
- [x] 3. Vector store — plain numpy cosine similarity, no FAISS/Chroma — `ragpkt/vectorstore.py`
- [x] 4. Retrieval — top-k, plus MMR for result diversity — `ragpkt/retrieval.py`
- [x] 5. Prompt engineering — zero-shot / few-shot / chain-of-thought templates — `ragpkt/prompts.py`
- [x] 6. Generation — chat completion with retry/backoff — `ragpkt/generate.py`
- [x] 7. LLM re-ranking — a second pass that re-scores retrieved chunks for relevance — `ragpkt/rerank.py`
- [x] 8. End-to-end pipeline — `ragpkt/pipeline.py`
- [x] 9. Ablation eval — same questions run through all 3 prompting techniques, scored and reported — `eval/run_eval.py`
- [x] 10. Hybrid search — hand-written BM25 keyword search, fused with cosine vector search via Reciprocal Rank Fusion — `ragpkt/keyword_search.py`, `retrieve_hybrid` in `ragpkt/retrieval.py`. **This is the default retriever** (`ask.py --retriever hybrid`, or no flag at all), not baseline cosine top-k: baseline measurably missed real questions, including this README's own flagship example, live, on 15 Sep 2026 (see eval/retrieval_report.md and Milestone 11 below). `--retriever baseline`/`mmr` are still there for comparison.
- [x] 11. Retrieval-quality eval — precision@k/recall@k against hand-labeled ground-truth chunk ids, comparing baseline/mmr/hybrid; scores *retrieval*, not generation, so a lucky answer can't hide a bad retrieval — `eval/eval_retrieval.py`
- [x] 12. Chunk-size ablation — same precision@k/recall@k metric, swept across a (window_lines, overlap) grid for text chunking — `eval/eval_chunking.py`
- [x] 13. Unit test suite (18 tests, zero OpenAI calls) covering BM25, the precision/recall math, RRF fusion, and the chunking ablation knobs — `tests/`

## Design notes

**Chunking is code-aware, not fixed-size.** A `.py` file is split by the
`ast` module into one chunk per top-level function/class, so a chunk is
always a complete unit (signature + docstring + body) instead of an
arbitrary character window that might cut a function in half.

**No vector database.** For a repo-sized corpus (hundreds to a few
thousand chunks), brute-force cosine similarity over a numpy matrix is a
single matrix multiply — a few milliseconds per query — and every line of
the retrieval math is visible instead of hidden behind a client library.

**MMR retrieval** trades pure relevance for some diversity in the top-k,
so three near-duplicate chunks don't crowd out context that approaches
the question from a different angle.

**LLM re-ranking** is a second, more expensive relevance judgment:
embedding similarity finds chunks that are topically close to the query,
which isn't always the same as *useful for answering it*. Fetching 3x
candidates and asking the chat model to score each one narrows back down
to `k` before the final prompt is built.

**Three prompting strategies, compared head-to-head.** `eval/run_eval.py`
runs the same questions through zero-shot, few-shot, and
chain-of-thought prompts and reports keyword recall for each, with the
raw answers included in `eval/report.md` — evidence, not just a claim,
that the prompt engineering changes answer quality.

## Repo layout
```
ragpkt/
  chunking.py    # Milestone 1
  embeddings.py  # Milestone 2
  vectorstore.py # Milestone 3
  retrieval.py   # Milestone 4
  prompts.py     # Milestone 5
  generate.py    # Milestone 6
  rerank.py      # Milestone 7
  pipeline.py    # Milestone 8
index.py         # CLI: build an index over a target repo
ask.py           # CLI: ask a question against a built index
eval/
  questions.json # hand-picked test questions + expected keywords
  run_eval.py    # Milestone 9: ablation across prompting techniques
```
