# CodeRAG-PKT

A retrieval-augmented Q&A system over a codebase, hand-built end to end (chunking,
embeddings cache, vector search, retrieval, re-ranking, prompting). No LangChain,
LlamaIndex, or vector database, that's the point of the repo, not an oversight.
Default target: indexes and answers questions about the AskPKT repo. See README.md
for the full design rationale and milestone checklist.

## Architecture (in pipeline order)

```
chunking.py    → code-aware chunks (AST-split .py, per-cell .ipynb, sliding-window .md/.txt)
embeddings.py  → embeddings with a disk cache (re-indexing an unchanged repo costs nothing)
vectorstore.py → plain numpy cosine similarity, no FAISS/Chroma
retrieval.py   → top-k retrieval, plus MMR for result diversity
prompts.py     → zero-shot / few-shot / chain-of-thought templates
generate.py    → chat completion with retry/backoff
rerank.py      → LLM re-ranking, a second relevance pass on the top candidates
pipeline.py    → orchestrates the above; the one file that knows the full flow
```

`index.py` builds the index from a `--repo` path. `ask.py` is the CLI entry point.
`serve.py` is a chat UI (`web/chat.html`) over the same `pipeline.answer()` ask.py
uses, one independent query per message, no conversation memory yet.
`eval/run_eval.py` runs the fixed question set through all prompting techniques and
scores the result; `eval/questions.json` is the fixture, hand-curated, not generated.

Default retriever is `"hybrid"`, not baseline cosine top-k: baseline measurably
missed real questions (60% vs. 100% recall in `eval/retrieval_report.md`, and live,
on this project's own flagship README example). `--retriever baseline`/`mmr` stay
available for comparison, not because either is the recommended default.

Embeddings default to local (`sentence-transformers`, no key); generation defaults
to whatever `CHAT_BASE_URL`/`CHAT_API_KEY` point at in `.env` (OpenAI if unset).
Every env var in `ragpkt/embeddings.py` and `ragpkt/generate.py` is read lazily,
inside a function, not at module import time, callers call `load_dotenv()` after
importing `ragpkt.pipeline`, so an eager read silently misses `.env` overrides.
This actually broke generation once already; don't reintroduce it.

## Dev commands

```bash
pip install -r requirements.txt
cp .env.example .env   # paste an OpenAI API key
python index.py --repo ../AskPKT
python ask.py "How does the tokenizer train?" --technique chain_of_thought --mmr --rerank
python eval/run_eval.py
```

## What NOT to do

- **Don't add LangChain, LlamaIndex, or a vector database.** The entire premise of
  this repo is that every RAG concept is visible in hand-written code. A dependency
  that hides retrieval math defeats the purpose, no matter how convenient.
- **Don't hardcode API keys.** `.env` only, `.env.example` stays placeholder-only.
- **Don't change the milestone checklist in README.md without updating it to match.**
  It's the record of what's actually been built and verified, not aspirational.
- **Don't add a metric or eval without also adding the fixture that grounds it.**
  `eval/run_eval.py` and `eval/questions.json` are a pair; a score with no fixed
  question set behind it isn't a real eval, it's a vibe.
