"""
Milestone 2: embeddings.

Two providers, chosen via EMBEDDING_PROVIDER in .env:
  "local"  (default) — sentence-transformers/all-MiniLM-L6-v2, runs on your
            own machine (Metal-accelerated on Apple Silicon via MPS), $0,
            no API key, no network call once the model's downloaded once.
  "openai" — OpenAI's embeddings API, opt-in, needs OPENAI_API_KEY.

Either way, the actual embedding call is a few lines — the part worth
writing by hand is the disk cache: re-embedding an unchanged repo should
cost nothing (in dollars, for "openai"; in time, for "local") on the
second run, and a partial run that gets interrupted should resume instead
of re-paying for chunks it already embedded.

Every env var here is read lazily (inside a function, at call time), not
at import time: callers (ask.py, index.py, eval/*.py) call load_dotenv()
*after* importing ragpkt.pipeline, which imports this module, so an
import-time os.environ.get() would silently see an empty environment.
This bit ragpkt/generate.py first (found by actually running ask.py
against Groq); fixed here proactively before it did the same.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

CACHE_DIR = Path(".ragpkt/embed_cache")
BATCH_SIZE = 100  # chunks per OpenAI API call; local encodes a batch in one forward pass

_local_model = None  # lazy singleton: only loaded (and only requires the
# sentence-transformers/torch dependency to be installed) if "local" is
# actually used, so an EMBEDDING_PROVIDER=openai setup doesn't need it.
_local_model_name = None  # the name it was loaded with, to detect a mid-run env change


def _provider() -> str:
    return os.environ.get("EMBEDDING_PROVIDER", "local")


def default_model() -> str:
    """The model name implied by the current EMBEDDING_PROVIDER. A function,
    not a module-level constant, for the same lazy-env-read reason as
    _provider() above; called fresh by embed_texts/embed_query whenever
    they're invoked without an explicit `model=`."""
    if _provider() == "local":
        return os.environ.get("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")


def _get_local_model(model_name: str):
    global _local_model, _local_model_name
    if _local_model is None or _local_model_name != model_name:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        _local_model = SentenceTransformer(model_name, device=device)
        _local_model_name = model_name
    return _local_model


def _cache_key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}::{text}".encode()).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _embed_batch_openai(texts: list[str], model: str, client) -> list[np.ndarray]:
    from openai import OpenAI

    client = client or OpenAI()
    resp = client.embeddings.create(model=model, input=texts)
    return [np.array(item.embedding, dtype=np.float32) for item in resp.data]


def _embed_batch_local(texts: list[str], model: str) -> list[np.ndarray]:
    st_model = _get_local_model(model)
    vectors = st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    return [np.asarray(v, dtype=np.float32) for v in vectors]


def _embed_batch(texts: list[str], model: str, client) -> list[np.ndarray]:
    if _provider() == "local":
        return _embed_batch_local(texts, model)
    return _embed_batch_openai(texts, model, client)


def embed_texts(texts: list[str], model: str | None = None, client=None) -> np.ndarray:
    """Embed a list of strings, returning an (N, D) float32 array.

    Cache-first: each text's embedding is looked up on disk before doing
    any real work (an API call for "openai", a model forward pass for
    "local"), and only cache misses are computed, batched, on a re-run.
    The cache key includes the model name, so switching EMBEDDING_PROVIDER
    (different model, different dimensionality) can't silently mix vectors
    from two different embedding spaces.
    """
    model = model or default_model()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    keys = [_cache_key(t, model) for t in texts]
    vectors: list[np.ndarray | None] = [None] * len(texts)

    misses_idx: list[int] = []
    for i, key in enumerate(keys):
        cp = _cache_path(key)
        if cp.exists():
            vectors[i] = np.array(json.loads(cp.read_text()), dtype=np.float32)
        else:
            misses_idx.append(i)

    for start in range(0, len(misses_idx), BATCH_SIZE):
        batch_idx = misses_idx[start : start + BATCH_SIZE]
        batch_texts = [texts[i] for i in batch_idx]
        batch_vecs = _embed_batch(batch_texts, model, client)
        for i, vec in zip(batch_idx, batch_vecs):
            vectors[i] = vec
            _cache_path(keys[i]).write_text(json.dumps(vec.tolist()))

    return np.stack(vectors)  # type: ignore[arg-type]


def embed_query(text: str, model: str | None = None, client=None) -> np.ndarray:
    """Embed a single query string. Not cached (queries are rarely repeated),
    same provider switch as embed_texts."""
    model = model or default_model()
    return _embed_batch([text], model, client)[0]
