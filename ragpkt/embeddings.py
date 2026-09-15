"""
Milestone 2: embeddings.

Turns chunk text into vectors via the OpenAI embeddings API. The API call
itself is one line — the part worth writing by hand is the disk cache:
re-embedding an unchanged repo should cost nothing on the second run, and
a partial run that gets interrupted (rate limit, network blip) should
resume instead of re-paying for chunks it already embedded.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from openai import OpenAI

DEFAULT_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CACHE_DIR = Path(".ragpkt/embed_cache")
BATCH_SIZE = 100  # chunks per API call


def _cache_key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}::{text}".encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def embed_texts(texts: list[str], model: str = DEFAULT_MODEL, client: OpenAI | None = None) -> np.ndarray:
    """Embed a list of strings, returning an (N, D) float32 array.

    Cache-first: each text's embedding is looked up on disk before hitting
    the API, and only the cache misses are sent, batched, to keep the
    request count (and cost) down on a re-run.
    """
    client = client or OpenAI()
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
        batch_idx = misses_idx[start:start + BATCH_SIZE]
        batch_texts = [texts[i] for i in batch_idx]
        resp = client.embeddings.create(model=model, input=batch_texts)
        for i, item in zip(batch_idx, resp.data):
            vec = np.array(item.embedding, dtype=np.float32)
            vectors[i] = vec
            _cache_path(keys[i]).write_text(json.dumps(item.embedding))

    return np.stack(vectors)  # type: ignore[arg-type]


def embed_query(text: str, model: str = DEFAULT_MODEL, client: OpenAI | None = None) -> np.ndarray:
    """Embed a single query string (no caching — queries are rarely repeated)."""
    client = client or OpenAI()
    resp = client.embeddings.create(model=model, input=[text])
    return np.array(resp.data[0].embedding, dtype=np.float32)
