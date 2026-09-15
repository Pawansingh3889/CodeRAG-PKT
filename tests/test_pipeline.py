"""Regression test for a real bug found by actually running ask.py:
load_index() used to require a Path and crash on the plain str that
ask.py's own argparse default passes it."""

from pathlib import Path

import numpy as np

from ragpkt.chunking import Chunk
from ragpkt.pipeline import load_index
from ragpkt.vectorstore import VectorStore


def test_load_index_accepts_a_plain_string_path(tmp_path):
    chunk = Chunk(id="a", path="f.py", kind="function", name="a", start_line=1, end_line=1, text="x")
    store = VectorStore([chunk], np.array([[1.0, 0.0]], dtype=np.float32))
    store.save(tmp_path)

    loaded = load_index(str(tmp_path))  # str, exactly what ask.py's argparse passes
    assert len(loaded) == 1


def test_load_index_still_accepts_a_path(tmp_path):
    chunk = Chunk(id="a", path="f.py", kind="function", name="a", start_line=1, end_line=1, text="x")
    store = VectorStore([chunk], np.array([[1.0, 0.0]], dtype=np.float32))
    store.save(tmp_path)

    loaded = load_index(Path(tmp_path))
    assert len(loaded) == 1
