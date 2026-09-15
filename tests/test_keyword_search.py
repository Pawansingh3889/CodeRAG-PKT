"""BM25 tests: pure Python, no OpenAI key, no network."""

from ragpkt.chunking import Chunk
from ragpkt.keyword_search import BM25Index, tokenize


def make_chunk(id_: str, text: str) -> Chunk:
    return Chunk(id=id_, path="f.py", kind="function", name=id_, start_line=1, end_line=1, text=text)


def test_tokenize_lowercases_and_splits_on_non_identifier_chars():
    assert tokenize("RESUME_FROM = os.environ.get()") == [
        "resume_from", "os", "environ", "get",
    ]


def test_exact_identifier_match_outranks_unrelated_chunk():
    chunks = [
        make_chunk("a", "def generate_cached(model, tok, prompt): pass"),
        make_chunk("b", "def unrelated_helper(x, y): return x + y"),
        make_chunk("c", "class SelfAttention: pass"),
    ]
    bm25 = BM25Index(chunks)
    hits = bm25.search("how does generate_cached work", k=3)
    assert hits[0][0].id == "a"


def test_search_respects_k():
    chunks = [make_chunk(str(i), f"function number {i} does something") for i in range(10)]
    bm25 = BM25Index(chunks)
    hits = bm25.search("function", k=3)
    assert len(hits) == 3


def test_query_with_no_matching_terms_returns_zero_scores():
    chunks = [make_chunk("a", "def tokenize_text(s): return s.split()")]
    bm25 = BM25Index(chunks)
    scores = bm25.score("xyzzy_nonexistent_term")
    assert scores == [0.0]


def test_empty_chunks_raises():
    import pytest

    with pytest.raises(ValueError):
        BM25Index([])
