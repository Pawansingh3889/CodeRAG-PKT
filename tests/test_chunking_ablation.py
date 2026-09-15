"""chunk_repo's text_window_lines/text_overlap parameters: pure text
processing, no OpenAI key needed. Verifies the ablation knobs actually
change chunk boundaries (the thing eval/eval_chunking.py depends on),
and that Python/notebook chunking is unaffected by them (the thing the
script's docstring claims about scope)."""

from pathlib import Path

from ragpkt.chunking import chunk_repo


def make_repo(tmp_path: Path, n_lines: int = 100) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "notes.md").write_text("\n\n".join(f"paragraph {i} of the notes file" for i in range(n_lines)))
    (repo / "code.py").write_text("def hello():\n    return 'hi'\n")
    return repo


def test_smaller_window_produces_more_text_chunks(tmp_path):
    repo = make_repo(tmp_path)
    small = chunk_repo(repo, text_window_lines=10, text_overlap=1)
    large = chunk_repo(repo, text_window_lines=80, text_overlap=1)

    small_text_chunks = [c for c in small if c.kind == "text"]
    large_text_chunks = [c for c in large if c.kind == "text"]
    assert len(small_text_chunks) > len(large_text_chunks)


def test_python_chunk_count_is_invariant_to_text_window_settings(tmp_path):
    repo = make_repo(tmp_path)
    small = chunk_repo(repo, text_window_lines=10, text_overlap=1)
    large = chunk_repo(repo, text_window_lines=80, text_overlap=1)

    small_py = [c for c in small if c.path.endswith(".py")]
    large_py = [c for c in large if c.path.endswith(".py")]
    assert small_py == large_py


def test_default_arguments_match_previous_behavior(tmp_path):
    # chunk_repo(repo) with no window/overlap args must chunk identically
    # to before this change, since callers (index.py) don't pass them.
    repo = make_repo(tmp_path)
    default = chunk_repo(repo)
    explicit_default = chunk_repo(repo, text_window_lines=40, text_overlap=6)
    assert default == explicit_default
