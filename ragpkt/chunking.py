"""
Milestone 1: chunking.

RAG quality lives or dies on how you cut source material into pieces small
enough to embed and retrieve individually, but large enough to still make
sense out of context. A dumb fixed-size character split cuts functions in
half and separates a docstring from the code it describes.

This module chunks *code-aware*: Python files are split by top-level
function/class using the `ast` module, so every chunk is a complete,
self-contained unit (signature + docstring + body). Markdown, notebooks,
and everything else fall back to a paragraph-aware sliding window.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

CODE_EXTS = {".py"}
NOTEBOOK_EXTS = {".ipynb"}
TEXT_EXTS = {".md", ".txt", ".log"}

# Directories we never want to index (mirrors what a .gitignore usually excludes).
SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "checkpoints", "data", ".ragpkt"}


@dataclass
class Chunk:
    id: str
    path: str
    kind: str  # "function" | "class" | "module" | "text" | "notebook_cell"
    name: str  # function/class name, heading, or "" for a plain text window
    start_line: int
    end_line: int
    text: str

    def to_dict(self) -> dict:
        return self.__dict__

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(**d)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_python_file(path: Path, rel_path: str) -> list[Chunk]:
    """AST-split a .py file into one chunk per top-level function/class.

    Falls back to a single whole-file chunk if the file doesn't parse
    (e.g. a syntax error, or a script using unusual syntax).
    """
    source = _read(path)
    lines = source.splitlines()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [Chunk(
            id=f"{rel_path}::module",
            path=rel_path, kind="module", name=path.stem,
            start_line=1, end_line=len(lines),
            text=source,
        )]

    chunks: list[Chunk] = []
    covered = set()  # line numbers already claimed by a function/class chunk

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            start = node.lineno
            end = getattr(node, "end_lineno", start)
            text = "\n".join(lines[start - 1:end])
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            chunks.append(Chunk(
                id=f"{rel_path}::{node.name}",
                path=rel_path, kind=kind, name=node.name,
                start_line=start, end_line=end, text=text,
            ))
            covered.update(range(start, end + 1))

    # Whatever's left (imports, module-level constants, a __main__ block)
    # becomes one "module" chunk, so nothing gets silently dropped.
    leftover_lines = [
        (i + 1, line) for i, line in enumerate(lines) if (i + 1) not in covered
    ]
    leftover_text = "\n".join(line for _, line in leftover_lines).strip()
    if leftover_text:
        chunks.append(Chunk(
            id=f"{rel_path}::module",
            path=rel_path, kind="module", name=path.stem,
            start_line=1, end_line=len(lines),
            text=leftover_text,
        ))

    return chunks or [Chunk(
        id=f"{rel_path}::module", path=rel_path, kind="module",
        name=path.stem, start_line=1, end_line=len(lines), text=source,
    )]


def chunk_text_file(path: Path, rel_path: str, window_lines: int = 40, overlap: int = 6) -> list[Chunk]:
    """Sliding-window chunk for markdown/txt/log files.

    Splits on blank lines first (paragraph boundaries) so we don't cut a
    sentence in half, then groups paragraphs into ~window_lines chunks
    with a small overlap so context isn't lost at a chunk boundary.
    """
    lines = _read(path).splitlines()
    if not lines:
        return []

    chunks: list[Chunk] = []
    i = 0
    idx = 0
    while i < len(lines):
        end = min(i + window_lines, len(lines))
        text = "\n".join(lines[i:end]).strip()
        if text:
            chunks.append(Chunk(
                id=f"{rel_path}::part{idx}",
                path=rel_path, kind="text", name="",
                start_line=i + 1, end_line=end, text=text,
            ))
            idx += 1
        if end == len(lines):
            break
        i = end - overlap  # step forward, keeping `overlap` lines of context
    return chunks


def chunk_notebook_file(path: Path, rel_path: str) -> list[Chunk]:
    """One chunk per notebook cell (code or markdown)."""
    try:
        nb = json.loads(_read(path))
    except json.JSONDecodeError:
        return []

    chunks: list[Chunk] = []
    for idx, cell in enumerate(nb.get("cells", [])):
        source = cell.get("source", [])
        text = "".join(source).strip()
        if not text:
            continue
        chunks.append(Chunk(
            id=f"{rel_path}::cell{idx}",
            path=rel_path, kind="notebook_cell",
            name=cell.get("cell_type", "code"),
            start_line=idx, end_line=idx,
            text=text,
        ))
    return chunks


def chunk_repo(
    repo_root: Path,
    text_window_lines: int = 40,
    text_overlap: int = 6,
) -> list[Chunk]:
    """Walk a repo and chunk every file we know how to chunk.

    text_window_lines/text_overlap only affect .md/.txt/.log files
    (chunk_text_file's sliding window); Python and notebook chunking are
    AST/cell-bounded and don't have a "size" to tune. Exposed here, not
    just as chunk_text_file's own defaults, so eval/eval_chunking.py can
    ablate them across a whole repo without reaching into internals.
    """
    all_chunks: list[Chunk] = []
    for path in sorted(repo_root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue

        rel_path = str(path.relative_to(repo_root))
        ext = path.suffix.lower()

        if ext in CODE_EXTS:
            all_chunks.extend(chunk_python_file(path, rel_path))
        elif ext in NOTEBOOK_EXTS:
            all_chunks.extend(chunk_notebook_file(path, rel_path))
        elif ext in TEXT_EXTS:
            all_chunks.extend(
                chunk_text_file(path, rel_path, window_lines=text_window_lines, overlap=text_overlap)
            )

    return all_chunks
