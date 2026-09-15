"""
Milestone 5: prompt engineering.

Retrieval only gets context in front of the model — how you phrase the
ask still swings answer quality a lot. This module holds three prompting
strategies as plain, inspectable templates (no framework abstraction
hiding what's actually sent to the API), so eval/run_eval.py can run the
same question through all three and show the difference side by side.

  - zero_shot: instructions + retrieved context + question. The baseline.
  - few_shot:  zero_shot plus 2 worked examples of the exact answer format
               we want, so the model matches style/structure instead of
               guessing it.
  - chain_of_thought: asks the model to reason over the retrieved chunks
               step by step before answering, which helps most on
               "how do these two pieces interact" questions that need
               connecting two chunks rather than looking up one fact.
"""

from __future__ import annotations

from ragpkt.chunking import Chunk

SYSTEM_PROMPT = (
    "You are a code assistant answering questions about a specific Python "
    "repository. Answer ONLY using the provided context chunks — if the "
    "context doesn't contain the answer, say so explicitly instead of "
    "guessing. Cite the file(s) and function/class name(s) you drew on."
)

_FEW_SHOT_EXAMPLES = """\
Example 1
Context:
[tokenizer/bpe.py::train] def train(self, text: str, vocab_size: int): ...
Question: How is the tokenizer trained?
Answer: `BPETokenizer.train()` in `tokenizer/bpe.py` runs byte-pair encoding: \
it repeatedly merges the most frequent adjacent token pair until the \
vocabulary reaches `vocab_size`.

Example 2
Context:
[model/attention.py::forward] def forward(self, x): ... causal mask ...
Question: Does the attention implementation use a causal mask?
Answer: Yes — `Attention.forward()` in `model/attention.py` applies a \
causal mask before the softmax so each position can only attend to \
itself and earlier positions.
"""


def _format_context(chunks: list[Chunk]) -> str:
    parts = []
    for c in chunks:
        label = f"[{c.path}::{c.name}]" if c.name else f"[{c.path}]"
        parts.append(f"{label}\n{c.text}")
    return "\n\n---\n\n".join(parts)


def build_zero_shot(question: str, chunks: list[Chunk]) -> list[dict]:
    context = _format_context(chunks)
    user = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_few_shot(question: str, chunks: list[Chunk]) -> list[dict]:
    context = _format_context(chunks)
    user = (
        f"Here are two examples of the answer style I want:\n\n{_FEW_SHOT_EXAMPLES}\n"
        f"Now answer this one the same way.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_chain_of_thought(question: str, chunks: list[Chunk]) -> list[dict]:
    context = _format_context(chunks)
    user = (
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "First, in a section titled 'Reasoning:', work through which context "
        "chunks are relevant and how they connect. Then give a section titled "
        "'Answer:' with a concise final answer citing file/function names."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


BUILDERS = {
    "zero_shot": build_zero_shot,
    "few_shot": build_few_shot,
    "chain_of_thought": build_chain_of_thought,
}
