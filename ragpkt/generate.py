"""
Milestone 6: generation.

Thin wrapper around the chat completions call. Kept separate from
prompts.py so the *what to send* (prompt engineering) and the *how to
call the API* (retries, model choice, temperature) stay independently
readable and testable.
"""

from __future__ import annotations

import os
import time

from openai import OpenAI, RateLimitError

DEFAULT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")


def generate(
    messages: list[dict],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    client: OpenAI | None = None,
    max_retries: int = 3,
) -> str:
    """Call the chat API with a small exponential backoff on rate limits."""
    client = client or OpenAI()
    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return ""  # unreachable, keeps type-checkers happy
