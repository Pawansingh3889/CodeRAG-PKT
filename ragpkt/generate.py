"""
Milestone 6: generation.

Thin wrapper around the chat completions call. Kept separate from
prompts.py so the *what to send* (prompt engineering) and the *how to
call the API* (retries, model choice, temperature) stay independently
readable and testable.

Provider-agnostic on purpose: any OpenAI-compatible chat completions
endpoint works, not just OpenAI itself. Set CHAT_BASE_URL + CHAT_API_KEY
in .env to point this at Groq's free tier (openai/gpt-oss-20b,
qwen/qwen3.8-27b, etc.) instead of paying for OpenAI's chat API. Leave
both unset and it behaves exactly as before: OpenAI's default endpoint,
OPENAI_API_KEY from the environment.
"""

from __future__ import annotations

import os
import time

from openai import OpenAI, RateLimitError


def generate(
    messages: list[dict],
    model: str | None = None,
    temperature: float = 0.2,
    client: OpenAI | None = None,
    max_retries: int = 3,
) -> str:
    """Call the chat API with a small exponential backoff on rate limits.

    model/api_key/base_url are all read from the environment here, at call
    time, not at import time: callers (ask.py, eval/*.py) call
    load_dotenv() *after* importing ragpkt.pipeline (which imports this
    module), so an import-time os.environ.get() here would silently see an
    empty environment and never pick up .env's CHAT_MODEL/CHAT_API_KEY/
    CHAT_BASE_URL. Found by actually running ask.py against Groq (it kept
    resolving to "gpt-4o-mini" and 404ing), not by reading the code.
    """
    if model is None:
        model = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
    if client is None:
        api_key = os.environ.get("CHAT_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("CHAT_BASE_URL")  # None = OpenAI's own endpoint
        client = OpenAI(api_key=api_key, base_url=base_url)
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
