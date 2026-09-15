"""Regression test for a real bug found by actually running ask.py against
Groq: api_key/base_url used to be read at import time, before ask.py's own
load_dotenv() call runs (it happens after the ragpkt imports), so .env's
CHAT_API_KEY/CHAT_BASE_URL were silently never picked up. Fixed by reading
them inside generate() at call time instead."""

from unittest.mock import MagicMock, patch

from ragpkt.generate import generate


def test_generate_reads_env_vars_at_call_time_not_import_time(monkeypatch):
    monkeypatch.setenv("CHAT_API_KEY", "test-groq-key")
    monkeypatch.setenv("CHAT_BASE_URL", "https://api.groq.com/openai/v1")

    with patch("ragpkt.generate.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="ok"))
        ]
        mock_openai_cls.return_value = mock_client

        generate([{"role": "user", "content": "hi"}])

        mock_openai_cls.assert_called_once_with(
            api_key="test-groq-key", base_url="https://api.groq.com/openai/v1"
        )


def test_generate_falls_back_to_openai_api_key_when_chat_api_key_unset(monkeypatch):
    monkeypatch.delenv("CHAT_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-fallback")
    monkeypatch.delenv("CHAT_BASE_URL", raising=False)

    with patch("ragpkt.generate.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="ok"))
        ]
        mock_openai_cls.return_value = mock_client

        generate([{"role": "user", "content": "hi"}])

        mock_openai_cls.assert_called_once_with(api_key="sk-openai-fallback", base_url=None)
