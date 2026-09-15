"""serve.py's HTTP layer: FastAPI's TestClient drives real request/response
handling, no live uvicorn process, no OpenAI/Groq call, no embedding call
(pipeline.answer is monkeypatched).

TestClient is used WITHOUT entering it as a context manager, on purpose:
that's what skips the lifespan startup hook (which would otherwise call
the real load_index() and require .ragpkt/index to already exist on
disk), so serve._store can be set to a lightweight stub instead and these
tests stay a true, portable unit test."""

from typing import ClassVar

from fastapi.testclient import TestClient

import serve
from ragpkt.chunking import Chunk


def make_store_stub():
    class FakeStore:
        chunks: ClassVar = [Chunk(id="a", path="f.py", kind="function", name="f", start_line=1, end_line=1, text="x")]

        def __len__(self):
            return 1

    return FakeStore()


def test_index_serves_chat_html():
    serve._store = make_store_stub()
    client = TestClient(serve.app)
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


def test_api_ask_calls_pipeline_answer(monkeypatch):
    serve._store = make_store_stub()
    captured = {}

    def fake_answer(question, store, technique="zero_shot", retriever="hybrid", use_rerank=False):
        captured.update(question=question, technique=technique, retriever=retriever, use_rerank=use_rerank)
        return {"question": question, "technique": technique, "retrieved": [], "answer": "fake"}

    monkeypatch.setattr(serve, "answer", fake_answer)

    client = TestClient(serve.app)
    res = client.post("/api/ask", json={"question": "how does X work"})
    assert res.status_code == 200
    assert res.json()["answer"] == "fake"
    assert captured["question"] == "how does X work"
    assert captured["retriever"] == "hybrid"  # the request default


def test_api_ask_rejects_unknown_retriever():
    serve._store = make_store_stub()
    client = TestClient(serve.app)
    res = client.post("/api/ask", json={"question": "q", "retriever": "nonexistent"})
    assert res.status_code == 400
    assert "nonexistent" in res.json()["error"]


def test_api_ask_passes_through_technique_and_rerank(monkeypatch):
    serve._store = make_store_stub()
    captured = {}

    def fake_answer(question, store, technique="zero_shot", retriever="hybrid", use_rerank=False):
        captured.update(technique=technique, use_rerank=use_rerank)
        return {"question": question, "technique": technique, "retrieved": [], "answer": "fake"}

    monkeypatch.setattr(serve, "answer", fake_answer)

    client = TestClient(serve.app)
    client.post("/api/ask", json={"question": "q", "technique": "chain_of_thought", "rerank": True})
    assert captured["technique"] == "chain_of_thought"
    assert captured["use_rerank"] is True
