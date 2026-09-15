#!/usr/bin/env python3
"""A chat interface over the same pipeline ask.py drives.

Each message is still one independent RAG query, same as ask.py, there's
no conversation memory yet: retrieval and generation don't see earlier
turns in the thread. That's a real limitation, not an oversight, adding
it means threading history into BUILDERS[technique]'s prompt construction
in ragpkt/prompts.py, a deliberate next step, not bundled in here.

Usage:
    python index.py --repo ../AskPKT   # once, if you haven't already
    python serve.py
Then open http://localhost:8010
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ragpkt.pipeline import RETRIEVERS, answer, load_index

load_dotenv()

_WEB_DIR = Path(__file__).resolve().parent / "web"
_store = None  # loaded once at startup, not per-request


@asynccontextmanager
async def _lifespan(app: FastAPI):
    global _store
    _store = load_index()
    yield


app = FastAPI(title="CodeRAG-PKT chat", lifespan=_lifespan)


class AskRequest(BaseModel):
    question: str
    technique: str = "zero_shot"
    retriever: str = "hybrid"
    rerank: bool = False


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_WEB_DIR / "chat.html")


@app.post("/api/ask")
def api_ask(req: AskRequest) -> JSONResponse:
    if req.retriever not in RETRIEVERS:
        return JSONResponse(
            status_code=400,
            content={"error": f"retriever must be one of {RETRIEVERS}, got {req.retriever!r}"},
        )
    result = answer(
        req.question, _store,
        technique=req.technique, retriever=req.retriever, use_rerank=req.rerank,
    )
    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("serve:app", host="127.0.0.1", port=8010, reload=False)
