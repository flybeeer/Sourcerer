"""HTTP routes. The route wires retrieval -> generation; the modules stay separate."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from sourcerer import corpus
from sourcerer.api.schemas import (
    CitationModel,
    DeleteResponse,
    HistoryItem,
    QueryRequest,
    QueryResponse,
    SourceInfo,
)
from sourcerer.config import get_settings
from sourcerer.generation import generator
from sourcerer.observability import logging as query_log
from sourcerer.retrieval import vector

router = APIRouter()

_STATIC_DIR = Path(__file__).resolve().parent / "static"


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the minimal single-page web UI."""
    return FileResponse(_STATIC_DIR / "index.html")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Retrieve relevant chunks and answer the question with citations."""
    settings = get_settings()
    top_k = request.top_k or settings.top_k_final

    started = time.perf_counter()
    chunks = vector.search(request.query, top_k=top_k)
    answer = generator.generate(request.query, chunks)
    latency_ms = int((time.perf_counter() - started) * 1000)

    query_log.log_query(
        query=request.query,
        answer=answer.text,
        num_citations=len(answer.citations),
        latency_ms=latency_ms,
    )

    return QueryResponse(
        answer=answer.text,
        citations=[CitationModel(**vars(c)) for c in answer.citations],
    )


@router.get("/history", response_model=list[HistoryItem])
def history(limit: int = 20) -> list[HistoryItem]:
    """Return recent queries, newest first."""
    return [HistoryItem(**row) for row in query_log.recent_queries(limit=limit)]


@router.get("/sources", response_model=list[SourceInfo])
def list_sources() -> list[SourceInfo]:
    """List the ingested sources in the knowledge base."""
    return [SourceInfo(**row) for row in corpus.list_sources()]


@router.delete("/sources/{source}", response_model=DeleteResponse)
def delete_source(source: str) -> DeleteResponse:
    """Delete all chunks for a given source. 404 if the source isn't found."""
    deleted = corpus.delete_source(source)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No such source: {source}")
    return DeleteResponse(source=source, deleted=deleted)
