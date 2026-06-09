"""HTTP routes. The route wires retrieval -> generation; the modules stay separate."""

from __future__ import annotations

from fastapi import APIRouter

from sourcerer.api.schemas import CitationModel, QueryRequest, QueryResponse
from sourcerer.config import get_settings
from sourcerer.generation import generator
from sourcerer.retrieval import vector

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """Retrieve relevant chunks and answer the question with citations."""
    settings = get_settings()
    top_k = request.top_k or settings.top_k_final

    chunks = vector.search(request.query, top_k=top_k)
    answer = generator.generate(request.query, chunks)

    return QueryResponse(
        answer=answer.text,
        citations=[CitationModel(**vars(c)) for c in answer.citations],
    )
