"""Request/response models for the API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's question.")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="How many chunks to retrieve. Defaults to TOP_K_FINAL from config.",
    )
    mode: Literal["vector", "hybrid"] | None = Field(
        default=None,
        description="Retrieval mode override. Defaults to RETRIEVAL_MODE from config.",
    )


class CitationModel(BaseModel):
    n: int
    source: str
    chunk_index: int
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationModel]
    retrieval_mode: str


class SourceInfo(BaseModel):
    source: str
    chunks: int
    ingested: str


class DeleteResponse(BaseModel):
    source: str
    deleted: int


class HistoryItem(BaseModel):
    id: int
    query: str
    answer: str
    num_citations: int
    latency_ms: int
    created_at: str
