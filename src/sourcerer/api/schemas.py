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
    rerank: bool = Field(
        default=False,
        description="Apply a reranker over the hybrid candidates (ignored for vector mode).",
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
    # Phase 4 routing trace
    route: str  # "local" | "api"
    model: str
    router_reason: str
    difficulty: float
    input_tokens: int
    output_tokens: int
    cost_usd: float


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
    route: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    router_reason: str | None = None
    difficulty: float | None = None
