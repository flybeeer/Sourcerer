"""HTTP routes. The route wires retrieval -> generation; the modules stay separate."""

from __future__ import annotations

import logging
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
from sourcerer.generation import generator, guardrails
from sourcerer.generation.prompts import NO_ANSWER
from sourcerer.llm import client as llm_client
from sourcerer.observability import logging as query_log
from sourcerer.observability import trace
from sourcerer.retrieval import retriever
from sourcerer.routing import pricing
from sourcerer.routing import router as query_router

router = APIRouter()

_log = logging.getLogger(__name__)

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

    # Guardrail 1: screen the input for obvious prompt injection before any work.
    if settings.enable_injection_check:
        reason = guardrails.injection_reason(request.query)
        if reason:
            _log.warning("rejected query (prompt injection): %s", reason)
            trace.log_query_event(
                query=request.query,
                retrieval_mode="-",
                chunks=[],
                route="-",
                model="-",
                router_reason="-",
                difficulty=0.0,
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                latency_ms=0,
                answered=False,
                guardrail="prompt_injection",
            )
            raise HTTPException(
                status_code=400,
                detail="Input rejected: it looks like a prompt-injection attempt.",
            )

    resolved_mode = (request.mode or settings.retrieval_mode).lower()

    # Optional per-request rerank: only meaningful for hybrid. Use the configured
    # reranker, or fall back to the offline LLM reranker when none is set.
    mode_label = resolved_mode
    if request.rerank and resolved_mode == "hybrid":
        rtype = settings.reranker_type if settings.reranker_type != "none" else "llm"
        settings = settings.model_copy(update={"reranker_type": rtype})
        mode_label = f"hybrid+rerank ({rtype})"

    # Route the query (local vs. frontier API) before doing any work, so the
    # rationale is logged even when the empty-context guardrail fires. The
    # heuristic always runs (cheap, and gives a difficulty to display); a manual
    # override from the request then wins, noting what auto would have chosen.
    decision = query_router.route(request.query, settings)
    override = (request.route_override or "auto").lower()
    if override in ("local", "api") and override != decision.route:
        forced_model = settings.api_model if override == "api" else settings.local_model
        decision = query_router.RouteDecision(
            route=override,
            model=forced_model,
            difficulty=decision.difficulty,
            reason=f"forced to {override} by request (auto would pick {decision.route})",
            signals=decision.signals,
        )

    if decision.route == "api" and not settings.anthropic_api_key:
        decision = decision.as_local_fallback(
            settings.local_model, "no ANTHROPIC_API_KEY set, fell back to local"
        )
    _log.info("route=%s model=%s — %s", decision.route, decision.model, decision.reason)

    started = time.perf_counter()
    chunks = retriever.retrieve(
        request.query, settings, mode=resolved_mode, top_k_final=request.top_k
    )
    # Guardrail 2: drop weakly-relevant chunks; empty context → generator says
    # "I don't know" instead of guessing.
    chunks = guardrails.filter_relevant(chunks, settings)
    gen_client = llm_client.client_for(decision.route, settings)
    answer = generator.generate(request.query, chunks, client=gen_client)
    latency_ms = int((time.perf_counter() - started) * 1000)

    # Cost is keyed to the model that actually ran (local = $0; see routing.pricing).
    ran_model = answer.model or decision.model
    cost_usd = pricing.estimate_cost(ran_model, answer.input_tokens, answer.output_tokens)

    answered = bool(chunks) and not answer.text.strip().startswith(NO_ANSWER)

    query_log.log_query(
        query=request.query,
        answer=answer.text,
        num_citations=len(answer.citations),
        latency_ms=latency_ms,
        route=decision.route,
        model=ran_model,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        cost_usd=cost_usd,
        router_reason=decision.reason,
        difficulty=decision.difficulty,
    )
    # Structured trace (machine-readable companion to the query_log row).
    trace.log_query_event(
        query=request.query,
        retrieval_mode=mode_label,
        chunks=chunks,
        route=decision.route,
        model=ran_model,
        router_reason=decision.reason,
        difficulty=decision.difficulty,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        answered=answered,
        guardrail=None if answered else "no_relevant_context",
    )

    return QueryResponse(
        answer=answer.text,
        citations=[CitationModel(**vars(c)) for c in answer.citations],
        retrieval_mode=mode_label,
        route=decision.route,
        model=ran_model,
        router_reason=decision.reason,
        difficulty=decision.difficulty,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        cost_usd=cost_usd,
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
