"""HTTP routes. The route wires retrieval -> generation; the modules stay separate."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse

from sourcerer import corpus, governance
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
from sourcerer.graphrag import index as graph_index
from sourcerer.graphrag import search as graph_search
from sourcerer.graphrag import store as graph_store
from sourcerer.llm import client as llm_client
from sourcerer.observability import logging as query_log
from sourcerer.observability import trace
from sourcerer.retrieval import retriever
from sourcerer.routing import pricing
from sourcerer.routing import router as query_router
from sourcerer.sqlkb import answer as sql_answer

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


def _graph_global_query(
    request: QueryRequest,
    settings,
    reduce_with_api: bool,
    reason_prefix: str,
    principal_obj=None,
) -> QueryResponse:
    """Answer a whole-corpus question via GraphRAG global search (Phase 6).

    Map runs on the local model; the reduce (synthesis) runs on the API model when
    `reduce_with_api` (and a key is set), else locally. Logged like any response.
    With governance on (Phase 10d), communities whose sources the principal can't
    fully read are dropped *before* map, so the reduce never sees forbidden content.
    """
    principal = principal_obj.id if principal_obj else None
    difficulty = query_router.route(request.query, settings).difficulty
    started = time.perf_counter()
    map_client = llm_client.get_llm_client(settings)  # cheap map calls stay local

    # Send only the final reduce (synthesis) to the frontier API model if requested.
    reduce_client = None
    reason = reason_prefix
    if reduce_with_api and settings.anthropic_api_key:
        reduce_client = llm_client.get_api_client(settings)
        reason += f"; reduce via API ({settings.api_model})"
    elif reduce_with_api:
        reason += "; API reduce requested but no ANTHROPIC_API_KEY, reduced locally"

    # Live-filter against the current corpus so a stale index (sources deleted
    # since indexing) doesn't answer from removed documents — and warn on drift.
    live_sources = {s["source"] for s in corpus.list_sources()}
    stale = graph_store.indexed_sources(settings) - live_sources
    if stale:
        reason += (
            f"; ⚠ graph index stale — {len(stale)} source(s) deleted since indexing "
            "(filtered out; re-index to refresh summaries)"
        )
        _log.warning("GraphRAG index stale; deleted sources excluded: %s", sorted(stale))

    # Select the communities to map over (json: largest; postgres: pgvector-ranked).
    communities = graph_store.select_for_global(request.query, settings, live_sources)
    # Phase 10d governance gate: drop communities the principal can't fully read,
    # so map/reduce only ever see authorized content (None principal/off = no-op).
    communities, denied_assets = governance.filter_readable_communities(
        communities, principal_obj, settings
    )
    if denied_assets:
        reason += f"; governance: {denied_assets} community(ies) hidden from {principal}"
    result = graph_search.global_search(
        request.query,
        communities,
        map_client,
        reduce_client=reduce_client,
        live_sources=live_sources,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)

    # Map + indexing are local ($0); only the reduce stage may be priced (API).
    ran_model = result.model or settings.local_model
    cost_usd = pricing.estimate_cost(
        ran_model, result.answer_input_tokens, result.answer_output_tokens
    )
    route_label = "api" if pricing.is_priced(ran_model) else "local"
    # Show the source files behind each community so the answer is traceable.
    citations = [
        CitationModel(
            n=i,
            source=f"{c.label} · {', '.join(c.sources)}" if c.sources else c.label,
            chunk_index=0,
            score=c.score,
            snippet=c.snippet,
        )
        for i, c in enumerate(result.citations, start=1)
    ]
    answered = bool(result.citations)
    audit_denied = denied_assets if (principal_obj and settings.governance_enabled) else None

    query_log.log_query(
        query=request.query,
        answer=result.text,
        num_citations=len(citations),
        latency_ms=latency_ms,
        route=route_label,
        model=ran_model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=cost_usd,
        router_reason=reason,
        difficulty=difficulty,
        principal=principal,
        denied_assets=audit_denied,
    )
    trace.log_query_event(
        query=request.query,
        retrieval_mode=result.path,
        chunks=[],
        route=route_label,
        model=ran_model,
        router_reason=reason,
        difficulty=difficulty,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        answered=answered,
        guardrail=None if answered else "no_relevant_context",
        principal=principal,
        denied_assets=audit_denied,
    )
    return QueryResponse(
        answer=result.text,
        citations=citations,
        retrieval_mode=result.path,
        route=route_label,
        model=ran_model,
        router_reason=reason,
        difficulty=difficulty,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=cost_usd,
        retrieval_path=result.path,
    )


def _text_to_sql_query(request: QueryRequest, settings, principal_obj=None) -> QueryResponse:
    """Answer an analytical question by generating + running SQL (Phase 7, Path B).

    The router decision still applies (privacy keeps sensitive queries local); the
    answer is phrased by the routed backend, and the SQL is generated by the API
    model only when SQL_KB_GENERATE_WITH_API is on and the query isn't sensitive.
    Logged like every other path so cost/latency stay comparable.
    """
    decision = query_router.route(request.query, settings)
    sensitive = bool(decision.signals.get("sensitive_hits"))
    principal = principal_obj.id if principal_obj else None
    reason = f"analytical question → Text-to-SQL ({settings.sql_kb_path})"
    if principal_obj is not None:
        reason += f"; governance gate as {principal}"

    # The SQL writer: API only when opted in, keyed, and not a sensitive query.
    use_api_for_sql = (
        settings.sql_kb_generate_with_api and settings.anthropic_api_key and not sensitive
    )
    if use_api_for_sql:
        sql_client = llm_client.get_api_client(settings)
        reason += f"; SQL generated via API ({settings.api_model})"
    else:
        sql_client = llm_client.get_llm_client(settings)
    # The answer phrasing follows the normal local/api routing decision.
    answer_client = llm_client.client_for(decision.route, settings)

    started = time.perf_counter()
    answer = sql_answer.answer(request.query, settings, sql_client, answer_client, principal_obj)
    latency_ms = int((time.perf_counter() - started) * 1000)

    ran_model = answer.model or decision.model
    cost_usd = pricing.estimate_cost(ran_model, answer.input_tokens, answer.output_tokens)
    route_label = "api" if pricing.is_priced(ran_model) else "local"
    answered = bool(answer.citations) and not answer.text.strip().startswith(NO_ANSWER)
    if not answered:
        reason += "; no safe SQL produced → I don't know"

    query_log.log_query(
        query=request.query,
        answer=answer.text,
        num_citations=len(answer.citations),
        latency_ms=latency_ms,
        route=route_label,
        model=ran_model,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        cost_usd=cost_usd,
        router_reason=reason,
        difficulty=decision.difficulty,
        principal=principal,
    )
    trace.log_query_event(
        query=request.query,
        retrieval_mode="text-to-sql",
        chunks=[],
        route=route_label,
        model=ran_model,
        router_reason=reason,
        difficulty=decision.difficulty,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        answered=answered,
        guardrail=None if answered else "no_safe_sql",
        principal=principal,
    )
    return QueryResponse(
        answer=answer.text,
        citations=[CitationModel(**vars(c)) for c in answer.citations],
        retrieval_mode="text-to-sql",
        retrieval_path="text-to-sql",
        route=route_label,
        model=ran_model,
        router_reason=reason,
        difficulty=decision.difficulty,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
        cost_usd=cost_usd,
    )


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest, http_request: Request) -> QueryResponse:
    """Retrieve relevant chunks and answer the question with citations."""
    settings = get_settings()

    # Phase 10a: resolve who is asking from the principal header (None when
    # governance is disabled). Recorded on the trace/query_log only for now —
    # the Cerbos gate that *enforces* it arrives in 10b+.
    principal_obj = governance.resolve_principal(
        http_request.headers.get(settings.principal_header), settings
    )
    principal = principal_obj.id if principal_obj else None

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
                principal=principal,
            )
            raise HTTPException(
                status_code=400,
                detail="Input rejected: it looks like a prompt-injection attempt.",
            )

    # Phase 7: Text-to-SQL path. Analytical/aggregation questions ("total sales
    # last year") can't be answered by retrieval — they need a SUM over many rows.
    # Taken when forced (route_override "sql") or auto-detected while enabled.
    sql_override = (request.route_override or "auto").lower()
    forced_sql = sql_override == "sql"
    auto_sql = (
        sql_override == "auto"
        and settings.sql_kb_enabled
        and query_router.is_analytical_query(request.query)
    )
    if forced_sql or auto_sql:
        if Path(settings.sql_kb_path).exists():
            return _text_to_sql_query(request, settings, principal_obj=principal_obj)
        if forced_sql:
            raise HTTPException(
                status_code=400,
                detail=f"Text-to-SQL requested but no SQLite DB at {settings.sql_kb_path} "
                "(set SQL_KB_PATH).",
            )
        _log.warning("Text-to-SQL: analytical query but no DB found; falling back to hybrid.")

    # Phase 6: GraphRAG global-search path. Taken when explicitly forced
    # (route_override graph-local / graph-api) or auto-detected for an overview
    # question while enabled. Specific questions fall through to hybrid below.
    override = (request.route_override or "auto").lower()
    forced_graph = override in ("graph-local", "graph-api")
    auto_graph = (
        override == "auto"
        and settings.graphrag_enabled
        and graph_search.is_overview_query(request.query)
    )
    if forced_graph or auto_graph:
        if graph_store.exists(settings):
            if override == "graph-api":
                reduce_with_api, prefix = True, "forced GraphRAG global search"
            elif override == "graph-local":
                reduce_with_api, prefix = False, "forced GraphRAG global search"
            else:
                reduce_with_api = settings.graphrag_reduce_with_api
                prefix = "overview/whole-corpus question → GraphRAG global search"
            return _graph_global_query(
                request, settings, reduce_with_api, prefix, principal_obj=principal_obj
            )
        if forced_graph:
            raise HTTPException(
                status_code=400,
                detail="GraphRAG path requested but no index found. "
                "Build it first: python scripts/graphrag_index.py",
            )
        _log.warning("GraphRAG: overview query but no index found; falling back to hybrid.")

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

    # Phase 10b governance gate: resolve the sources this principal may read and
    # push them into retrieval (None = governance off → no filter). An empty set
    # means nothing is visible → retrieval returns [] → guardrail says "I don't
    # know" rather than leaking. Forbidden chunks never enter the candidate set.
    allowed_sources = governance.allowed_document_sources(principal_obj, settings)
    denied_assets = None
    if allowed_sources is not None:
        visible = len(allowed_sources)
        total = len(corpus.list_sources())
        denied_assets = total - visible  # audit: how many sources the gate hid
        decision = query_router.RouteDecision(
            route=decision.route,
            model=decision.model,
            difficulty=decision.difficulty,
            reason=f"{decision.reason}; governance: {visible}/{total} source(s) visible "
            f"to {principal}",
            signals=decision.signals,
        )

    started = time.perf_counter()
    chunks = retriever.retrieve(
        request.query,
        settings,
        mode=resolved_mode,
        top_k_final=request.top_k,
        allowed_sources=allowed_sources,
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
        principal=principal,
        denied_assets=denied_assets,
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
        principal=principal,
        denied_assets=denied_assets,
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
def delete_source(source: str, background: BackgroundTasks) -> DeleteResponse:
    """Delete all chunks for a given source. 404 if the source isn't found.

    When GRAPHRAG_REINDEX_ON_DELETE is on, schedule a background graph rebuild so
    the (snapshot) graph catches up: deleted entities pruned, communities
    re-clustered, only changed ones re-summarized (no re-extraction).
    """
    deleted = corpus.delete_source(source)
    if deleted == 0:
        raise HTTPException(status_code=404, detail=f"No such source: {source}")

    settings = get_settings()
    reindex = (
        settings.graphrag_enabled
        and settings.graphrag_reindex_on_delete
        and graph_store.exists(settings)
    )
    if reindex:
        _log.info("scheduling background GraphRAG rebuild after deleting %s", source)
        background.add_task(graph_index.rebuild_index, settings)
    return DeleteResponse(source=source, deleted=deleted, reindex_scheduled=reindex)
