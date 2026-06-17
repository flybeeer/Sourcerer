# Sourcerer

> A hybrid RAG knowledge assistant that answers from your documents — with sources.

Persistent project context for Claude Code. Read this at the start of every session.

## What this project is

**Sourcerer** is an internal-knowledge assistant that answers questions grounded in a document
corpus, built as a **portfolio project** to demonstrate production LLM-engineering skills.
The name plays on *source* + *sorcerer*: every answer is grounded in cited sources.
The defining features (the things that make it stand out) are:

1. **Hybrid retrieval** — vector + keyword (BM25) + reranker, not vector search alone.
2. **Evaluation harness** — measurable proof the system works, not "looks fine to me".
3. **Hybrid routing** — easy/sensitive/high-volume queries → local LLM; hard queries → frontier API.

Full design rationale and phase plan live in `docs/hybrid-rag-portfolio-project.md`. Read it before
making architectural decisions.

## Goal & priorities

- This is a learning + portfolio project, not a product. Optimize for **clarity and demonstrable
  engineering judgment**, not feature count.
- Every quality claim must be backed by an eval number. No "this should be better" without measuring.
- Prefer writing core logic explicitly first (to show understanding), then introduce frameworks.

## Target stack

| Layer            | Choice                                              |
|------------------|-----------------------------------------------------|
| Local inference  | Ollama (dev) → vLLM (production)                    |
| Local model      | Qwen 2.5 32B or Llama 3.x, INT4 quantized           |
| API model        | A frontier API model, for the "hard query" route    |
| Vector DB        | pgvector (start here) — Qdrant/Weaviate if scaling   |
| Keyword search   | BM25 (Postgres FTS or OpenSearch)                   |
| Reranker         | bge-reranker (local) or a rerank API                |
| Embeddings       | bge-m3 (local, keeps the hybrid theme consistent)   |
| Orchestration    | Write explicitly first; LlamaIndex if it earns its place |
| Backend          | FastAPI                                             |
| Eval             | RAGAS or a hand-written harness                     |
| Packaging        | docker-compose; one command to run the whole stack  |

## Architecture (target)

```
Documents → Ingestion (chunk + embed + index)
Query → Hybrid Retrieval (vector + BM25 → RRF fusion → reranker)
      → Router (simple/sensitive → local LLM; hard → API)
      → Generation with citations
      → Eval harness + logging/observability
```

Optional Phase 6 extension: GraphRAG as a *parallel* retrieval path for global/whole-corpus
questions, routed separately from hybrid retrieval, with eval comparing the two.

## Build order (phases)

Work strictly in order. Get each phase end-to-end working before improving quality.

- **Phase 1** — MVP: ingestion + vector-only retrieval + generation with citations, behind FastAPI.
- **Phase 2** — Hybrid retrieval: add BM25, fuse with RRF, add reranker, experiment with chunking.
- **Phase 3** — Evaluation harness (most important): retrieval metrics (recall@k, MRR, hit rate)
  and generation metrics (faithfulness, answer relevancy via LLM-as-judge). Comparison tables.
- **Phase 4** — Hybrid routing: classify queries, route local vs API, track cost/latency, compute savings.
- **Phase 5** — Production polish: migrate to vLLM, observability, guardrails (refuse when no answer
  in context), dockerize, write a strong README.
- **Phase 6 (optional)** — GraphRAG extension (see blueprint).

Update the "Current status" section below as phases complete.

## Conventions

- Language: Python. Type hints required. Format with ruff/black.
- Keep retrieval, routing, generation, and eval as cleanly separated modules.
- All LLM calls go through one thin client wrapper (so local vs API is swappable).
- Log every query: retrieval trace, which route was taken, tokens, latency, cost.
- Secrets in `.env`, never committed. Provide `.env.example`.
- Each phase merges only when its eval numbers are recorded.

## Hard rules

- **Never skip evaluation.** A change isn't "better" until the harness says so.
- When the answer isn't in the retrieved context, the system says "I don't know" — it does not guess.
- Start the local model with Ollama; only move to vLLM in Phase 5.
- Keep the corpus small while iterating (especially before any GraphRAG indexing — it's expensive).

## Current status

- [x] Phase 1 — MVP (ingestion + vector retrieval + cited generation behind FastAPI POST /query)
- [x] Phase 2 — Hybrid retrieval (BM25 FTS ∥ vector → RRF → reranker; swappable chunking; vector path kept behind RETRIEVAL_MODE flag)
- [x] Phase 3 — Evaluation harness (recall@k/MRR/hit + faithfulness/answer-relevancy via LLM judge; vector vs hybrid vs hybrid+rerank table in README)
- [x] Phase 4 — Hybrid routing (heuristic difficulty router + privacy override → local vs frontier API; per-query route/tokens/latency/cost logged to query_log; `scripts/route_report.py` reports split + % saved vs API-only; API route via official anthropic SDK, gateway-aware base_url; Thai-aware markers)
- [x] Phase 5 — Production polish (vLLM backend swappable via LOCAL_BACKEND behind the LLM wrapper; embeddings stay on Ollama; structured per-query JSON trace in observability/trace.py; guardrails: empty/low-relevance context → "I don't know" + prompt-injection screen; docker-compose runs db+ollama+api, Dockerfile installs .[api]; full README per blueprint)
- [x] Phase 6 — GraphRAG (optional): src/sourcerer/graphrag/ (extraction→graph/communities→summaries→store as JSON under GRAPHRAG_ROOT; local + global map-reduce search). Indexing uses the LOCAL model (GRAPHRAG_EXTRACTION_MODEL) and is gated behind a confirm prompt (scripts/graphrag_index.py — EXPENSIVE). Router sends overview questions → graph global via /query when GRAPHRAG_ENABLED. scripts/graphrag_eval.py compares vs hybrid (quality + cost) on eval/overview_eval_set.jsonl. networkx is the [graphrag] extra. NOTE: index has NOT been built yet (needs the expensive run).
- [x] Phase 7 — SQL knowledge base (optional): src/sourcerer/sqlkb/ treats a SQLite DB (SQL_KB_PATH, opened read-only) as a source via two paths. Path A (RAG): scripts/ingest_sql.py runs a user SELECT, 1 row = 1 doc (source label kb:<id>), reuses the chunk→embed→store pipeline (pipeline._store_document shared with ingest_directory). Path B (Text-to-SQL): analytical questions (router.is_analytical_query, EN+Thai markers) → generate read-only SELECT → safety.safe_select validates (single SELECT/WITH, no DML, enforces LIMIT) → execute → cited NL answer (SQL+rows = the citation). Wired in api/routes.py via _text_to_sql_query (auto when SQL_KB_ENABLED + analytical, or route_override="sql"). scripts/build_sql_demo.py seeds a demo sales DB; scripts/sql_eval.py reports execution accuracy on eval/sql_eval_set.jsonl. Gated behind SQL_KB_ENABLED. Storage engine is swappable via SQL_KB_BACKEND behind sqlkb/backends.py (sqlite default | duckdb columnar for analytics-scale tables — aggregations push down; introspection via information_schema; [duckdb] extra), mirroring the LOCAL_BACKEND pattern. Runtime cost guards (backends.query_guard, used by text_to_sql._execute): wall-clock SQL_KB_TIMEOUT_S aborts a runaway query on both engines (SQLite progress-handler deadline; DuckDB watchdog-thread interrupt) and SQLite-only SQL_KB_MAX_SCAN_OPS budgets VM ops as a rows/bytes-scanned proxy — a trip raises QueryCostError → "I don't know". Schema retrieval (sqlkb/schema_retrieval.py, used by text_to_sql.run): when a source has more than SQL_KB_SCHEMA_TOP_K tables, only the most relevant ones go in the prompt instead of every table's DDL — tables ranked by lexical overlap + bge-m3 embedding cosine, fused with RRF (same hybrid idea as document retrieval); ranks on cheap name+column signatures (schema.table_signatures), then builds full descriptions w/ sample rows for survivors only (describe_schema(..., tables=)). Embedding leg is best-effort (degrades to lexical); 0 or table count ≤ top_k = describe all (unchanged behaviour). scripts/sql_eval.py now reports, beside execution accuracy, the operational numbers the scale work moves: end-to-end latency (mean/median/p95), SQL-gen tokens/query, prompt-schema narrowing (mean prompt_tables/total — SQLExecution.prompt_tables exposes per-query table count), and cost-guard aborts. Demo run (qwen2.5 + 1-table demo DB): 6/6 accuracy, ~1s median latency, 1/1 tables (no narrowing on a 1-table schema), 0 guard aborts.
- [ ] Phase 8 — Orchestration (Dagster, optional, PLANNED — design only): wrap the existing scripts/ entrypoints (ingest, ingest_sql, graphrag_index, run_eval) as software-defined assets (raw_corpus→chunks→[graph_index]→eval_report) with schedules/sensors/partitions; reuse the existing idempotent ingest as re-run safety; Dagster run metadata + everything stays on the existing Postgres. Chosen over Airflow for asset lineage (fits the answer-with-sources theme). Design doc: docs/orchestration-analytics-phase-8-9.md. NOT built yet.
- [ ] Phase 9 — Analytics layer (dbt, optional, PLANNED — design only): dimensional model over query_log (the existing per-query log = raw fact table): staging → fct_queries/fct_eval_runs + dim_route/model/date → marts (cost_daily, route_split, latency_by_route, eval_trend); dbt tests for data quality; fct_queries incremental; Dagster runs `dbt build` as a downstream asset. Design doc: docs/orchestration-analytics-phase-8-9.md. NOT built yet.

(Claude Code: keep this checklist current as work progresses.)
