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

- [ ] Phase 1 — MVP
- [ ] Phase 2 — Hybrid retrieval
- [ ] Phase 3 — Evaluation harness
- [ ] Phase 4 — Hybrid routing
- [ ] Phase 5 — Production polish
- [ ] Phase 6 — GraphRAG (optional)

(Claude Code: keep this checklist current as work progresses.)
