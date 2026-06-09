# Sourcerer

> A hybrid RAG knowledge assistant that answers from your documents — **with sources**.
> The name plays on *source* + *sorcerer*: every answer is grounded in cited sources.

<!-- TODO: badges — build, license, python version -->

## Demo

<!-- TODO: a short GIF/video showing a question → cited answer in the first 10 seconds -->

## The Problem

<!-- TODO: "Answers questions from internal docs X that plain ChatGPT can't." Name the corpus. -->

## What Makes It Stand Out

1. **Hybrid retrieval** — vector (pgvector) + keyword (BM25) → RRF fusion → reranker, not vector search alone.
2. **Evaluation harness** — measurable proof the system works (recall@k, MRR, faithfulness), not "looks fine to me".
3. **Hybrid routing** — easy/sensitive/high-volume queries → local LLM; hard queries → frontier API.

## Architecture

<!-- TODO: architecture diagram -->

```
Documents → Ingestion (chunk + embed + index)
Query → Hybrid Retrieval (vector + BM25 → RRF fusion → reranker)
      → Router (simple/sensitive → local LLM; hard → API)
      → Generation with citations
      → Eval harness + logging/observability
```

### Technology choices (and why)

| Layer            | Choice                          | Why |
|------------------|---------------------------------|-----|
| Local inference  | Ollama (dev) → vLLM (prod)      | <!-- TODO --> |
| Local model      | Qwen 2.5 32B / Llama 3.x (INT4) | <!-- TODO --> |
| API model        | Frontier API (hard-query route) | <!-- TODO --> |
| Vector DB        | pgvector                        | <!-- TODO --> |
| Keyword search   | BM25 (Postgres FTS)             | <!-- TODO --> |
| Reranker         | bge-reranker / Cohere           | <!-- TODO --> |
| Embeddings       | bge-m3                          | <!-- TODO --> |
| Backend          | FastAPI                         | <!-- TODO --> |

## Evaluation Results ⭐

<!-- TODO (Phase 3): comparison table — vector-only vs hybrid vs hybrid+rerank.
     retrieval metrics (recall@k, MRR, hit rate) + generation metrics (faithfulness, answer relevancy). -->

## Hybrid Routing

<!-- TODO (Phase 4): % of queries routed local, cost/latency per route, % saved vs API-for-everything. -->

## Quickstart (Phase 1 — vector RAG MVP)

```bash
# 1. Config
cp .env.example .env                      # .env is gitignored; adjust if needed

# 2. Start Postgres (pgvector) + Ollama
docker-compose up -d db ollama

# 3. Pull the local models into the Ollama container (first run only)
docker compose exec ollama ollama pull bge-m3        # embeddings
docker compose exec ollama ollama pull qwen2.5:32b   # generation (or set LOCAL_MODEL smaller)

# 4. Install the package (host) and ingest your corpus
python -m venv .venv && source .venv/bin/activate
pip install -e .
cp your-docs/*.pdf your-docs/*.md data/raw/         # PDF + Markdown supported
python scripts/ingest.py                            # load → chunk → embed → store

# 5. Serve the API and ask a question
uvicorn sourcerer.api.main:app --reload
curl -s localhost:8000/query \
  -H 'content-type: application/json' \
  -d '{"query": "What is X?"}' | jq
```

The response contains an `answer` and a `citations` list (each with `source`,
`chunk_index`, `score`, and a snippet). If the answer isn't in the corpus, the
system returns *"I don't know based on the provided documents."* rather than guessing.

> Tip: `qwen2.5:32b` is large. For a quick local test set `LOCAL_MODEL=qwen2.5:3b`
> (or any small model) in `.env`. Alternatively run the whole stack — including the
> API container — with `docker-compose up`.

## Project Structure

```
src/sourcerer/   ingestion · retrieval · routing · generation · eval · api · llm (one client wrapper)
eval/            eval set + generated results
data/            document corpus (gitignored, keep small)
scripts/         CLI entrypoints (ingest, run_eval)
```

## Trade-offs & Lessons

<!-- TODO: what didn't work and how you fixed it (chunking, context size, "no answer" handling). -->

## Roadmap / Phase Status

- [ ] Phase 1 — MVP: ingestion + vector retrieval + cited generation behind FastAPI
- [ ] Phase 2 — Hybrid retrieval: BM25 + RRF + reranker + chunking experiments
- [ ] Phase 3 — Evaluation harness ⭐
- [ ] Phase 4 — Hybrid routing
- [ ] Phase 5 — Production polish (vLLM, observability, guardrails, docker)
- [ ] Phase 6 — GraphRAG (optional)
