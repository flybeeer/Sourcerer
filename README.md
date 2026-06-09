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

Measured with the Phase 3 harness on a 10-question eval set over the synthetic
`eval/corpus` (12 docs). Generator `qwen2.5:3b`, LLM judge `qwen2.5:7b`,
embeddings `bge-m3`. Reproduce with `python scripts/run_eval.py`.

| Config | Recall@5 | MRR | Hit rate | Faithfulness | Answer rel. | Latency ms |
|--------|---------:|----:|---------:|-------------:|------------:|-----------:|
| vector-only    | 1.000 | 0.950 | 1.000 | 0.900 | 0.880 | 159 |
| hybrid         | 1.000 | 0.950 | 1.000 | **1.000** | 0.930 | 205 |
| hybrid+rerank  | 1.000 | **1.000** | 1.000 | 1.000 | **0.950** | 2482 |

**Reading the numbers — which config wins, and why:**

- **Recall@5 and hit rate saturate at 1.0.** With a small corpus the one relevant
  doc per question always lands in the top 5 — so retrieval *recall* is not a
  useful discriminator here. This is itself a lesson: recall alone can hide real
  quality differences.
- **MRR is the retrieval differentiator.** `hybrid+rerank` reaches a perfect
  **1.000** (relevant doc always ranked #1) vs **0.950** for the others — the
  reranker fixed the one case where the right doc sat at rank 2.
- **Generation metrics show the real story.** Hybrid lifts faithfulness
  **0.90 → 1.00** and answer relevancy **0.88 → 0.93**; adding the reranker nudges
  relevancy to **0.95**. Better-ordered, less-noisy context lets the small
  generator ground its answers more reliably — even when retrieval recall is identical.
- **Latency is the trade-off.** The reranker here is an *LLM listwise* reranker
  (one extra local-LLM call), costing ~12× latency (205 → 2482 ms). Whether a
  cross-encoder is cheaper depends on hardware — measured both below.

**Verdict:** **hybrid** is the clear win over vector-only — materially better
generation at roughly the same latency. **hybrid+rerank** gives the best ranking,
but the gain and its cost depend heavily on *which* reranker (see below). (A
noisier, multilingual corpus widens the hybrid-vs-vector gap further — vector-only's
faithfulness collapsed to ~0.20 when an unrelated Thai-language PDF polluted its
top-k, while hybrid's keyword signal suppressed it.)

### Reranker backends: bge cross-encoder vs LLM listwise

Four reranker backends over the same eval. **Retrieval metrics (recall/MRR/hit) are
deterministic and trustworthy; generation metrics are not** — see the caveat below —
so the ranking story is told by **MRR**, and the cost story by **warm rerank latency**
(measured separately; the eval's per-query latency conflates one-time model load).

| Config | Recall@5 | MRR | Hit rate | Warm rerank latency |
|--------|---------:|----:|---------:|--------------------:|
| hybrid (no rerank)                              | 1.000 | 0.950 | 1.000 | — |
| hybrid + rerank, LLM listwise (`qwen2.5:3b`)    | 1.000 | 0.925–0.950 | 1.000 | ~2.5 s |
| hybrid + rerank, cross-encoder `bge-reranker-v2-m3` (568M) | 1.000 | **1.000** | 1.000 | 7.9 s – 36 s+ (CPU) |
| hybrid + rerank, cross-encoder `ms-marco-MiniLM-L6-v2` (22M) | 1.000 | **1.000** | 1.000 | **13 ms** (CPU) |

Measured findings:

- **A weak LLM reranker can *hurt*.** The `qwen2.5:3b` listwise reranker scored MRR
  **0.925–0.950** across runs — sometimes *below* no-rerank, demoting a doc fusion had
  already ranked well. "Add a reranker" is not automatically a win; it must be better
  than your first stage.
- **A small cross-encoder is the sweet spot.** The 22M `ms-marco-MiniLM-L6-v2` matches
  the 568M `bge-reranker-v2-m3` on ranking here (**MRR 1.000**) at **~13 ms/query** on
  CPU — vs 7.9 s–36 s+ for bge. bge may pull ahead on a larger/harder corpus, but needs
  a GPU to be practical (and note: Apple **MPS deadlocks** the big model on this torch
  build — the local reranker is pinned to CPU/CUDA).
- This is the **web UI's** "Hybrid + rerank (MiniLM)" option: rerank adds ~13 ms, so
  end-to-end query time is unchanged — the ~15 s a query takes is entirely `qwen2.5:3b`
  *generation*, identical across all modes.

> ⚠️ **Generation metrics are high-variance at this scale.** Re-running the *same*
> judged eval moved vector-only's faithfulness from 0.90 to 0.26, and MiniLM landed at
> 0.40 faithfulness despite the *best* retrieval — an obvious contradiction. With a
> 10-item set and a local `qwen2.5:7b` judge, one question swings a metric by 10% and
> judge noise dominates. **Lesson:** trust the deterministic retrieval metrics; treat
> the LLM-judged generation numbers as directional only until the eval set grows to the
> 50–100 items the blueprint calls for. (The harness is the deliverable; the small
> starter set is not yet a stable generation benchmark.)

**Takeaway:** for this setup, **hybrid + a small cross-encoder (MiniLM)** is the best
balance — bge-level ranking (MRR 1.000) at negligible latency. Plain **hybrid** remains
the safe default; the big `bge` model is for when there's a GPU.

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

- [x] Phase 1 — MVP: ingestion + vector retrieval + cited generation behind FastAPI
- [x] Phase 2 — Hybrid retrieval: BM25 + RRF + reranker + swappable chunking
- [x] Phase 3 — Evaluation harness ⭐ (see results above)
- [ ] Phase 4 — Hybrid routing
- [ ] Phase 5 — Production polish (vLLM, observability, guardrails, docker)
- [ ] Phase 6 — GraphRAG (optional)
