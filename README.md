# Sourcerer

> A hybrid RAG knowledge assistant that answers from your documents — **with sources**.
> The name plays on *source* + *sorcerer*: every answer is grounded in cited sources.

<!-- TODO: badges — build, license, python version -->

## Demo

Ask a question in the web UI (`http://localhost:8000`) → get a grounded answer
with numbered citations, plus a **routing-trace card** showing which model
answered, why, and what it cost:

```
┌──────────────────────────────────────────────────────────────┐
│  Q: Compare incident severity levels and explain why a Sev-1  │
│     is handled differently than a Sev-3.                      │
├──────────────────────────────────────────────────────────────┤
│  [API] claude-sonnet-4-6                                      │
│  hard query (difficulty 0.79 ≥ 0.70; drivers: compare,       │
│  explain, why) → API route                                   │
│  difficulty 0.79 · tokens 18413/497 · est. cost $0.0627      │
├──────────────────────────────────────────────────────────────┤
│  Sev-1 is a full outage / data-loss risk [1], so it is       │
│  escalated immediately to the primary on-call [3]; Sev-3 …    │
│  Sources: [1] incident-severity.md  [3] on-call.md           │
└──────────────────────────────────────────────────────────────┘
```

<!-- TODO: replace the mockup with a real GIF once recorded. -->

## The Problem

Plain ChatGPT can't answer questions about *your* internal documents — and when
it tries, it guesses. Sourcerer answers strictly from an indexed corpus (the demo
ships a synthetic company handbook in `eval/corpus/`: PTO, on-call, incident
severity, SLAs, security policy…) and **cites every claim** or says *"I don't
know"*. Swap in your own PDFs/Markdown via `scripts/ingest.py`.

## What Makes It Stand Out

1. **Hybrid retrieval** — vector (pgvector) + keyword (BM25) → RRF fusion → reranker, not vector search alone.
2. **Evaluation harness** — measurable proof the system works (recall@k, MRR, faithfulness), not "looks fine to me".
3. **Hybrid routing** — easy/sensitive/high-volume queries → local LLM; hard queries → frontier API, with measured cost savings.

## Architecture

```
                 ┌──────────────────────────────────────────────┐
  Documents ───▶ │  Ingestion:  load → chunk → embed (bge-m3)    │
 (PDF / MD)      │              → store in pgvector + FTS index  │
                 └───────────────────────┬──────────────────────┘
                                         ▼
                 ┌──────────────────────────────────────────────┐
  Query ───────▶ │  Guardrail: prompt-injection screen          │
                 ├──────────────────────────────────────────────┤
                 │  Hybrid Retrieval                            │
                 │    vector (pgvector)  ∥  BM25 (Postgres FTS) │
                 │        └──── RRF fusion ────┘ → reranker      │
                 ├──────────────────────────────────────────────┤
                 │  Guardrail: relevant context? else "I don't  │
                 │             know" (no guessing)              │
                 ├──────────────────────────────────────────────┤
                 │  Router:  easy/sensitive → local LLM         │
                 │           hard/complex   → frontier API      │
                 ├──────────────────────────────────────────────┤
                 │  Generation with inline [n] citations        │
                 └───────────────────────┬──────────────────────┘
                                         ▼
                 ┌──────────────────────────────────────────────┐
                 │  Observability: structured per-query trace   │
                 │  (retrieval · route · tokens · latency · $)  │
                 │  + query_log table  ·  Eval harness          │
                 └──────────────────────────────────────────────┘

  Inference behind one LLM client wrapper:  Ollama (dev) ⇄ vLLM (prod) · frontier API
```

### Technology choices (and why)

| Layer            | Choice                          | Why |
|------------------|---------------------------------|-----|
| Local inference  | Ollama (dev) → vLLM (prod)      | Ollama is one-command to run locally; vLLM's continuous batching wins on throughput in prod. One wrapper, swap via `LOCAL_BACKEND`. |
| Local model      | Qwen 2.5 (INT4)                 | Strong open weights at a "local-grade" size; INT4 fits commodity hardware. Dev uses `qwen2.5:3b` for speed. |
| API model        | Anthropic Sonnet (gateway-aware)| Frontier quality for the hard-query route; reached via the official SDK + optional `ANTHROPIC_BASE_URL` so a corp gateway works. |
| Vector DB        | pgvector                        | Vectors + BM25 (FTS) + the query log in **one** Postgres — no extra infra to operate. |
| Keyword search   | BM25 (Postgres FTS)             | The keyword half of hybrid; a generated `tsvector` column stays in sync with content. Catches exact terms vectors miss. |
| Reranker         | cross-encoder / LLM listwise    | Re-scores fused candidates before generation. Small MiniLM cross-encoder is ~13 ms/query and matches the 568M bge here (see eval). |
| Embeddings       | bge-m3                          | Strong multilingual local embeddings; keeps the whole retrieval stack self-hosted and on-theme. |
| Backend          | FastAPI                         | Async Python standard for serving; auto OpenAPI docs at `/docs`. |

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

A per-query **router** decides where each question is answered:

- **local model** (Ollama) — easy / high-volume lookups *and* anything **sensitive**.
- **frontier API model** (Anthropic Messages API, direct or via a gateway) — **hard,
  complex-reasoning** queries that justify the cost.

The default `ROUTER_STRATEGY=heuristic` is a transparent difficulty scorer (complex-reasoning
markers, query length, multi-part structure) gated by `ROUTER_HARD_THRESHOLD`. A **privacy
override** keeps any sensitive query (salary, PII, NDA, credentials …) on the local model
*regardless* of difficulty — the kind of trade-off a real org cares about. Every decision logs
its rationale and is returned in the API response (`route`, `model`, `router_reason`,
`difficulty`, tokens, `cost_usd`), so routing is fully inspectable.

Each query is instrumented (route, tokens, latency, estimated cost) into `query_log`.
`scripts/route_report.py` aggregates that — or runs offline over a built-in query set
(`--simulate`) for a reproducible number:

```
$ python scripts/route_report.py --simulate     # also: make route-report

  [local] What is the company's PTO policy?                  → easy/high-volume (0.00 < 0.70)
  [local] What is the employee salary band …?                → sensitive 'salary' (privacy override)
  [  api] Compare the trade-offs between on-prem and cloud …  → hard (0.79 ≥ 0.70; compare, trade-off, why)
  …
```

| Metric                                   | Value                          |
|------------------------------------------|--------------------------------|
| Queries                                  | 12 (representative mix)        |
| Routed **local** / **API**               | **9 (75%)** / 3 (25%)          |
| Cost per query (actual)                  | **$0.00129**                   |
| Cost per query (API-only baseline)       | $0.00514                       |
| **Saved vs API-for-everything**          | **≈ 75%**                      |

> The `--simulate` cost is a *projection*: routing is real (deterministic heuristics), but
> token counts use fixed per-query assumptions (see the script) since no generation runs. The
> savings track the industry-cited 40–70% range. Live numbers come from the same report without
> `--simulate`, aggregating real `query_log` rows. The API route uses the official `anthropic`
> SDK; install it with `pip install -e ".[api]"` and set `ANTHROPIC_API_KEY` (and optionally
> `ANTHROPIC_BASE_URL` for a gateway).

The router also recognizes **Thai** (markers + character-based length), so Thai
queries route on meaning, not just whitespace — e.g. *"เปรียบเทียบและอธิบายว่าทำไม…"* → API.

## Guardrails

The system is built to **refuse rather than guess**:

- **No relevant context → "I don't know".** Empty retrieval never reaches the
  model — the generator returns *"I don't know based on the provided documents."*
  The system prompt also hard-requires answering only from the numbered sources.
  An optional `MIN_RELEVANCE_SCORE` floor (applied to cross-encoder rerank scores,
  which are calibrated) drops weakly-relevant chunks so a confident-but-off-topic
  top hit doesn't get answered. *(In a live trace, the relevant chunk scored
  `4.75` while the rest scored `-9.x` — exactly what the floor filters.)*
- **Prompt-injection screening.** Inputs are checked against a transparent
  denylist ("ignore previous instructions", "reveal your system prompt",
  jailbreak/DAN, Thai equivalents…). A match returns HTTP 400 and is logged — a
  deliberately simple *first* layer, not a complete defense.

## Observability

Every query emits a **structured JSON trace** on the `sourcerer.query` logger —
the machine-readable companion to the `query_log` table:

```json
{"event": "query", "query": "Where are the offices located?",
 "retrieval_mode": "hybrid", "num_retrieved": 5,
 "retrieved": [{"source": "offices.md", "chunk_index": 0, "score": 4.7551}, …],
 "route": "local", "model": "qwen2.5:3b", "difficulty": 0.0,
 "input_tokens": 4095, "output_tokens": 40, "cost_usd": 0.0,
 "latency_ms": 37435, "answered": true, "guardrail": null}
```

Pipe stdout to any log collector and the retrieval trace, route, tokens, latency,
and cost are all queryable. The same fields persist to Postgres (`query_log`),
which the routing report and the `/history` endpoint read.

## Local serving: Ollama → vLLM

Both local backends sit behind one `chat()` wrapper, so switching is config-only:

```bash
# dev (default): Ollama
LOCAL_BACKEND=ollama

# prod: vLLM (OpenAI-compatible, continuous batching)
python -m vllm.entrypoints.openai.api_server --model <model> --port 8001
LOCAL_BACKEND=vllm  VLLM_BASE_URL=http://localhost:8001  VLLM_MODEL=<model>
```

**Measuring the throughput win:** serve the *same* model on each, then drive N
concurrent `/query` requests and compare tokens/sec and p50/p95 latency (the
logged `output_tokens` and `latency_ms` per query feed this). vLLM's batching
pulls ahead decisively as concurrency rises; Ollama stays simplest for single-user
dev. Embeddings always stay on Ollama (`bge-m3`), so only generation moves.

## How to Run

### One command (whole stack)

```bash
cp .env.example .env                 # gitignored; set POSTGRES_PASSWORD, ANTHROPIC_API_KEY…
docker-compose up                    # Postgres (pgvector) + Ollama + the FastAPI app

# First run only — pull the local models into the Ollama container:
docker compose exec ollama ollama pull bge-m3        # embeddings
docker compose exec ollama ollama pull qwen2.5:3b    # generation (or a larger LOCAL_MODEL)

# Ingest a corpus, then open the UI:
docker compose exec api python scripts/ingest.py eval/corpus
open http://localhost:8000           # web UI  ·  /docs for the API
```

### Local dev (no app container)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,api]"          # add ".[rerank]" for the local cross-encoder
docker-compose up -d db ollama       # just the infra
python scripts/ingest.py eval/corpus
uvicorn sourcerer.api.main:app --reload
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"query": "What is the on-call rotation?"}' | jq
```

The response carries `answer`, `citations`, and the routing trace (`route`,
`model`, `router_reason`, `difficulty`, tokens, `cost_usd`). If the answer isn't
in the corpus, you get *"I don't know based on the provided documents."*

> `qwen2.5:32b` is large; for a quick test set `LOCAL_MODEL=qwen2.5:3b` in `.env`.
> Corp proxy blocking in-container model pulls? Run Ollama on the host and point
> `OLLAMA_BASE_URL` at it.

### Make targets

```bash
make test            # pytest          make eval          # run the eval harness
make fmt / make lint # ruff + black    make route-report  # routing split + % saved
```

<details><summary>Older quickstart (Phase 1 — vector-only MVP)</summary>

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

The response contains an `answer` and a `citations` list. If the answer isn't in
the corpus, the system returns *"I don't know based on the provided documents."*

</details>

## Project Structure

```
src/sourcerer/   ingestion · retrieval · routing · generation · eval · api · llm (one client wrapper)
eval/            eval set + generated results
data/            document corpus (gitignored, keep small)
scripts/         CLI entrypoints (ingest, run_eval, route_report)
```

## Trade-offs & Lessons

- **Routing saves money, not always latency.** In a live run the local model
  (`qwen2.5:3b`, CPU) took ~59 s while the frontier API (Sonnet via gateway) answered
  the *harder* query in ~21 s. The cost story still holds ($0 vs ~$0.06/query), but
  "route easy traffic local" is a **cost/privacy** lever — latency depends entirely on
  local hardware. On a GPU (or vLLM, Phase 5) the local path gets competitive; on CPU it
  doesn't. The router instruments both so the trade-off is measured, not assumed.
- **Privacy beats difficulty in routing.** A sensitive query (salary, PII, NDA) is kept
  local even when it scores as "hard" — the opposite of a pure cost-optimiser. That
  ordering is the realistic enterprise default and is enforced before the difficulty gate.
- **Tune heuristics against real queries.** The first live hard query routed *local*
  because the difficulty list was missing the verb **"explain"** — an obvious
  complex-reasoning marker. Live testing (not unit tests) surfaced it; adding it fixed the
  route. Transparent, list-based heuristics make gaps like this debuggable from one logged
  rationale line.
- **Calibrated scores make guardrails possible; raw ones don't.** The relevance floor
  only applies to *cross-encoder rerank* scores — those are comparable across queries.
  RRF/vector scores aren't (an RRF score of 0.016 means nothing in absolute terms), so a
  blanket threshold there would silently drop good answers. Knowing *which* score you can
  threshold is the whole game.

## Roadmap / Phase Status

- [x] Phase 1 — MVP: ingestion + vector retrieval + cited generation behind FastAPI
- [x] Phase 2 — Hybrid retrieval: BM25 + RRF + reranker + swappable chunking
- [x] Phase 3 — Evaluation harness ⭐ (see results above)
- [x] Phase 4 — Hybrid routing: heuristic local-vs-API router + privacy override, per-query cost/latency/route instrumentation, savings report
- [x] Phase 5 — Production polish: vLLM backend (swappable via `LOCAL_BACKEND`), structured per-query observability, guardrails (no-context refusal + prompt-injection screen), one-command `docker-compose`, full README
- [ ] Phase 6 — GraphRAG (optional)
