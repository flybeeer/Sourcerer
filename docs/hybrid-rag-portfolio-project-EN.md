# Sourcerer — Portfolio Project Blueprint

> A hybrid RAG knowledge assistant that answers from your documents — with sources.
> The name plays on *source* + *sorcerer*: every answer is grounded in cited sources.

**Sourcerer** is an assistant that answers questions grounded in an internal document corpus, built
on a **hybrid** architecture (local LLM + API), **hybrid retrieval** (vector + keyword + reranker),
with an **evaluation harness** and production readiness — designed to showcase the full stack of
LLM-engineering skills.

---

## Why This Project Works for a Portfolio

People hiring for LLM roles don't care whether you can call an API (anyone can). They care whether
you can "build something that works in the real world and prove it." This project deliberately
includes the three things that separate amateurs from professionals:

1. **Hybrid retrieval that is measured** — not bare vector search.
2. **An evaluation harness** — numbers that prove the system is actually good, not "looks fine to me."
3. **Local vs API routing** — demonstrates understanding of cost, privacy, and the real trade-offs
   organizations face.

---

## Target Architecture

```
   Documents (PDF/MD/HTML) ─┐
                            ├─►  Ingestion: chunk + embed + index → pgvector + BM25
   DB rows (SQLite) ────────┘    (Phase 7: 1 row = 1 doc, same pipeline)


   User query ──►   ┌─────────────────────────────────────────────┐
                    │  Query classifier — route by question shape │
                    ├─────────────────────────────────────────────┤
                    │  analytical?  → Text-to-SQL (Phase 7)        │
                    │     schema → SELECT → validate → run → cite  │
                    │  overview?    → GraphRAG global (Phase 6)    │
                    │  otherwise    → Hybrid Retrieval (default)   │
                    │     vector (pgvector) + BM25 + reranker      │
                    └──────────────────────┬──────────────────────┘
                                           ▼
                    ┌─────────────────────────────────────────────┐
                    │  Router: easy/sensitive → local             │
                    │          hard → API (frontier)              │
                    └──────────────────────┬──────────────────────┘
                                           ▼
                    ┌─────────────────────────────────────────────┐
                    │  Generation + citations                     │
                    └──────────────────────┬──────────────────────┘
                                           ▼
                    ┌─────────────────────────────────────────────┐
                    │  Eval harness + logging/observability       │
                    └─────────────────────────────────────────────┘
```

### Technology Choices per Layer

| Layer | Choice | Reason |
|-------|--------|--------|
| Local inference | **Ollama** (dev) → **vLLM** (prod) | The standard enterprise path |
| Local model | **Qwen 2.5 32B** or **Llama 3.x**, quantized (INT4) | The "local-grade" size with the best value right now |
| API model | Claude / GPT (a frontier model) | Baseline comparison + handling hard queries |
| Vector DB | **pgvector** (start) or **Qdrant/Weaviate** | pgvector builds on existing Postgres, easy to control |
| Keyword | **BM25** (via Postgres FTS or OpenSearch) | Half of hybrid retrieval |
| Reranker | **bge-reranker** or Cohere Rerank API | Clearly boosts result precision |
| Orchestration | **LlamaIndex** or write it yourself | Write it yourself first to understand the mechanics |
| Embedding | **bge-m3** or OpenAI embeddings | Pick one that can run locally to stay consistent with the theme |
| Backend/API | **FastAPI** | Python standard for serving |
| Frontend | Streamlit (fast) or Next.js (shows craft) | Depends on your time budget |
| Graph index *(Phase 6)* | **networkx** + modularity communities | Builds the entity/relationship graph and clusters it for whole-corpus "overview" questions; pure-Python, no graph DB to operate. The `[graphrag]` extra. |
| Graph store *(Phase 6)* | **JSON file** → **Postgres + pgvector** | `GRAPHRAG_STORE`: JSON loads the whole graph in memory (small corpora); Postgres ranks community summaries via pgvector HNSW — O(k) reads at scale, reusing existing infra. |
| Database source *(Phase 7)* | **SQLite** (read-only) → **DuckDB** (columnar) | Treat a database as a knowledge source too. SQLite is zero-setup for a demo; DuckDB pushes aggregations down at analytics scale. Swappable via `SQL_KB_BACKEND`, mirroring the `LOCAL_BACKEND` pattern. |
| Text-to-SQL *(Phase 7)* | **Local LLM** → validated read-only `SELECT` | Analytical/aggregation questions chunking can't answer (e.g. `SUM` over thousands of rows). The executed SQL + result rows become the citation. |
| Schema retrieval *(Phase 7)* | **bge-m3 + lexical → RRF** | On a wide schema, send only the most relevant tables to the prompt — the same hybrid-retrieval idea, applied to schema selection. |

> Pick a dataset that is "hard to answer with plain ChatGPT" — e.g. internal company-policy
> documents, domain-specific technical manuals, or a public corpus (e.g. public legal/medical
> documents). The more specialized, the more it showcases the value of RAG.

> **Note — keyword search when the vector store isn't Postgres.** BM25 is free here because
> Postgres holds both the vectors (pgvector) and the FTS keyword index. If you swap in a dedicated
> vector DB, get the keyword leg from: (1) a vector DB with hybrid built in — Weaviate (native
> BM25) or Qdrant/Milvus/Pinecone (sparse vectors like SPLADE/BM42 — BM25-style keyword importance
> stored *as* a vector, keeping it one system); (2) a search engine — Elasticsearch/OpenSearch;
> or (3) an in-process lib (`bm25s`, `rank_bm25`) for small corpora. The RRF fusion + reranker
> stages are unchanged — only the source of the keyword list differs.

---

## Broken into 5 Phases

### Phase 1 — MVP RAG (Week 1)
Goal: get the pipeline working end-to-end first, then improve quality.

- Set up Ollama + pull a model to run locally.
- Write a simple ingestion pipeline: load documents → chunk → embed → store in pgvector.
- Write vector-only retrieval → pass context to the model → answer with citations.
- Wrap it in FastAPI + a simple UI.

**Deliverable:** asking a question returns an answer that cites documents, even if not yet polished.

### Phase 2 — Hybrid Retrieval (Week 2)
Goal: raise retrieval quality, which is the real heart of RAG.

- Add BM25 keyword search in parallel with vector search.
- Fuse the two result sets with reciprocal rank fusion (RRF).
- Add a reranker to re-score results before passing them to the model.
- Experiment with multiple chunking strategies (fixed-size vs semantic vs contextual).

**Deliverable:** show that hybrid + reranker beats vector-only (proven in Phase 3).

### Phase 3 — Evaluation Harness (Weeks 2–3) ⭐ The Standout Piece
Goal: prove it with numbers. This is what puts your portfolio ahead of 90% of others.

- Build an eval set: tuples of (question, correct answer, documents that should be retrieved),
  about 50–100 items.
- Measure **retrieval metrics**: recall@k, MRR, hit rate.
- Measure **generation metrics**: faithfulness (does the answer match the context?),
  answer relevancy — using an LLM-as-judge approach.
- Build a comparison table of configurations (vector-only vs hybrid vs hybrid+rerank) with clear numbers.
- Helpful tools: **RAGAS** or write your own harness.

**Deliverable:** a results table + charts saying "which config is best and why."

### Phase 4 — Hybrid Routing (Weeks 3–4)
Goal: demonstrate the cost/privacy thinking that real organizations care about.

- Write a router: classify a query as "easy/sensitive" → local, "hard/complex reasoning" → API.
- Capture metrics: share of queries going local, cost per query, latency per route.
- Compute the percentage saved versus calling the API for everything
  (industry reference target ~40–70%).

**Deliverable:** cost/latency charts + a routing rationale you can explain.

### Phase 5 — Production Polish (Week 4)
Goal: make it look like the real thing, not a toy.

- Migrate inference from Ollama → **vLLM**, measure the throughput gain.
- Add observability: log every query, retrieval trace, token/cost tracking.
- Add guardrails: handle the "no answer in context" case (answer "I don't know" instead of
  hallucinating), basic prompt-injection checks.
- Dockerize the whole system with docker-compose.
- Write a strong README (very important — see below).

**Deliverable:** a repo that runs immediately with `docker-compose up`.

---

## The README That Gets You an Interview

Good code isn't enough if you can't tell the story. Your README should have:

1. **Demo** — a short GIF or video that shows it working in the first 10 seconds.
2. **The problem solved** — "answers questions from internal docs X that plain ChatGPT can't."
3. **Architecture** — a diagram + the reasons you chose each technology (shows you can *decide*).
4. **Evaluation results** — the tables/charts from Phases 3 and 4. **This is the star.**
5. **Trade-offs encountered** — what didn't work and how you fixed it (shows engineering maturity).
6. **How to run** — clear, actually reproducible.

---

## Phase 6 (Optional Extension) — GraphRAG ⭐ Makes the Portfolio Stand Out

If you've finished Phases 1–5 and want to elevate it well beyond a typical portfolio, this is the
most worthwhile extension — and more popular in industry than a full KAG implementation.

### What GraphRAG Is, and How It Differs from Plain RAG

GraphRAG (by Microsoft) was created to fix one specific weakness of vector RAG: plain RAG **can't
answer big-picture questions about the whole corpus**, because it searches by text similarity. A
question like "What are the top 5 themes across all the data?" performs poorly because nothing in the
query points to the right text. GraphRAG fixes this by building a graph structure that describes the
"overview" of the dataset in advance.

It works in two major stages:

1. **Indexing** — use an LLM to extract entities and relationships from the documents into a
   knowledge graph, then cluster the nodes into hierarchical "communities" (Microsoft uses the Leiden
   algorithm), and have the LLM pre-write a summary of each community (a "community report"). The
   result is a multi-level knowledge graph, from fine detail up to broad overview.
2. **Query** — pull this structure in as extra context for the LLM when answering.

### Two Search Modes You Must Understand

| Mode | Best for | Example question |
|------|----------|------------------|
| **Local search** | Specific questions about a single entity | "Who is company X related to?" |
| **Global search** | Whole-corpus overview questions | "What are the main themes of all the data?" |

Global search works map-reduce style: it splits community reports into chunks, has the LLM summarize
each chunk with an importance score, then aggregates only the most important points for the final
answer. Your existing vector RAG is still best for who/what/when/where questions whose answer lives
in a specific span of text.

### Recent Developments Worth Knowing (and Mentioning in Interviews)

- **DRIFT search** — combines global + local; the recommended mode for questions needing both breadth
  and depth.
- **LazyGraphRAG** — addresses GraphRAG's "expensive" problem by deferring graph extraction, cutting
  indexing cost dramatically (claimed to ~0.1% of full GraphRAG). A far more practical option for a
  personal project.

### ⚠️ Important Cost Warning

Full GraphRAG is **expensive**. The indexing stage burns tokens having the LLM read and extract over
the whole corpus; a 1MB corpus has been reported to cost roughly $5–20 in API fees depending on the
model. So:

- **Always start with a small corpus.** Don't index a large one yet.
- Consider **LazyGraphRAG** or a lightweight library (e.g. nano-graphrag) instead of the full version.
- Or use **your own local LLM for the indexing stage** to cut API cost — which fits this project's
  hybrid theme perfectly and is a good selling point.

### How to Extend the Existing Project

The nice part: you don't rip anything out. Add it as **another retrieval path** parallel to the
existing hybrid retrieval:

1. Add a pipeline to build the graph + community summaries (use the local LLM for extraction to
   control cost).
2. Add router logic: overview questions → GraphRAG global; specific questions → existing hybrid retrieval.
3. **Reuse the Phase 3 eval harness** to compare whether GraphRAG actually beats hybrid RAG on
   "overview" questions, and how much more it costs.

Step 3 is the gold — you'll have a table showing "for which question type which technique wins, and
whether it's worth the cost," which is exactly the kind of engineering decision senior interviewers
look for.

---

## Phase 7 (Optional Extension) — Database as a Knowledge Base

A knowledge base isn't only documents — a lot of an organization's truth lives in a **database**.
This phase makes a SQLite DB a first-class source. The key insight (and the classic RAG mistake to
avoid) is that database content splits into **two question types that need different machinery**:

| Question type | Example | Right tool |
|---------------|---------|-----------|
| Content / semantic | "What do customers complain about?" | **Hybrid RAG** (chunk + embed the rows) |
| Analytical / aggregation | "Total sales last year", "How many customers in the north?" | **Text-to-SQL** (generate `SELECT SUM(...)`, run it) |

Chunking **cannot** answer the aggregation question: retrieval only pulls the top-k chunks, so it can
never `SUM` thousands of rows, and an LLM adding numbers from text is unreliable. So you route each:

1. **Ingestion (RAG path)** — a read-only `SELECT` pulls rows; turn **1 row = 1 document** (source
   label `kb:<id>` for traceable citations) and run them through the *existing* chunk→embed→store
   pipeline. DB rows become just another `source` — no new retrieval code.
2. **Text-to-SQL path** — analytical questions are auto-detected by the router (markers like
   *total / how many / average / per year* + Thai equivalents) and answered live: introspect the
   schema → LLM writes one `SELECT` → **validate it's a single read-only query** → execute → phrase
   the answer. The **executed SQL + result rows are the citation** — transparent and re-runnable.

### Safety Is Layered (the part interviewers probe)

Letting an LLM write SQL against your data is scary unless you box it in:

- The SQLite file is opened **read-only** (`mode=ro` URI).
- The generated SQL is rejected unless it's a **single statement** starting with `SELECT`/`WITH`,
  with no `INSERT/UPDATE/DELETE/DROP/PRAGMA/…`; a row `LIMIT` is always enforced.
- A rejected query yields *"I don't know"*, never a guess.
- Privacy still applies — a sensitive query keeps SQL generation on the local model.

### Scaling It Beyond a Demo (where the engineering judgment shows)

The same way the main project shows judgment via eval tables, this path shows it via three scaling
levers — each with a clear *what / how / why*:

- **Swappable storage engine** (`SQL_KB_BACKEND`) — SQLite (default) ↔ DuckDB (columnar). DuckDB
  pushes aggregations down for analytics-scale tables; introspection via `information_schema`.
  Mirrors the `LOCAL_BACKEND` swap pattern.
- **Runtime cost guards** — a wall-clock timeout aborts a runaway query on both engines, plus a
  SQLite scan-op budget as a rows-scanned proxy. A trip → "I don't know", never a hung request.
- **Schema retrieval** — on a wide schema, rank tables by lexical overlap + bge-m3 embedding cosine,
  fused with **RRF**, and send only the top-k tables' DDL to the prompt. The hybrid-retrieval idea
  reused for schema selection.

### How to Extend the Existing Project

Like GraphRAG, you don't rip anything out — it's another **answer path** chosen by the router:

1. Add `ingest_sql.py` (reuses the existing storage pipeline) for the RAG path.
2. Add a Text-to-SQL module: schema introspection → SQL generation → `safe_select` validation →
   execution → cited NL answer. Wire the router to send analytical questions here.
3. **Reuse the eval idea** — measure *execution accuracy* (does the generated query's result match a
   hand-written reference query's?), the standard Text-to-SQL metric, alongside latency, SQL-gen
   tokens, schema-narrowing, and cost-guard aborts.

Step 3 is again the gold: a number that proves the SQL path works ("6/6 execution accuracy on the
demo set"), plus operational metrics that prove the scaling levers do something — not just "it
looks right."

---

## Common Pitfalls (Watch Out)

- **Bad chunking breaks the whole system** — no matter how good retrieval is, it can't help if chunks
  are split poorly. Invest in this step.
- **Skipping evaluation** — then you'll never know if a change actually improved things. Do not skip
  Phase 3.
- **Stuffing in too much context** — more chunks doesn't mean better; sometimes it's worse due to noise.
- **Forgetting the "no answer" case** — a good system dares to say it doesn't know, instead of
  guessing every time.
- **Starting at vLLM** — it's fiddlier to set up. Get Ollama working end-to-end first, then migrate.

---

## Order of Execution (If Starting Today)

1. Pick a dataset specialized enough to showcase the value of RAG.
2. Set up Ollama + pgvector with docker-compose.
3. Get Phase 1 working end-to-end first — don't worry about quality yet.
4. Once the MVP runs, work through Phases 2 → 5 one step at a time, with eval as your compass.

Start small, get it working, then make it good — and measure at every step.
