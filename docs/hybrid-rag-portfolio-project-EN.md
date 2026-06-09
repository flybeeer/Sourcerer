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
                    ┌─────────────────────────────────────┐
   Documents ───►   │  Ingestion: chunk + embed + index    │
 (PDF/MD/HTML)      └──────────────┬──────────────────────┘
                                   ▼
                    ┌─────────────────────────────────────┐
   User query ──►   │  Hybrid Retrieval                    │
                    │  vector (pgvector) + BM25 + reranker │
                    └──────────────┬──────────────────────┘
                                   ▼
                    ┌─────────────────────────────────────┐
                    │  Router: easy/sensitive → local      │
                    │          hard → API (frontier)       │
                    └──────────────┬──────────────────────┘
                                   ▼
                    ┌─────────────────────────────────────┐
                    │  Generation + citations              │
                    └──────────────┬──────────────────────┘
                                   ▼
                    ┌─────────────────────────────────────┐
                    │  Eval harness + logging/observability│
                    └─────────────────────────────────────┘
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

> Pick a dataset that is "hard to answer with plain ChatGPT" — e.g. internal company-policy
> documents, domain-specific technical manuals, or a public corpus (e.g. public legal/medical
> documents). The more specialized, the more it showcases the value of RAG.

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
