# Scaling the SQL knowledge base

How the Text-to-SQL / SQL-as-a-source path (Phase 7) was hardened to hold up when
the source database is large — not just a demo SQLite file. Four changes, shipped
across two PRs:

| # | Change | PR |
|---|--------|----|
| 1 | Swappable storage backend (SQLite ↔ DuckDB) | #4 (merged) |
| 2 | Runtime cost guards (timeout + scan budget) | #4 (merged) |
| 3 | Schema retrieval (send only the relevant tables) | #5 |
| 4 | Measured operational metrics in the eval harness | #5 |

Everything is gated or a no-op on small schemas, so the demo behaviour is
unchanged. The work lives under `src/sourcerer/sqlkb/`.

---

## Background: the problem at scale

The SQL knowledge base answers questions over a database via two paths:

- **Path A (RAG ingestion)** — a `SELECT` pulls rows; 1 row = 1 document, embedded
  into the normal hybrid-retrieval pipeline. Good for *content/semantic* questions.
- **Path B (Text-to-SQL)** — analytical/aggregation questions generate a read-only
  `SELECT`, run it, and cite the SQL + result rows. Retrieval can never `SUM`
  thousands of rows; running real SQL gives exact numbers.

This is fine for a demo, but three things break as the source grows toward
"big data":

1. **Path A explodes.** Embedding every transactional row (billions of rows →
   billions of vectors) is an anti-pattern — expensive, and the wrong tool for
   numeric questions anyway.
2. **The engine becomes the bottleneck.** A single read-only SQLite file is a
   row store with no parallelism — a full scan on a large table is slow, and there
   is no statement timeout to stop a runaway query.
3. **The schema stops fitting the prompt.** Dumping every table's DDL into the
   Text-to-SQL prompt blows the context window and buries the relevant tables in
   noise once there are hundreds of them.

The guiding principle behind all four changes: **at scale, push structured work
down to an engine built for it and keep the prompt small and relevant.** Separate
*facts* (numbers → SQL/OLAP) from *semantics* (summaries → vectors); don't embed
raw rows; make Text-to-SQL the primary path and govern its cost.

---

## 1. Swappable storage backend

**Problem.** SQLite (row store, single file, no parallelism) is the wrong engine
for analytics-scale aggregations, but it's the right default for a demo. We need
to swap the engine without touching ingestion, retrieval, or generation.

**What.** A backend abstraction selected by `SQL_KB_BACKEND`, mirroring the
existing `LOCAL_BACKEND` pattern that makes the LLM backend swappable:

- `sqlkb/backends.py` — a `SqlBackend` interface with two implementations:
  - `SQLiteBackend` (default) — opens the file with the `mode=ro` URI.
  - `DuckDBBackend` — a **columnar** engine; aggregations push down to a column
    store instead of scanning a row store. `[duckdb]` is an optional extra, imported
    lazily (clear error if missing).
- `sqlkb/connection.py` became a thin facade over `get_backend(settings)`.
- Each backend owns the three engine-specific things; everything else (running a
  validated `SELECT`, rendering rows) is plain DB-API in the callers:
  1. opening the database read-only,
  2. schema introspection — `sqlite_master` / `PRAGMA` for SQLite vs
     `information_schema` for DuckDB,
  3. bounding a runaway query (see #2).

**How it stays clean.** `loader.py` and `schema.py` access row values
*positionally* (DuckDB returns plain tuples; SQLite Rows also index positionally),
and execution errors are caught engine-agnostically, so the same code drives
either engine.

**Why.** This is the highest-leverage change: it lets Text-to-SQL run on an engine
designed for analytical queries (column pruning, vectorized aggregation, Parquet)
while the demo keeps zero extra dependencies. It also matches the project's
established "swap the backend behind one interface" convention.

**Config.** `SQL_KB_BACKEND=sqlite|duckdb` · install with `pip install '.[duckdb]'`.

---

## 2. Runtime cost guards

**Problem.** `safety.safe_select` enforces a `LIMIT`, but `LIMIT` caps the rows
*returned*, not the rows *scanned*. A generated query can still full-scan a huge
table (or aggregate over it) before the limit applies. SQLite has no statement
timeout; DuckDB has no per-statement timeout setting either.

**What.** `backends.query_guard(conn, settings)` — a context manager wrapped around
`execute`/`fetchall` in `text_to_sql._execute`, with two budgets:

- **`SQL_KB_TIMEOUT_S`** — a wall-clock cap that aborts on *both* engines:
  - SQLite: a progress handler fires every N VM ops and returns non-zero past the
    deadline (SQLite then aborts the statement).
  - DuckDB: a watchdog thread calls `conn.interrupt()` when the deadline passes.
- **`SQL_KB_MAX_SCAN_OPS`** — a SQLite VM-op budget, a proxy for rows/bytes scanned
  (the warehouse analogue is e.g. BigQuery's `maximum_bytes_billed`).

A trip flips a `Guard.tripped` reason; the execution path turns the engine's generic
"interrupted" error into a precise `QueryCostError("query exceeded time limit / scan
budget")`, which flows into the existing **"I don't know"** guardrail — never a guess,
never a crash.

**Why.** Defence in depth against a query that would otherwise hammer the database.
On a real warehouse the same shape maps to native cost controls; here it's a
portable guard that works identically on both backends.

**Config.** `SQL_KB_TIMEOUT_S=5.0` (0 = off) · `SQL_KB_MAX_SCAN_OPS=0` (0 = off).

---

## 3. Schema retrieval

**Problem.** The Text-to-SQL prompt previously described *every* table (columns +
sample rows). On a wide schema (hundreds of tables) that overflows the context and
makes the model pick the wrong tables — accuracy drops as the schema grows.

**What.** Retrieve the tables a question actually needs (`sqlkb/schema_retrieval.py`),
reusing the project's signature move — **hybrid ranking fused with RRF**:

- **Lexical** ranking — token overlap between the question and each table's
  signature. Deterministic, free, no service.
- **Embedding** ranking — `bge-m3` cosine between the question and each signature.
  `bge-m3` is multilingual, so it catches semantic / cross-language matches a Thai
  question needs against English column names.
- The two rankings are fused with **Reciprocal Rank Fusion** (the same scale-free
  fusion used for document retrieval), and the top-K tables are kept.

**The scalable shape.** Ranking runs over cheap **signatures** (`TABLE x (cols)`,
no data scan — `schema.table_signatures`), built for *every* table. The expensive
full descriptions *with sample rows* are then built only for the **survivors**
(`describe_schema(..., tables=...)`). Cheap ranking over all; costly description over
a handful.

**Robustness.** The embedding leg is **best-effort** — if the embedding backend is
down it returns nothing and the system falls back to lexical-only, so Text-to-SQL
never breaks. Below the `SQL_KB_SCHEMA_TOP_K` threshold (or with it set to 0) every
table is described — identical to the old behaviour, and a small DB never touches
the embedding backend.

**Why.** A wide schema is exactly where a naive Text-to-SQL prompt falls apart.
Retrieving the relevant subset keeps the prompt small and on-topic, and reuses the
exact hybrid-retrieval idea the project is built around — applied to schema instead
of documents.

**Config.** `SQL_KB_SCHEMA_TOP_K=8` (0 = always describe all tables).

---

## 4. Measured operational metrics

**Problem.** The project's hard rule: *a change isn't "better" until the harness
says so.* The scale work above (backend, guards, retrieval) moves operational
numbers — latency, tokens, prompt size — that the eval didn't capture.

**What.** `scripts/sql_eval.py` now reports, beside execution accuracy:

- **Latency** per query — mean / median / p95 (a `_percentile` helper).
- **SQL-generation tokens** per query.
- **Prompt-schema narrowing** — mean tables in the prompt vs the whole schema; the
  visible effect of schema retrieval. `SQLExecution.prompt_tables` exposes the
  per-query table count for free (the schema string already exists in `run()`).
- **Cost-guard aborts** — how many queries were killed by the timeout / scan budget.

**Result** (qwen2.5, 1-table demo DB):

```
Execution accuracy: 6/6 = 100%
Latency ms: mean 2174 · median 1053 · p95 6367
SQL-gen tokens/query: mean 284
Prompt schema: mean 1.0/1 tables (100% of schema)
Cost-guard aborts: 0/6
```

The demo DB has one table, so there's nothing to narrow — the metric correctly
reports `1/1`. On a wide schema this is where retrieval shows up; unit tests
exercise that path against a 10-table fixture.

**Why.** These numbers are what make the scale work defensible rather than
"this should be faster/cheaper" — and they're the table you'd extend when pointing
the backend at a real warehouse.

---

## Configuration reference

| Setting | Default | Purpose |
|---------|---------|---------|
| `SQL_KB_BACKEND` | `sqlite` | Storage engine: `sqlite` \| `duckdb` (columnar). |
| `SQL_KB_TIMEOUT_S` | `5.0` | Wall-clock cap per query, both engines (0 = off). |
| `SQL_KB_MAX_SCAN_OPS` | `0` | SQLite VM-op budget ~ rows/bytes scanned (0 = off). |
| `SQL_KB_SCHEMA_TOP_K` | `8` | Above this many tables, retrieve only the relevant ones (0 = all). |

---

## Intentionally deferred

Pragmatic stopping points, noted so the next step is obvious:

- **Native warehouse cost controls.** `SQL_KB_MAX_SCAN_OPS` is a SQLite op-count
  proxy; a real engine (BigQuery `maximum_bytes_billed`, ClickHouse `max_rows_to_read`)
  would enforce a true bytes-scanned cap, and `EXPLAIN`/dry-run could estimate cost
  before executing.
- **Pre-aggregation for Path A.** At scale, ingest periodic rollups/summaries (à la
  GraphRAG community summaries) and embed those, instead of raw rows.
- **Semantic layer / curated metrics + few-shot SQL** for very wide or ambiguous
  schemas, on top of table retrieval.
- **Result + semantic caching** for repeated questions.
