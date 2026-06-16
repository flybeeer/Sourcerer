# Sourcerer — End-to-end flow

How a document or row becomes searchable knowledge (ingest), and how a question is
routed to the right answer path (query). Names below match the real code: env flags,
functions, and modules in `src/sourcerer/`.

---

## 1) INGEST — getting knowledge into the system

```
SOURCES                          TRANSFORM (load → chunk → embed → store)
─────────────────────────────────────────────────────────────────────────────

┌─ FILE ────────────────┐
│ scripts/ingest.py     │   iter_documents()        ┌──────────────────────────┐
│ data/raw/*.pdf .md    │──► load_pdf / load_text ──►│                          │
│ .txt .markdown        │   → (filename, text)      │  _store_document()       │
└───────────────────────┘                           │                          │
                                                     │  1. chunk(text)          │
┌─ DATABASE (SQLite) ───┐                            │     • CHUNK_STRATEGY      │
│ scripts/ingest_sql.py │   iter_rows()             │       = fixed | semantic │
│ read-only SELECT      │──► safe_select() (ro)  ───►│  2. embed_texts()        │
│ 1 row = 1 document    │   → (kb:<id>, "col: val") │     • bge-m3 (Ollama)     │
└───────────────────────┘                            │  3. DELETE old source    │
                                                     │     (idempotent re-ingest)│
                                                     │  4. INSERT → chunks       │
                                                     └────────────┬─────────────┘
                                                                  ▼
                                              ┌──────────── Postgres: chunks ─────────────┐
                                              │  embedding vector(1024)  ← VECTOR search   │
                                              │  content_tsv (generated) ← BM25 / keyword  │
                                              └────────────────────────────────────────────┘
                                                                  │
                              (separate, expensive, run manually) │  scripts/graphrag_index.py
                              gate: GRAPHRAG_ENABLED               ▼  reads ALL chunks
                                              ┌──────────── GRAPH build ───────────────────┐
                                              │ extraction (entities + relationships)      │
                                              │   • local model (GRAPHRAG_EXTRACTION_MODEL)│
                                              │   • or API if GRAPHRAG_EXTRACT_WITH_API     │
                                              │ → build graph → communities (modularity)   │
                                              │ → summaries (embedded into pgvector)       │
                                              │ store: GRAPHRAG_STORE = json | postgres     │
                                              └────────────────────────────────────────────┘
```

**Transform conditions**

| Step | Condition |
|---|---|
| Which chunker | `CHUNK_STRATEGY` = `fixed` (sliding window) or `semantic` (split on meaning) |
| Embeddings | Always bge-m3 on Ollama (even when generation runs on vLLM) |
| BM25 index | `content_tsv` is a generated column → built for every row, unconditionally |
| Graph | Built only when you run `graphrag_index.py` (expensive); used at query time only if `GRAPHRAG_ENABLED` |

> Vector and BM25 are produced **together** on the same row (two columns), always.
> **Ranking (rerank)** is *not* an ingest step — it happens only at **query** time.

---

## 2) QUERY — `POST /query` (the real decision order in `routes.py`)

```
                          ┌─────────────────────────────┐
   question ─────────────►│ Guardrail 1: injection check│  ENABLE_INJECTION_CHECK
                          └──────────────┬──────────────┘  detected → 400 reject
                                         ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ (1) Text-to-SQL?                                              │
        │   forced route_override="sql"  OR                            │
        │   (auto + SQL_KB_ENABLED + is_analytical_query)               │
        │   markers: total / how many / average / per year / ยอดรวม ... │
        └───────────────┬──────────────────────────────────────────────┘
              DB exists? │ yes                      no → fall to (2)
                         ▼
        ┌──── TEXT-TO-SQL PATH (_text_to_sql_query) ──────────────────┐
        │ schema introspect → LLM writes SELECT → safe_select() check  │
        │ → execute on SQLite (read-only) → phrase the answer          │
        │ citation = the executed SQL + result rows                    │
        │   • write SQL: local (default) / API if SQL_KB_GENERATE_WITH_API & not sensitive │
        │   • phrase answer: follows the router decision (local/api)   │
        └──────────────────────────────────────────────────────────────┘
                                         │ (if not (1))
                                         ▼
        ┌──────────────────────────────────────────────────────────────┐
        │ (2) GraphRAG global?                                         │
        │   forced graph-local / graph-api  OR                        │
        │   (auto + GRAPHRAG_ENABLED + is_overview_query)              │
        │   markers: main themes / overall / across all / ภาพรวม ...    │
        └───────────────┬──────────────────────────────────────────────┘
          index exists? │ yes                       no → fall to (3)
                        ▼
        ┌──── GRAPHRAG GLOBAL PATH (_graph_global_query) ─────────────┐
        │ select communities → MAP (per community) → REDUCE (synthesis)│
        │   • MAP : always local ($0)                                  │
        │   • REDUCE: local  → "graph-local"                           │
        │             API    → "graph-api" (GRAPHRAG_REDUCE_WITH_API)  │
        │ citation = community · source files                          │
        └──────────────────────────────────────────────────────────────┘
                                         │ (if not (2))
                                         ▼
        ┌──── HYBRID RAG PATH (default) ──────────────────────────────┐
        │ mode = RETRIEVAL_MODE (hybrid | vector)                      │
        │                                                              │
        │  retrieve:                                                   │
        │   vector search ─┐                                           │
        │                  ├─► RRF fusion ─► [rerank?] ─► top_k chunks  │
        │   BM25 search  ──┘   (RRF_K)      request.rerank & hybrid     │
        │                                   RERANKER_TYPE              │
        │                                                              │
        │  Guardrail 2: filter_relevant()                              │
        │   empty / low score (MIN_RELEVANCE_SCORE) → "I don't know"   │
        │                                                              │
        │  generate with the client the router picks ▼                 │
        └──────────────────────────┬───────────────────────────────────┘
                                   ▼
                   ┌──── ROUTER: local vs API (query_router.route) ────┐
                   │ 1. sensitive term? (salary/PII/เงินเดือน/ความลับ) │
                   │      → LOCAL always (privacy override wins)        │
                   │ 2. difficulty ≥ ROUTER_HARD_THRESHOLD (0.7)?       │
                   │      → API  (markers: why/compare/analyze/ทำไม...) │
                   │ 3. else → LOCAL                                    │
                   │ * API but no ANTHROPIC_API_KEY → fall back LOCAL   │
                   └────────────────────────────────────────────────────┘
                          LOCAL = qwen2.5 (Ollama/vLLM, $0)
                          API   = frontier model (Anthropic, billed)
```

---

## 3) Routing conditions (priority, top to bottom)

| # | Path | Entry condition | LLM used |
|---|---|---|---|
| 0 | **Reject** | injection check trips | — (400) |
| 1 | **Text-to-SQL** | `"sql"` OR (auto + `SQL_KB_ENABLED` + analytical question) + DB file exists | SQL: local/API · answer: per router |
| 2 | **GraphRAG global** | `graph-local/graph-api` OR (auto + `GRAPHRAG_ENABLED` + overview question) + index exists | map: local · reduce: local/API |
| 3 | **Hybrid RAG** (default) | every remaining question | local/API per router |

**Router rule** (used inside path 3, and to pick the answer client of path 1):

1. **sensitive → local always** (privacy beats difficulty)
2. **difficulty ≥ 0.7 → API**
3. **otherwise → local**
4. would go API but no key → fall back to local

---

**Design principle:** different question shapes go to different paths — numbers /
aggregation → Text-to-SQL (runs real SQL), whole-corpus themes → GraphRAG,
specific content → Hybrid RAG. Inside each path, the router then chooses local vs
API by difficulty and privacy. Every path answers **with citations**.

---

## 4) Query classification — markers & scoring

There are **two levels** of classification, and they do different jobs:

| Kind | Answers | Output | Mechanism |
|---|---|---|---|
| **Path predicates** (`is_analytical_query`, `is_overview_query`) | "which path?" | binary True/False | substring match |
| **Router scoring** (`_SENSITIVE` / `_REASONING` / `_SIMPLE`) | "local or API?" | score 0–1 | additive scoring |

All three are deliberately **list-based** — transparent and debuggable from one
logged rationale line — trading some classifier accuracy for explainability.

### 4a) Path predicates (binary)

Both work identically: lowercase the query, return True if **any** marker is a
substring. One hit is enough.

```python
def is_analytical_query(query: str) -> bool:   # routing/router.py
    q = query.lower()
    return any(m in q for m in _ANALYTICAL)

def is_overview_query(query: str) -> bool:     # graphrag/search.py
    q = query.lower()
    return any(m in q for m in _OVERVIEW_MARKERS)
```

**`_ANALYTICAL`** (→ Text-to-SQL) — aggregation/number questions:
- **EN:** `total`, `sum of`, `how many`, `count of`, `number of`, `average`, `avg`,
  `mean`, `median`, `maximum`, `minimum`, `highest`, `lowest`, `most`, `least`,
  `per month/year/day`, `by month/year/region`, `grouped by`, `trend`, `growth`,
  `percentage of`
- **Thai:** `ยอดรวม`, `ยอดขาย`, `รวมทั้งหมด`, `ทั้งหมดกี่`, `จำนวน`, `กี่`, `เฉลี่ย`,
  `ค่าเฉลี่ย`, `มากที่สุด`, `น้อยที่สุด`, `สูงสุด`, `ต่ำสุด`, `ต่อเดือน`, `ต่อปี`, `ต่อวัน`,
  `แต่ละเดือน`, `แต่ละปี`, `เปอร์เซ็นต์`, `ร้อยละ`, `แนวโน้ม`

**`_OVERVIEW_MARKERS`** (→ GraphRAG global) — whole-corpus themes no single chunk
holds:
- **EN:** `overall`, `in general`, `main theme(s)`, `key themes/topics`,
  `across all`, `whole corpus`, `high-level`, `summarize the`, `summary of all`,
  `recurring`, `what themes`, `common themes`, `big picture`
- **Thai:** `ภาพรวม`, `ธีมหลัก`, `หัวข้อหลัก`, `โดยรวม`, `สรุปทั้งหมด`

> **Limitation:** raw substring matching can false-positive (e.g. Thai `ยอดขาย`
> inside an explanatory "explain the sales" question, or the very short `กี่`
> matching mid-word). Two safety nets: the user can override with
> `route_override`, and even a wrong SQL route stays safe — if no valid SELECT
> passes `safe_select`, the answer is "I don't know", never a guess.

### 4b) Router scoring (local vs API)

These markers don't pick a path — they feed `_difficulty`, which decides the
generation backend *inside* a path.

- **`_SENSITIVE`** — privacy; if hit, force **local always** (before scoring):
  `salary`, `password`, `ssn`, `confidential`, `nda`, `pii`, `bank account`,
  `layoff` … / Thai `เงินเดือน`, `รหัสผ่าน`, `บัตรประชาชน`, `ความลับ`,
  `ข้อมูลส่วนบุคคล`, `บัญชีธนาคาร`, `เลิกจ้าง` …
- **`_REASONING`** — "hard" signal, pushes the score up: `why`, `how does`,
  `compare`, `explain`, `analyze`, `evaluate`, `trade-off`, `root cause`,
  `design` … / Thai `ทำไม`, `เปรียบเทียบ`, `อธิบาย`, `วิเคราะห์`, `ประเมิน`,
  `ข้อดีข้อเสีย`, `ความสัมพันธ์`, `ออกแบบ` …
- **`_SIMPLE`** — "easy" signal, pulls the score down: `what is`, `who is`,
  `define`, `list the`, `how many`, `how much` … / Thai `คืออะไร`, `ใคร`, `ที่ไหน`,
  `เมื่อไร`, `กี่`, `เท่าไร` …

**Scoring (`_difficulty`)** — additive, then clamped to [0, 1]:

```
score = 0
if any _REASONING marker:        score += min(0.7, 0.4 + 0.15 × (num_markers − 1))
if word_count ≥ 40:              score += 0.3        (≥ 20 → +0.15)
if num '?' ≥ 2:                  score += 0.15       (multi-part question)
if _SIMPLE and words < 20 and no _REASONING: score −= 0.3
score = clamp(0, 1)
```

**Decision:**
1. `_SENSITIVE` hit → **local** (stop, ignore score)
2. `score ≥ ROUTER_HARD_THRESHOLD` (0.7) → **API**
3. otherwise → **local**
4. would be API but no `ANTHROPIC_API_KEY` → fall back to **local**

> Thai has no spaces, so `_word_count` approximates Thai length as characters ÷ 4.

| Example | Score | Route |
|---|---|---|
| "What is the PTO policy?" | simple + short → −0.3 → **0.0** | local |
| "เงินเดือน senior เท่าไหร่" | sensitive! | local (override) |
| "Compare A and B and explain why …" | reasoning×2 (0.55) + long → **≥ 0.7** | API |

**Summary:** `is_analytical_query` / `is_overview_query` are switches for the
**retrieval path** (one marker = True); `_SENSITIVE` / `_REASONING` / `_SIMPLE`
feed a **score** that picks the **generation backend** (local/API) within a path.
