# Confluence Document Data Quality (design — not built yet)

> Status: **PLANNED, design only.** Nothing in this doc is implemented.
> Prerequisite: Confluence ingestion + `document_metadata` table (done — see CLAUDE.md).

## Why

The live QA-space test batch (44 pages, scbtechx.atlassian.net) surfaced real data-quality
problems that directly hurt RAG answer quality:

- ~40/44 pages were last modified in **2018–2020** — stale content enters the context and the
  model answers confidently from 7-year-old process docs.
- Ownership is broken: most authors are `(Unlicensed)` accounts and 3 pages belong to
  `ผู้ใช้เดิม (Deleted)` — nobody to ask when content is wrong.
- **0/44 pages carry a label** — no categorization signal at all.
- Several pages are `version = 1` — written once, never reviewed.

Goal: a measurable per-document quality score (project rule: every claim gets a number),
usable to exclude bad documents from retrieval or to warn in citations
("this source hasn't been touched in 7 years").

## Case 1 — metadata-based checks (cheap: SQL over `document_metadata`, no LLM)

| Check | DQ dimension | Rule | Observed in QA corpus |
|---|---|---|---|
| Staleness | Timeliness | `last_modified` older than threshold (default 2y) | ~40/44 stale |
| Orphaned ownership | Accountability | `created_by`/`last_modified_by` contains `(Deleted)` / `(Unlicensed)` | 3 deleted, most unlicensed |
| Label completeness | Completeness | `labels = []` | 44/44 unlabeled |
| Never reviewed | Currency | `version = 1` | several |
| Orphan page | Consistency | `ancestors = []` (page floats outside the tree) | to check |
| Status validity | Validity | `status != 'current'` (draft/trashed leaked in) | none currently |

All of these are one query:

```sql
SELECT source, metadata->>'title' AS title,
       (metadata->>'last_modified')::timestamptz < now() - interval '2 years' AS stale,
       metadata->>'last_modified_by' LIKE '%(Deleted)%'
         OR metadata->>'last_modified_by' LIKE '%(Unlicensed)%'   AS orphaned_owner,
       jsonb_array_length(metadata->'labels') = 0                 AS unlabeled,
       (metadata->>'version')::int = 1                            AS never_reviewed,
       jsonb_array_length(metadata->'ancestors') = 0              AS orphan_page,
       metadata->>'status' <> 'current'                           AS bad_status
FROM document_metadata;
```

## Case 2 — content-based checks (catch what metadata can't see)

Ordered cheap → expensive:

1. **Stub detection** (Completeness) — total stripped text below ~50 words = a stub page that
   pollutes retrieved context. SQL only: `SELECT source, sum(length(content)) FROM chunks GROUP BY source`.
2. **Markup residue** (Validity) — Confluence macro/table leftovers that `storage_to_text`
   flattened into unreadable run-together lines. Heuristic: high ratio of consecutive very-short lines.
3. **Staleness markers in text** (Timeliness) — regex for `TODO`, `TBD`, `deprecated`, old years,
   names of decommissioned systems.
4. **Near-duplicate detection** (Uniqueness) — pairwise cosine similarity between per-source mean
   embeddings, flag pairs > 0.95. **Free**: embeddings already sit in pgvector, nothing to re-embed.
5. **LLM-as-judge** (Accuracy/Clarity) — local model (qwen2.5 via Ollama, zero cost) scores each
   document for clarity, self-containedness, and apparent currency — same pattern as the Phase 3
   eval harness (faithfulness/relevancy judges).

## Implementation plan

- One script: `scripts/dq_report.py`, printing a per-document table (every check + a weighted
  overall score), mirroring `scripts/route_report.py` / `scripts/sql_eval.py` style.
- **Stage 1** (SQL-only, fast): all of Case 1 + content checks 1 & 4. Covers 6 of 8 DQ dimensions
  with zero model calls.
- **Stage 2** (opt-in): content checks 2, 3, and the LLM judge behind a `--with-llm` flag,
  reusing the thin LLM client wrapper.
- Thresholds (staleness window, stub word count, dup similarity) as CLI flags with the defaults above.

## Wiring back into RAG (later, optional)

- Store the score in `document_metadata` (e.g. `metadata->'dq'`) on each report run.
- Retrieval filter: exclude sources below a score floor (same `WHERE source = ANY(...)` pushdown
  mechanism the governance gate already uses in `retrieval/filter.py`).
- Citation warning: surface staleness in the API/UI next to the source
  ("last modified 2020-03-24").

## Non-goals (for now)

- Cross-document contradiction detection (expensive, needs pairwise LLM calls).
- Auto-fixing content in Confluence itself — this reports, it doesn't write back.
