# Phase 10 — Data governance: demo & test cases

Hands-on test cases for the governance gate (Phase 10), runnable from the web UI
or via `curl`. Every answer is filtered by **who is asking** (the principal): the
gate authorizes at retrieval/execution time, so forbidden documents/rows never
reach the model.

> All governance is **off by default** (`GOVERNANCE_ENABLED=false`). These cases
> assume it's on — see Setup.

## Principals (stub IdP)

Seeded by `python scripts/catalog.py seed-demo` (or `build_governance_demo.py`):

| Principal | roles | clearance | teams |
|-----------|-------|-----------|-------|
| `anonymous` | — | public | — |
| `alice` | analyst | internal | ops |
| `bob` | hr | confidential | hr |
| `carol` | admin | restricted | hr |

Access rule (`governance/policy.py`): **read allowed if** `admin` role **OR**
`clearance ≥ classification` **OR** `owner_team ∈ principal.teams`.
Lattice: `public < internal < confidential < restricted`.

---

## Setup

### 1. Start the stack
```bash
docker compose up -d db          # Postgres + pgvector (and cerbos, if using GOVERNANCE_PDP=cerbos)
# Ollama must be running with bge-m3 (embeddings) + a small chat model.
```

### 2. Seed the demo document corpus (real embeddings) + tags
Create four small files and ingest them (the `--classification` flags tag them in
the asset catalog as they're ingested):
```bash
mkdir -p data/govdemo
printf '# Office FAQ\nThe office opens at 9am and closes at 6pm on weekdays.\n'            > data/govdemo/faq.md
printf '# Security Policy\nPasswords must be rotated every 90 days and use MFA.\n'         > data/govdemo/security.md
printf '# Employee Handbook (HR)\nEmployee salary bands are defined by HR; a senior\n engineer is band 5.\n' > data/govdemo/handbook.md
printf '# Board Minutes\nThe board approved the confidential acquisition of a competitor.\n' > data/govdemo/board.md

python scripts/ingest.py data/govdemo                      # real bge-m3 embeddings
python scripts/catalog.py set-asset --kind document --id faq.md       --classification public
python scripts/catalog.py set-asset --kind document --id security.md  --classification internal     --owner-team it
python scripts/catalog.py set-asset --kind document --id handbook.md  --classification confidential  --owner-team hr
python scripts/catalog.py set-asset --kind document --id board.md      --classification restricted
python scripts/catalog.py seed-demo                        # principals
```

### 3. Seed the SQL knowledge base (for the SQL cases)
```bash
python - <<'PY'
import sqlite3
c = sqlite3.connect("data/govkb.sqlite")
c.executescript("""
CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, team TEXT, salary INTEGER);
INSERT INTO employees (name,team,salary) VALUES
  ('Anan','eng',95000),('Beam','eng',120000),('Cha','hr',80000),('Dao','sales',70000),('Ek','sales',65000);
CREATE TABLE board (id INTEGER PRIMARY KEY, topic TEXT, decision TEXT);
INSERT INTO board (topic,decision) VALUES ('Acquisition','approved'),('FY budget','approved');
""")
c.commit()
PY
python scripts/catalog.py set-asset --kind table  --id employees        --classification public
python scripts/catalog.py set-asset --kind column --id employees.salary --classification confidential --owner-team hr --pii salary
python scripts/catalog.py set-asset --kind table  --id board            --classification restricted
```

### 4. Run the API with governance on
```bash
GOVERNANCE_ENABLED=true LOCAL_MODEL=qwen2.5:7b RERANKER_TYPE=none \
  SQL_KB_ENABLED=true SQL_KB_PATH=data/govkb.sqlite \
  uvicorn sourcerer.api.main:app --host 127.0.0.1 --port 8000
```
Open <http://localhost:8000>. The **Principal** dropdown sets the `X-Principal`
header; **Route** picks the path.

---

## Document cases (RAG path)

**UI:** Route = `Local LLM` (forces the document/hybrid path), then pick the Principal.
Expected ✅ = answered with the right citation; ❌ = "I don't know" (source filtered out).

### Case A — public doc
> **What are the office opening hours?**  → `faq.md` (public)

| anonymous | alice | bob | carol |
|:---:|:---:|:---:|:---:|
| ✅ | ✅ | ✅ | ✅ |

### Case B — internal doc
> **What is the password rotation policy?**  → `security.md` (internal)

| anonymous | alice | bob | carol |
|:---:|:---:|:---:|:---:|
| ❌ | ✅ | ✅ | ✅ |

### Case C — confidential doc
> **What are the employee salary bands?**  → `handbook.md` (confidential / hr)

| anonymous | alice | bob | carol |
|:---:|:---:|:---:|:---:|
| ❌ | ❌ | ✅ | ✅ |

### Case D — restricted doc
> **What acquisition did the board approve?**  → `board.md` (restricted)

| anonymous | alice | bob | carol |
|:---:|:---:|:---:|:---:|
| ❌ | ❌ | ❌ | ✅ |

The `router_reason` under each answer shows `governance: N/M source(s) visible to <principal>`.

---

## SQL knowledge-base cases (Text-to-SQL path)

**UI:** Route = `Text-to-SQL (database)`, then pick the Principal. The citation shows
the **SQL that actually ran** — masked columns appear as `'***'`.

### Case 1 — PII column masking
> **What is the average salary of employees?**  (`employees` public, `salary` confidential/hr PII)

| Principal | SQL that ran | Answer |
|-----------|--------------|--------|
| `alice` (can't read salary) | `SELECT '***' AS "***" FROM employees` | can't compute — masked |
| `bob` (hr) | `SELECT AVG(salary) FROM employees` | **86,000** |

### Case 2 — forbidden-table rejection
> **How many board decisions were approved?**  (`board` restricted)

| Principal | Result |
|-----------|--------|
| `bob` | "I don't know" — query rejected (no SQL runs); audit logs `no safe SQL produced` |
| `carol` (admin) | `SELECT COUNT(*) FROM board WHERE decision='approved'` → "Two…" |

### Case 3 — public / no-PII (control)
> **How many employees are there in total?**  (public table, no PII column)

| Principal | Result |
|-----------|--------|
| any (e.g. `anonymous`) | `SELECT COUNT(*) FROM employees` → "5 employees" |

### More to try
| Prompt | Toggle | What to watch |
|--------|--------|---------------|
| `List employee names and their salaries` | alice ↔ bob | alice: salary `***`, name visible; bob: both |
| `Show all board meeting topics` | bob ↔ carol | bob rejected; carol answers |
| `What is the total salary cost?` | alice ↔ bob | alice masked; bob real number |

---

## Verifying enforcement

- **Citations** change with the principal (forbidden sources never appear).
- **`router_reason`** records `governance: N/M source(s) visible` (RAG) or
  `governance gate as <principal>` (SQL).
- **Audit:** `query_log.principal` + `query_log.denied_assets` record who asked and
  how many assets were hidden:
  ```sql
  SELECT principal, denied_assets, num_citations, left(router_reason, 80)
  FROM query_log ORDER BY id DESC LIMIT 10;
  ```
- **RLS backstop** (`scripts/rls_demo.py`, needs `setup_rls.py`): proves the database
  blocks forbidden rows even with the app-level filter disabled.

## curl equivalents

```bash
# Document path
curl -s localhost:8000/query -H 'content-type: application/json' -H 'X-Principal: bob' \
  -d '{"query":"What are the employee salary bands?","route_override":"local"}'

# SQL path
curl -s localhost:8000/query -H 'content-type: application/json' -H 'X-Principal: alice' \
  -d '{"query":"What is the average salary of employees?","route_override":"sql"}'
```

## Related
- Design: [`governance-phase-10.md`](governance-phase-10.md)
- Status & build notes: `CLAUDE.md` (Phase 10)
- Evals: `scripts/governance_eval.py` (doc leakage), `scripts/sql_gate_eval.py` (SQL gate), `scripts/rls_demo.py` (RLS backstop)
