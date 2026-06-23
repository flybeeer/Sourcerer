# Sourcerer — Phase 10 Blueprint: Data Governance (Cerbos)

> เติม pillar ที่ **applied LLM/RAG + DE** ยังไม่แตะ: **authorization & data governance** —
> ใครถามอะไรได้, เห็น asset ไหนได้, query ที่ LLM generate ออกมาแตะ table/column ที่ "ห้าม" หรือเปล่า
> โดย**ต่อยอดของที่มีอยู่** (`sqlkb/safety.py`, `query_log`, Postgres เดิม) ไม่ใช่แปะเทคโนโลยีเพื่อ tick box

ภาคต่อของ [`hybrid-rag-portfolio-project.md`](hybrid-rag-portfolio-project.md) (Phase 1–6),
Phase 7 (SQL KB) และ [`orchestration-analytics-phase-8-9.md`](orchestration-analytics-phase-8-9.md).
ยึด ethos เดิม: **"ทุก quality claim ต้องมีตัวเลขรองรับ"** และ **"optimize for demonstrable
engineering judgment, not feature count"** — governance ก็ต้องมี **eval** เป็นของตัวเอง (leakage test),
ไม่ใช่ "น่าจะปลอดภัย"

**สถานะ:** PLANNED — design only. ยังไม่เขียนโค้ด

---

## ทำไมต้องทำ phase นี้

Sourcerer ตอบคำถามจาก corpus + database ได้แม่นและมี citation แล้ว แต่ **ทุกคนเห็นทุกอย่าง** —
ไม่มีแนวคิดของ *principal* (ใครถาม) และไม่มี *policy* (ใครเข้าถึง asset ไหนได้) เลย ในระบบจริง
นี่คือ blocker อันดับหนึ่งของการเอา RAG ไปใช้กับข้อมูลภายใน: เอกสาร HR/เงินเดือน/สัญญาลูกค้า
ไม่ควรหลุดผ่านคำตอบของ LLM

| Pillar | สถานะเดิม | Phase ที่เติม |
|--------|-----------|--------------|
| Retrieval / generation quality | Phase 1–6 | — |
| Database-as-source / Text-to-SQL | Phase 7 | — |
| Orchestration / analytics | Phase 8–9 (planned) | — |
| **AuthZ / data governance** | **ไม่มี — ทุกคนเห็นทุกอย่าง** | **Phase 10 — Cerbos** |

**จุดสำคัญ — กฎเหล็กของ RAG governance:** authorization ต้องเกิดที่ **retrieval / execution time
ไม่ใช่ generation time** ถ้ากรองหลัง LLM ตอบ โมเดลได้อ่านเนื้อหาที่ห้ามไปแล้ว และรั่วผ่านคำตอบได้
แม้ไม่มี citation chunk/row/community ที่ห้าม **ต้องไม่เข้า candidate set ตั้งแต่แรก**

---

## สถาปัตยกรรมอ้างอิง (reference) → map ลง Sourcerer

ทรงที่อิงคือ governance gate 3 ชั้น (discover → scoped context → validate → execute):

```
AI Agent / LLM
   │ (1) discover — เห็นเฉพาะ asset ที่มีสิทธิ์  ──────────▶  MCP resources (gated by principal)
   ▼
Catalog (metadata)         ◀─ RBAC+ABAC, PII tags, glossary, metric defs
   │ (2) scoped context — column tags, ownership, lineage
   ▼
Governance Gate            ◀─ validate "SQL ที่ generate ออกมา"
 (SQLGlot → Cerbos PDP)         parse → tables/cols → CheckResources → deny / mask (+ RLS backstop)
   │ (3) execute under read-only service account
   ▼
Data Plane                 ◀─ defense-in-depth: Postgres row-level security
```

**Map ของแต่ละชั้นเข้ากับโค้ดจริง:**

| ชั้นใน reference | เทียบเท่าใน Sourcerer | สถานะ |
|------------------|----------------------|-------|
| Discover via MCP (allowed assets) | MCP server expose sources + SQL tables/columns, กรองด้วย principal | ใหม่ |
| Catalog (OpenMetadata: RBAC+ABAC, PII tags) | `asset_catalog` ตารางใน Postgres เดิม (default) → OpenMetadata = production swap | ใหม่ |
| **Governance Gate (SQLGlot + Cerbos)** | SQLGlot parse SQL → tables/cols → Cerbos PDP ตัดสิน; `sqlkb/safety.py` regex ยังทำ read-only guard ต่อ | อัปเกรดของเดิม |
| Execute w/ service account | `sqlkb/connection.connect_ro` (read-only อยู่แล้ว) + per-principal scope | ต่อยอด |
| Data plane RLS (Postgres/Unity/S3) | Postgres **row-level security** บน `chunks` | ใหม่ |

> Delta Lake / Unity Catalog / Ranger / S3 ในภาพต้นทาง = production data plane ที่ Sourcerer
> ไม่มี (ไม่มี lakehouse) — เราใช้ Postgres + SQLite/DuckDB เป็น data plane จริง แล้ว map
> แนวคิด RLS/IAM-scoping ลงตรงนั้นแทน

---

## Cerbos input model — และทำไม "parser" ถึงโผล่เข้ามา

เข้าใจจุดนี้ก่อน แล้วเหตุผลของทุกการตัดสินใจด้านล่างจะกระจ่าง Cerbos เป็น **decision function**
บริสุทธิ์ `decide(principal, resource, action) → ALLOW/DENY` — มันรู้จักแค่ object ที่มี attribute
**ไม่รู้จัก SQL เลย** input คือ 3 ก้อน:

```jsonc
// CheckResources request
{
  "principal": { "id": "alice", "roles": ["analyst"],
                 "attr": { "clearance": "internal", "teams": ["ops"], "region": "APAC" } },
  "resources": [
    { "kind": "table",  "id": "employees",        "attr": { "classification": "confidential", "owner_team": "hr" } },
    { "kind": "column", "id": "employees.salary", "attr": { "pii": true } }
  ],
  "actions": ["read"]
}
// → { "employees": {"read":"ALLOW"}, "employees.salary": {"read":"DENY"} }
```

Cerbos เทียบ `principal.attr` กับ `resource.attr` ตาม policy YAML แล้วคืน allow/deny รายคู่
(resource × action) — เท่านั้น

**ทำไม parse SQL ถึงจำเป็น (ในเส้นทาง post-gen):** ข้อมูลว่า query แตะ table/column ไหน
ถูก "ขัง" อยู่ในสตริง SQL — Cerbos ดึงเองไม่ได้ ต้องแปลง SQL → resource list ก่อนถึงจะ
*ถามคำถามที่ถูก* ได้:

```
"SELECT salary FROM employees"
   │  parse  ← ขั้นนี้ที่ต้องใช้ parser (SQLGlot ฯลฯ)
   ▼
tables=[employees], columns=[employees.salary]
   │  เติม attr จาก catalog → ประกอบเป็น resources[]
   ▼
ส่งเข้า Cerbos
```

> **parser ไม่ใช่ส่วนหนึ่งของ Cerbos** — มันคือสะพานแปลง *SQL → input ของ Cerbos* Cerbos ทำงาน
> หลังสะพานนี้เสมอ ดังนั้น granularity ที่ละเอียด (table/column) จาก SQL ที่ generate มา **ผูกกับการ
> ต้องมี parser อย่างเลี่ยงไม่ได้**

**แนวทางที่เราเลือก:** ใช้ **SQLGlot เป็น parser** แกะ table/column จาก SQL ที่ generate ออกมา →
ประกอบเป็น resources[] → ถาม Cerbos (post-gen, fine) เพราะนี่คือทางเดียวที่ได้ **hard guarantee**
ระดับ column/row บน SQL จริงที่จะรัน เสริมด้วย pre-filter schema (กัน UX/token) และ RLS (backstop)
เป็น defense-in-depth

| ถาม Cerbos ด้วย resource แบบไหน | ต้อง parse SQL ก่อนไหม | granularity | บทบาทในดีไซน์ |
|---|---|---|---|
| `table`/`column` ที่ SQLGlot แกะจาก SQL (post-gen, fine) | ✅ | ละเอียด, **hard** | **หลัก** |
| `table`/`column` จาก catalog (pre-gen, restrict-schema) | ❌ | ละเอียด, soft (prevention) | เสริม (UX/token) |
| `sql_source` ทั้งก้อน (post-gen, coarse) | ❌ | หยาบ (all-or-nothing/DB) | — |

---

## Governance gate = SQLGlot (parser) + Cerbos (PDP) — คนละหน้าที่ เสริมกัน

ภาพต้นทางเขียน "Python+SQLGlot **or** Cerbos" — แต่จริง ๆ มันคนละหน้าที่และต้องใช้คู่กัน (ตาม Cerbos
input model ด้านบน: Cerbos ต้องการ resource ที่แกะมาแล้ว แต่มันแกะ SQL เองไม่ได้):

- **SQLGlot = ตา + มือ ของ gate** — parse SQL ที่ LLM generate เป็น AST, ดึงว่า *แตะ table/column ไหน*
  (เชื่อถือได้กว่า regex ของ `safe_select` — `created_at` ไม่ติด "create" อีก) และ **rewrite** ได้
  (mask column PII เป็น `'***'`, splice row-filter `WHERE`)
- **Cerbos = สมอง ของ gate (PDP)** — รับ `principal` + `resources[]` (ที่ SQLGlot แกะ + เติม attr จาก
  catalog) + `action` → คืน ALLOW/DENY รายตัว หรือผ่าน `PlanResources` คืน condition (row-filter)

**Flow ของ path SQL:**

```
LLM gen SQL ─▶ safe_select (read-only guard, ของเดิม)
            ─▶ SQLGlot parse ─▶ tables/cols ─▶ เติม attr จาก catalog ─▶ resources[]
            ─▶ Cerbos CheckResources(principal, resources, "read")
                 ├─ table ต้องห้าม DENY      → reject ทั้ง query → "I don't know"
                 ├─ PII column DENY          → SQLGlot mask column นั้น (ไม่ทิ้งทั้ง query)
                 └─ PlanResources row-filter → SQLGlot splice เข้า WHERE
            ─▶ run ใต้ read-only conn ─▶ SQL (ที่ rewrite แล้ว) + rows = citation
```

- path เอกสารใช้แค่ Cerbos (ไม่ต้อง parse): `PlanResources(principal,"read","document")` → `WHERE
  source = ANY(:allowed)` push ลง pgvector/BM25 โดยตรง — **"push the predicate down"** เดียวกับ DuckDB
  aggregation pushdown / schema retrieval ของ Phase 7
- **defense-in-depth (เลือกเปิดได้):** (1) pre-filter schema เข้า prompt = prevention + ประหยัด token,
  (2) SQLGlot+Cerbos = hard gate บน SQL จริง, (3) Postgres **RLS** = backstop ที่ data plane เผื่อ gate
  พลาด — สามชั้นทับกัน ไม่พึ่งชั้นเดียว

---

## ครอบคลุมทั้งสาม retrieval path

หลักเดียวกัน (authorize ก่อน candidate เข้าถึง LLM) แต่ "query" ของแต่ละ path ต่างกัน:

### A. Document RAG (vector + BM25 + rerank)
- **Gate** = `PlanResources(principal, "read", "document")` → `WHERE` filter
- ฉีดลง `retrieval/vector.py` (`<=>` query) และ `retrieval/bm25.py` (FTS) **ก่อน** fusion/rerank
- asset = `source`; attributes (classification / owner_team / pii) มาจาก `asset_catalog`

### B. SQL KB / Text-to-SQL (Governance Gate — SQLGlot + Cerbos)
1. (เสริม) **ก่อน generate:** Cerbos `CheckResources` เหนือ catalog → ส่งเฉพาะ schema ที่อนุญาตเข้า
   prompt (ต่อยอด `sqlkb/schema_retrieval.py`) — prevention + ประหยัด token
2. LLM generate SQL → `safe_select` เดิม (read-only guard)
3. **SQLGlot parse** → ดึง tables + columns ที่ถูกอ้าง → เติม attr จาก catalog → `resources[]`
4. **Cerbos `CheckResources`:** principal เข้าถึงทุก table/column ที่แตะได้ไหม
   - table ทั้งใบ DENY → reject → "I don't know"
   - PII column DENY → SQLGlot **mask** (`'***'`/hash) แทนการทิ้งทั้ง query
   - `PlanResources` คืน row-filter (เช่น `region = :principal.region`) → SQLGlot splice เข้า `WHERE`
5. รันใต้ read-only connection; **Postgres RLS** เป็น backstop ที่ data plane เผื่อ gate พลาด
6. ผลลัพธ์ + SQL (ที่ rewrite แล้ว) = citation — โปร่งใส ตรวจสอบได้

> `safe_select` regex ยังอยู่เป็น read-only guard ชั้นแรก; SQLGlot+Cerbos เพิ่มชั้น **policy แบบ
> AST-based** (table/column-aware) ทับเข้าไป — ไม่แทนที่ของเดิม

### C. GraphRAG (community map-reduce)
- community summary trace กลับไปยัง **source files** ได้อยู่แล้ว (Phase 6)
- ก่อน map: ตัด community ที่ source ไม่ผ่าน ACL ของ principal ออก → reduce เห็นเฉพาะที่อนุญาต

---

## Access model: RBAC + ABAC + PII tags

```yaml
# resource: document / table / column  (attributes จาก asset_catalog)
asset:
  classification: confidential        # public < internal < confidential < restricted
  owner_team: finance
  pii_tags: [email, salary]

# principal
principal:
  roles: [analyst]
  clearance: internal
  teams: [ops]
  region: APAC

# Cerbos resource policy (สรุปกฎ)
#   ALLOW read  if principal.clearance >= asset.classification
#               OR asset.owner_team in principal.teams
#   MASK  column if column.pii_tags ∩ {restricted-for(principal.roles)}
#   FILTER rows  by region == principal.region   (PlanResources → WHERE)
```

- **RBAC** = roles → ความสามารถพื้นฐาน; **ABAC** = clearance/team/region เทียบ attribute ของ asset;
  **PII tags** = ขับ column masking โดยเฉพาะ ครบตามที่ภาพ reference ระบุ

---

## องค์ประกอบที่ต้องสร้าง

| ส่วน | รายละเอียด | ใหม่/ต่อยอด |
|------|-----------|-------------|
| `asset_catalog` (Postgres) | source/table/column → classification, pii_tags[], owner_team, glossary | ใหม่ |
| Principal บน request | `QueryRequest` + `X-Principal` header / JWT claims (stub IdP) | ต่อยอด schema |
| `governance/` module | `parse()` (SQLGlot → tables/cols), `check()` (Cerbos CheckResources), `plan()` (PlanResources→WHERE), `mask()` (SQLGlot rewrite PII col) | ใหม่ |
| SQLGlot | parse/inspect/rewrite SQL ที่ generate; `[governance]` extra | ใหม่ dep |
| Cerbos policies | `policies/*.yaml` (resource: document/table/column) | ใหม่ |
| Cerbos PDP | sidecar ใน `docker-compose` (stateless, gRPC/HTTP), mount `policies/` | ใหม่ infra เดียว |
| MCP server | expose assets ที่อนุญาต ให้ agent discover (ชั้น 1) | ใหม่ (optional depth) |
| Postgres RLS | row-level security บน `chunks` ตาม catalog (defense-in-depth) | ใหม่ |
| Audit | Cerbos decision logs + เพิ่ม principal/denied-assets ลง `query_log` trace | ต่อยอด |

**Config ใหม่ (mirror รูปแบบ `LOCAL_BACKEND` / `SQL_KB_BACKEND`):**
`GOVERNANCE_ENABLED`, `CERBOS_ENDPOINT`, `CATALOG_BACKEND` (`local` Postgres | `openmetadata`),
`PRINCIPAL_HEADER`

---

## Eval — governance ต้องมีตัวเลข (กฎเหล็ก: never skip eval)

`eval/governance_eval_set.jsonl` = matrix ของ `(principal, query, expected_visibility)`:

1. **Leakage (headline = ต้องเป็น 0)** — query ที่คำตอบอยู่ใน asset ที่ principal **ห้ามเห็น** →
   ระบบต้องตอบ "I don't know" / **ไม่ cite** asset นั้น และคำตอบต้องไม่มีเนื้อหาที่หลุดมา
2. **Over-restriction (false denial)** — principal ที่ **มีสิทธิ์** ต้องยังได้คำตอบถูก →
   วัด `recall@k` แยกราย principal เทียบ baseline ไม่มี governance (governance ไม่ควรทำ recall ของ
   คนที่มีสิทธิ์พัง)
3. **Gate metrics** — % SQL ที่ถูก reject/rewrite, จำนวน PII column ที่ mask, latency ที่
   SQLGlot+Cerbos เพิ่ม (p95), ความครบของ audit log

รายงานคู่กับ Phase 7 `sql_eval.py` style: leakage rate, false-denial rate, gate-decision latency

---

## สิ่งที่จงใจ "ไม่ทำ" (scope guard)

- **OpenMetadata cluster เต็มรูปแบบ** (Java + Elasticsearch) — หนักเกินสำหรับ portfolio; ใช้
  `asset_catalog` ใน Postgres เดิมเป็น default, เปิด seam `CATALOG_BACKEND=openmetadata` ไว้เป็น
  production swap (รูปแบบเดียวกับ Ollama→vLLM, SQLite→DuckDB, JSON→Postgres graph store)
- **Unity Catalog / Ranger / S3 / lakehouse** — ไม่มี data plane นั้น; map แนวคิด RLS ลง Postgres แทน
- **IdP / OAuth จริง** — principal มาจาก header/JWT-claim stub; การ integrate IdP คือ production detail
- **Policy admin UI** — policy เป็น YAML ใน git (reviewable, versioned) พอแล้ว

---

## Build order (sub-phases — ทำตามลำดับ, end-to-end ก่อน improve)

1. **10a — Catalog + principal:** `asset_catalog`, ingest tagging (`--classification/--pii`),
   principal บน request. ยังไม่ enforce — แค่มี metadata + identity
2. **10b — Document RAG gate:** Cerbos PDP sidecar + `PlanResources`→`WHERE` ฉีดลง vector/BM25 +
   **leakage eval** บน path เอกสาร (เห็นเลขก่อน)
3. **10c — SQL KB gate:** SQLGlot parse → tables/cols → Cerbos CheckResources → PII column masking +
   row-filter splice; `safe_select` regex อยู่ต่อเป็น read-only guard ชั้นแรก (+ pre-filter schema เสริม)
4. **10d — GraphRAG gate + RLS + MCP discovery + audit:** ปิด path ที่เหลือ, Postgres RLS เป็น
   defense-in-depth, MCP scoped discovery, principal+denials ลง `query_log`

---

## Open decisions (ข้อเสนอแนะ default — review ก่อนลงมือ)

| ประเด็น | ข้อเสนอแนะ (default) | ทางเลือก |
|--------|---------------------|----------|
| Catalog backend | `local` Postgres `asset_catalog` ก่อน | OpenMetadata เป็น swap ทีหลัง |
| Gate engine | **SQLGlot (parser) + Cerbos (PDP)** — fine-grained, hard guarantee บน SQL จริง | Cerbos อย่างเดียว + RLS (ไม่ parse, granularity หยาบกว่า) |
| MCP discovery | ทำใน 10d (โชว์ scoped discovery, เข้าธีม MCP) | ตัดออกถ้าจะ lean |
| Principal source | `X-Principal` header → lookup (stub) | JWT claims |
```
