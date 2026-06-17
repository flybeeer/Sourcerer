# Sourcerer — Phase 8–9 Blueprint: Orchestration + Analytics

> ต่อยอด Sourcerer ให้ครอบคลุม **classic Data Engineering pillars** ที่ตัวโปรเจกต์เดิม (LLM/RAG)
> ยังไม่แตะ — **orchestration** และ **warehouse/analytics modeling** — โดยใช้ของที่มีอยู่แล้วเป็นฐาน
> ไม่ใช่แปะเทคโนโลยีเพื่อ tick box

ฉบับนี้เป็นภาคต่อของ [`hybrid-rag-portfolio-project.md`](hybrid-rag-portfolio-project.md) (Phase 1–6)
และ Phase 7 (SQL knowledge base) อ่าน blueprint หลักก่อนเพื่อเข้าใจ ethos: **"ทุก quality claim
ต้องมีตัวเลขรองรับ ไม่ใช่ดูแล้วน่าจะดี"** และ **"optimize for demonstrable engineering judgment,
not feature count"** — สอง phase นี้ยึดหลักเดียวกัน

---

## ทำไมต้องทำสอง phase นี้

Sourcerer เดิมแข็งด้าน *applied data + ML engineering* (pipeline design, SQL, data-quality
measurement, cost/observability) แต่ยังขาด pillar คลาสสิกของ DE ที่ recruiter สาย data มองหา:

| Pillar | สถานะเดิม | Phase ที่เติม |
|--------|-----------|--------------|
| Pipeline orchestration / scheduling | มี `scripts/` รันมือทีละตัว | **Phase 8 — Dagster** |
| Warehouse / dimensional modeling | มี `query_log` ดิบ แต่ไม่มี analytics layer | **Phase 9 — dbt** |
| Streaming / distributed (Kafka/Spark) | — | *ไม่ทำ* (ดู "สิ่งที่จงใจไม่ทำ") |

**จุดสำคัญ:** ทั้งสอง phase ไม่ได้สร้างของใหม่จากศูนย์ — มัน **ห่อ (orchestrate)** และ **ต่อยอด
(model)** สิ่งที่ Phase 1–7 ทำไว้แล้ว นี่คือเหตุผลที่ ROI สูงและดูเป็นธรรมชาติ ไม่ใช่ cargo-culting

---

## สถาปัตยกรรมเป้าหมาย (หลัง Phase 8–9)

```
   ┌──────────────── Phase 8: Dagster (orchestration) ────────────────┐
   │                                                                   │
   │   raw docs / DB ─▶ [ingest] ─▶ [graphrag_index?] ─▶ [run_eval]    │
   │        (assets)        │              │                 │          │
   │                        ▼              ▼                 ▼          │
   │                     chunks         graph            eval_report    │
   │                   (pgvector)      (artifacts)        (metrics)     │
   └───────────────────────────────────┬───────────────────────────────┘
                                        │  เขียน metrics + query_log อยู่ใน Postgres เดิม
                                        ▼
   ┌──────────────── Phase 9: dbt (analytics layer) ──────────────────┐
   │                                                                   │
   │   query_log (raw)  ─▶  staging  ─▶  fct_queries (1 row=1 query)   │
   │   eval_report      ─▶  staging  ─▶  fct_eval_runs                 │
   │                                  +  dim_route / dim_model / dim_day│
   │                                          │                         │
   │                                          ▼                         │
   │                            marts: cost_per_day, route_split,       │
   │                                   p95_latency_by_route,            │
   │                                   eval_trend_over_time             │
   └───────────────────────────────────────────────────────────────────┘

   ทั้งหมดอยู่บน Postgres เดียวกับที่ pgvector/BM25/query_log ใช้ — ไม่เพิ่ม infra ใหม่
```

> หลักการเดียวกับ pgvector: **ใช้ Postgres ตัวเดียวให้คุ้ม** — Dagster เก็บ run metadata,
> dbt build marts, ทุกอย่างอยู่ที่เดิม operate ง่าย

---

## Phase 8 — Orchestration (Dagster) ⭐ ทำก่อน

**เป้าหมาย:** เปลี่ยนชุดสคริปต์ที่รันมือ (`ingest.py`, `ingest_sql.py`, `graphrag_index.py`,
`run_eval.py`) ให้เป็น pipeline ที่มี dependency, schedule, retry, และ lineage ที่ตรวจสอบได้

### ทำไม Dagster ไม่ใช่ Airflow

| ประเด็น | Dagster | Airflow |
|---------|---------|---------|
| Mental model | **software-defined assets** (corpus → chunks → index → eval) | task/DAG เน้น operator |
| ความเข้ากับ RAG | ตรง — RAG คือ data assets ที่ derive ต่อกันเป็นชั้น | ต้องคิดเป็น task เอง |
| Local dev | `dagster dev` ขึ้นมาเดียว เบา | ต้องตั้ง scheduler+webserver+db |
| Type/IO | typed inputs/outputs, IO managers | ต้องจัดการ XCom เอง |

> **เหตุผลที่เลือก** (พูดได้ตอนสัมภาษณ์): asset-based model ของ Dagster ทำให้ lineage ของ RAG
> ชัดเจน (เห็นว่า eval report มาจาก index ไหน มาจาก corpus เวอร์ชันไหน) ซึ่งตรงกับธีม
> "answer with sources / measure everything" ของโปรเจกต์ Airflow เหมาะกว่าถ้าเน้น
> task-scheduling ล้วนๆ ในองค์กรใหญ่ — แต่ที่นี่ asset lineage มีค่ากว่า

### สิ่งที่ต้องทำ

- กำหนด **assets**: `raw_corpus → chunks → (graph_index) → eval_report`
  ห่อ logic เดิมใน `scripts/` เป็น asset (เรียก function เดิม ไม่ rewrite)
- ใช้ **idempotent ingest ที่มีอยู่แล้ว** (ลบ source เก่าก่อน insert) เป็น re-run safety —
  นี่คือ DE discipline ที่โปรเจกต์มีอยู่แล้ว แค่ทำให้ orchestrator มองเห็น
- **Schedule**: rebuild eval report ทุกวัน / **Sensor**: trigger ingest เมื่อมีไฟล์ใหม่ใน `data/raw`
- **Partitions**: partition ingest ตาม source/วันที่ เพื่อโชว์ backfill
- เก็บ Dagster run metadata ลง Postgres เดิม
- เพิ่ม `[dagster]` extra ใน pyproject + service ใน `docker-compose`

### ส่งมอบ

- `dagster dev` แล้วเห็น **asset graph** ของ Sourcerer ทั้งสาย (lineage จาก corpus → eval)
- การเปลี่ยน corpus → trigger → ได้ eval report ใหม่อัตโนมัติ โดยไม่รันสคริปต์มือ
- README section + ภาพ asset graph

### Definition of Done

- [ ] ทุก `scripts/` entrypoint หลักมี asset/job ห่อ และรันผ่าน Dagster ได้
- [ ] มี schedule อย่างน้อย 1 + sensor อย่างน้อย 1 ที่ทำงานจริง
- [ ] re-run ซ้ำได้ปลอดภัย (idempotent) — พิสูจน์ด้วยการรันซ้ำแล้วข้อมูลไม่ซ้ำ
- [ ] บันทึก phase ใน CLAUDE.md "Current status"

---

## Phase 9 — Analytics layer (dbt)

**เป้าหมาย:** เปลี่ยน `query_log` (operational table) ให้เป็น **dimensional model** ที่ตอบคำถามเชิง
วิเคราะห์ได้ — โชว์ SQL transformation, star schema, และ data testing แบบ warehouse จริง

### ทำไมคุ้ม: ข้อมูลมีอยู่แล้ว

`query_log` บันทึก route, model, tokens, latency, cost ต่อ query อยู่แล้ว — นี่คือ **fact table ดิบ**
ที่รอ model เท่านั้น Phase 4 ทำ `route_report.py` แบบ ad-hoc ไว้แล้ว Phase 9 คือการยกมันขึ้นเป็น
analytics layer ที่ถูกต้องตามหลัก

### โมเดล (star schema)

```
staging (1:1 กับ source, ทำความสะอาด/cast)
  stg_query_log, stg_eval_runs
        │
        ▼
facts (grain ชัดเจน)
  fct_queries     — 1 row = 1 query   (tokens, latency_ms, cost_usd, answered, guardrail)
  fct_eval_runs   — 1 row = 1 metric ต่อ config ต่อ run
        │
dimensions
  dim_route (local/api/sql/graph) · dim_model · dim_date · dim_guardrail
        │
        ▼
marts (ตอบคำถามธุรกิจ)
  mart_cost_daily          — cost/วัน, % saved vs API-only
  mart_route_split         — สัดส่วน route ตามเวลา
  mart_latency_by_route    — p50/p95 latency แยก route
  mart_eval_trend          — faithfulness/MRR เปลี่ยนตามเวลา/commit
```

### สิ่งที่ต้องทำ

- ตั้ง dbt project (`dbt-postgres`) ชี้ Postgres เดิม
- เขียน staging → facts → dimensions → marts ตามด้านบน
- **dbt tests**: `not_null`, `unique`, `accepted_values` (route ต้องเป็นค่าที่รู้จัก),
  relationship tests ระหว่าง fact/dim — นี่คือ **data-quality discipline** ที่ต่อยอดจาก eval harness
- ใช้ **incremental model** กับ `fct_queries` (append เฉพาะ query ใหม่) เพื่อโชว์ความเข้าใจ
  incremental processing
- `dbt docs generate` → มี data catalog + lineage graph
- (เชื่อมกับ Phase 8) ให้ Dagster รัน `dbt build` เป็น asset ปลายสาย — orchestration + transformation
  ทำงานร่วมกัน

### ส่งมอบ

- ตาราง mart ที่ query ได้จริง เช่น "cost/วัน 30 วันล่าสุด แยก route" และ "MRR เทรนด์ข้าม commit"
- `dbt docs` lineage graph
- README section + ตัวอย่าง query

### Definition of Done

- [ ] staging/facts/dims/marts ครบและ `dbt build` เขียว
- [ ] มี dbt tests อย่างน้อย 1 ชุดต่อ model หลัก และผ่านทั้งหมด
- [ ] `fct_queries` เป็น incremental และพิสูจน์ว่า append ไม่ rebuild ทั้งตาราง
- [ ] Dagster เรียก `dbt build` เป็น asset ได้ (ถ้าทำ Phase 8 แล้ว)
- [ ] บันทึก phase ใน CLAUDE.md "Current status"

---

## ตัวเลขที่ต้องวัด (ยึด ethos เดิม: prove with numbers)

สองอย่างนี้ไม่ใช่ feature เฉยๆ — ต้องมีตัวเลขโชว์เหมือนทุก phase:

| Phase | วัดอะไร | ทำไม |
|-------|---------|------|
| 8 | เวลา pipeline end-to-end, จำนวน manual step ที่หายไป (เช่น 4 คำสั่งมือ → 0), retry/recovery ที่เกิดจริง | โชว์ว่า orchestration ลดงานมือและเพิ่มความน่าเชื่อถือ |
| 9 | จำนวน dbt tests ที่ผ่าน, freshness ของ mart, ตัวอย่าง insight ที่ ad-hoc ทำไม่ได้ (เช่น p95 latency by route over time) | โชว์ data-quality + analytics value |

---

## สิ่งที่จงใจ "ไม่ทำ" (และเหตุผล — นี่คือ senior signal)

- **Streaming (Kafka/Flink)** — corpus เล็กตาม ethos ของโปรเจกต์ (ดู "กับดัก" ใน blueprint หลัก)
  ทำ streaming บน 12 docs = ของเล่น **การอธิบายว่า "ไม่ใช้เพราะ volume ไม่ถึง batch ก็พอ" คือ
  engineering judgment ที่ดีกว่าการฝืนใส่** ถ้าจะโชว์ pattern จริง ค่อยทำ event-driven ingest
  (drop ไฟล์ → event → consumer ingest) เป็น stretch แยก พร้อมระบุชัดว่าเป็น demo
- **Spark / distributed** — เหตุผลเดียวกัน ทำต่อเมื่อโหลด public dataset ใหญ่จริงมา ingest เท่านั้น
  มิฉะนั้น DuckDB (Phase 7) ครอบ analytics-scale ได้พอแล้ว
- **BI tool เต็มรูป (Superset/Metabase)** — optional มาก marts จาก dbt query ตรงก็พอโชว์ value แล้ว
  เพิ่ม dashboard ได้ถ้าเหลือเวลา

---

## ลำดับลงมือ

1. **Phase 8 ก่อน** — ROI สูงสุด, เติม pillar ที่ขาดชัดสุด, และ Phase 9 จะได้เกาะ Dagster รัน dbt
2. **Phase 9 ตาม** — เกาะ `query_log` ที่มีอยู่ ทำ staging → facts/dims → marts → tests
3. วัดผลตามตารางด้านบน + อัปเดต README และ CLAUDE.md ทุกครั้งที่ phase จบ

> ทำ **8 + 9 ให้ลึกและ measure ได้** มีค่ากว่าใส่ครบทุก buzzword แบบผิวๆ — ตรงกับหัวใจของ Sourcerer
> ตั้งแต่ Phase 1: เริ่มเล็ก ทำให้เดินได้ แล้วทำให้ดี และวัดผลทุกขั้น
