# Sourcerer — Claude Code Prompt Playbook

ชุด prompt สำหรับสั่ง Claude Code ทีละ phase ก็อปวางได้เลย ตัว prompt เป็นภาษาอังกฤษให้เข้าชุดกับ
`CLAUDE.md` ส่วนคำอธิบายภาษาไทยคือบริบทให้คุณเข้าใจว่าแต่ละอันทำอะไร

## วิธีใช้ playbook นี้

- ทำ **ทีละ phase** อย่าเร่งข้าม ให้แต่ละ phase รันได้จริงก่อนค่อยไปต่อ
- ก่อนเริ่ม phase ใหม่ พิมพ์ `/clear` ใน Claude Code เพื่อล้าง context window
  (CLAUDE.md จะถูกโหลดกลับมาอัตโนมัติ จึงไม่เสียบริบทโปรเจกต์)
- หลังจบแต่ละ phase บอกให้มันอัปเดต checklist "Current status" ใน CLAUDE.md
- ถ้ามันเริ่มทำเกินขอบเขต phase ให้หยุดแล้วเตือนว่า "stay within Phase N scope"

---

## Phase 0 — Kickoff (รันครั้งเดียวตอนเริ่ม)

> Read `CLAUDE.md` and `docs/hybrid-rag-portfolio-project-EN.md` fully before doing anything.
> Then propose the repository structure for the whole project (folders for ingestion, retrieval,
> routing, generation, eval, and the FastAPI app), plus a `.gitignore` (ensure `.env` is ignored)
> and a `README.md` skeleton. Do not write feature code yet — just scaffold the structure and show
> me the proposed tree for approval first.

**ทำอะไร:** ให้มันอ่านบริบททั้งหมดก่อน แล้ววางโครงโฟลเดอร์ + .gitignore + โครง README โดยยังไม่เขียน
โค้ดฟีเจอร์ และให้ขออนุมัติโครงสร้างก่อน เป็นการตั้งหลักให้ดี

---

## Phase 1 — MVP RAG

> Implement Phase 1 from the blueprint. Set up a `docker-compose.yml` with Ollama and Postgres
> (with the pgvector extension). Then build, in this order:
> 1. An ingestion pipeline: load documents (start with PDF + Markdown), chunk them, embed with the
>    local embedding model, and store vectors in pgvector.
> 2. Vector-only retrieval over pgvector.
> 3. A generation step that passes retrieved context to the local model and returns an answer
>    **with citations** to the source chunks.
> 4. A minimal FastAPI endpoint (`POST /query`) wrapping the above.
>
> Read all config from `.env`. Keep retrieval, generation, and the API as separate modules. Stop at
> a working end-to-end MVP — do not add hybrid retrieval or reranking yet. Show me how to run it.

**ทำอะไร:** ปั้นให้ทะลุปลายทางก่อน ถามแล้วได้คำตอบพร้อม citation อย่าเพิ่งสนใจคุณภาพ retrieval
ย้ำให้มันไม่ล้ำไปทำ hybrid

---

## Phase 2 — Hybrid Retrieval

> Implement Phase 2. Extend retrieval into a hybrid pipeline:
> 1. Add BM25 keyword search (via Postgres full-text search) running in parallel with vector search.
> 2. Fuse the two result sets with reciprocal rank fusion (RRF), using `RRF_K` from `.env`.
> 3. Add a reranker stage (respect `RERANKER_TYPE` in `.env`) that re-scores the fused candidates
>    down to `TOP_K_FINAL`.
> 4. Make the chunking strategy swappable (fixed-size vs semantic) via config, so we can compare
>    them later.
>
> Keep the Phase 1 vector-only path available behind a flag so we can benchmark against it in
> Phase 3. Don't build the eval harness yet.

**ทำอะไร:** ยกคุณภาพ retrieval ด้วย BM25 + RRF + reranker และทำ chunking ให้สลับได้ สำคัญคือให้คงเส้นทาง
vector-only ไว้เทียบ benchmark ใน Phase 3

---

## Phase 3 — Evaluation Harness ⭐

> Implement Phase 3 — this is the most important phase. Build an evaluation harness:
> 1. Define an eval set format (JSONL) of `{question, expected_answer, relevant_doc_ids}` and create
>    a small starter set under `./eval/` (~10 examples) I can expand.
> 2. Implement retrieval metrics: recall@k, MRR, and hit rate.
> 3. Implement generation metrics: faithfulness and answer relevancy using an LLM-as-judge
>    (use `EVAL_JUDGE_MODEL`).
> 4. Write a runner that evaluates a given retrieval config and prints a results table.
> 5. Run it to compare: vector-only vs hybrid vs hybrid+rerank. Save results to a file.
>
> The goal is a reproducible comparison table proving which configuration is best and why.

**ทำอะไร:** สร้างเครื่องวัดผล + รันเทียบ 3 config นี่คือพระเอกของ portfolio อย่าข้าม ให้ได้ตารางผลที่
รันซ้ำได้

---

## Phase 4 — Hybrid Routing

> Implement Phase 4. Add a router that decides, per query, between the local model and the frontier
> API model:
> 1. Implement the routing logic (start with `ROUTER_STRATEGY=heuristic`): route easy/sensitive/
>    high-volume queries to local, hard/complex-reasoning queries to the API model.
> 2. Instrument every query: which route was taken, tokens, latency, and estimated cost.
> 3. Add a small script that reports the local-vs-API split, cost per query, and the percentage saved
>    versus an API-only baseline.
>
> Make the routing rationale inspectable (log why each query was routed where).

**ทำอะไร:** เพิ่ม router local vs API พร้อมเก็บ metric ต้นทุน/latency และคำนวณ % ที่ประหยัดได้ ให้ตรวจสอบ
เหตุผลการ route ได้

---

## Phase 5 — Production Polish

> Implement Phase 5. Make Sourcerer production-grade:
> 1. Add a vLLM option for inference (OpenAI-compatible API via `VLLM_BASE_URL`), swappable with
>    Ollama through the existing LLM client wrapper. Add a note on measuring throughput vs Ollama.
> 2. Add observability: structured logging of every query with its retrieval trace, route, tokens,
>    latency, and cost.
> 3. Add guardrails: when no relevant context is retrieved, the system must answer "I don't know"
>    rather than guess. Add a basic prompt-injection check on inputs.
> 4. Finalize `docker-compose` so `docker-compose up` runs the whole stack.
> 5. Write the full README per the blueprint (demo, problem, architecture diagram, eval results,
>    trade-offs, how to run).

**ทำอะไร:** ทำให้ดูเป็นของจริง ย้าย vLLM, ใส่ observability + guardrails, dockerize ครบ และเขียน README
ตัวเต็ม (อย่าลืมเอาผล eval จาก Phase 3–4 ใส่ใน README)

---

## Phase 6 — GraphRAG (ทางเลือกต่อยอด)

> Implement Phase 6 (optional extension) only after Phases 1–5 are solid. Add GraphRAG as a parallel
> retrieval path:
> 1. Build a graph-indexing pipeline that extracts entities/relationships and pre-generates community
>    summaries. **Use the local model for extraction** (`GRAPHRAG_EXTRACTION_MODEL`) to control cost,
>    and start with a deliberately small corpus.
> 2. Support both local search (entity-specific) and global search (whole-corpus themes).
> 3. Extend the router: overview/whole-corpus questions → GraphRAG global; specific questions →
>    existing hybrid retrieval.
> 4. Reuse the Phase 3 eval harness to compare GraphRAG vs hybrid RAG on overview-type questions,
>    reporting both quality and cost.
>
> Gate this behind `GRAPHRAG_ENABLED`. Warn me before any step that would trigger expensive indexing.

**ทำอะไร:** เพิ่ม GraphRAG เป็นเส้น retrieval ขนาน ใช้ local model สกัดเพื่อคุมต้นทุน เริ่ม corpus เล็ก
และเอา eval มาวัดเทียบว่าคุ้มไหม

---

## เกร็ดการสั่งงานที่ช่วยให้คุณภาพดีขึ้น

- **ให้มันวางแผนก่อนลงมือ:** เติมท้าย prompt ว่า "Plan first, show me the steps, then implement after I
  confirm." ลดงานที่ต้องรื้อ
- **บังคับให้วัดผล:** ถ้ามันอ้างว่า "this is better" ให้ถามกลับว่า "show me the eval numbers"
- **กันงานบวม:** ถ้ามันเริ่มทำหลายอย่างพร้อมกัน สั่ง "one step at a time; stay within Phase N"
- **ให้มันเขียนเทสต์:** "add tests for this module before moving on" โดยเฉพาะส่วน retrieval กับ routing
- **อัปเดตสถานะ:** จบ phase แล้วสั่ง "update the Current status checklist in CLAUDE.md"
