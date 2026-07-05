# Design Note — CAG ใช้งานร่วมกับ Cerbos (PoC)

**สถานะ:** Proof of Concept — รันได้จริง (Cerbos sidecar + Ollama + Postgres)
**ขอบเขต:** ตอบโจทย์ "ลอง CAG ร่วมกับ Cerbos" — ไม่แตะ production path เดิม (Phase 10)
**โค้ด:** `src/sourcerer/cag/`, demo: `scripts/cag_poc.py`

---

## สรุปผู้บริหาร (TL;DR)

- **CAG (Cache-Augmented Generation)** โหลดเอกสารทั้งชุดเข้า context ของโมเดลไว้ล่วงหน้า แล้วตอบจาก cache โดย **ไม่ retrieve ต่อ query** — เร็วกว่า RAG เพราะตัดขั้น embed + search ออก
- **กับดัก:** ถ้า cache โหลด "ทุกเอกสาร" → ทุกคนมีเนื้อหาต้องห้ามใน context หมด → **ขัดกฎ governance** ที่ว่า "เนื้อหาต้องห้ามต้องไม่เข้า context"
- **ทางออก (ที่ PoC พิสูจน์แล้ว):** ย้าย Cerbos ไปตัดสินที่ **ตอน build cache** แทนตอน retrieve → cache แยกตามสิทธิ์ เอกสารต้องห้ามไม่เคยเข้า cache ของ principal นั้น → **ได้ความเร็ว CAG โดยไม่เสีย governance**
- **ผลลัพธ์:** ใช้ได้จริง, reuse policy + gate เดิมจาก Phase 10 ทั้งหมด ไม่ต้องเขียน authz ใหม่

---

## โจทย์และกับดัก

| | RAG (ระบบปัจจุบัน) | CAG |
|---|---|---|
| ตอน query | embed → vector/BM25 search → ดึง chunk ที่เกี่ยว | ตอบจาก context ที่ preload ไว้ (ไม่ search) |
| Governance hook | gate ตอน retrieve (Phase 10) | **ไม่มี retrieve ให้ hook** ← ปัญหา |
| ความเร็ว/query | ช้ากว่า (embed + search ทุกครั้ง) | เร็วกว่า (ไม่มี retrieval) |
| เหมาะกับ | corpus ใหญ่ | corpus เล็กพอใส่ context |

กฎเหล็ก Phase 10: *authorize ที่ retrieval/execution time — forbidden content ไม่เข้า candidate set*
→ CAG ไม่มี retrieval step ให้ใส่ gate ถ้าทำ cache รวมก้อนเดียว = เนื้อหาต้องห้ามอยู่ใน context ของทุกคน

---

## แนวทาง: gate ที่ "cache-build time"

ย้ายจุดที่ Cerbos ทำงาน ไม่ใช่ตัด governance ทิ้ง:

```
RAG:  query ──► Cerbos gate (retrieval time) ──► ดึงเฉพาะที่อนุญาต ──► generate
CAG:  [build] Cerbos gate ──► โหลดเฉพาะที่อนุญาตเข้า cache (ต่อ access-profile)
      [query] generate จาก cache  (ไม่มี gate, ไม่มี retrieval)
```

- ใช้ `governance.allowed_document_sources(principal)` ตัวเดียวกับ RAG path (PDP = Cerbos)
- Cache แยกตาม **access-profile** (allowed-set) ไม่ใช่ราย user — principal ที่สิทธิ์เท่ากัน share cache เดียว
- กฎเหล็กยังอยู่ครบ: เอกสารต้องห้าม **ไม่เคยถูกโหลด** เข้า cache ของ principal นั้นตั้งแต่แรก

---

## PoC พิสูจน์อะไร (ผลรันจริง)

**1. Cerbos partition cache ตั้งแต่ build time**
```
anonymous → [faq.md]                              (public)
alice     → [faq.md, security.md]                 (+internal)
bob       → [faq.md, handbook.md, security.md]    (+confidential/hr)
carol     → [faq.md, handbook.md, security.md, board.md]   (admin = ทั้งหมด)
```

**2. คำถามเดียวกัน ไม่มี retrieval → ผลต่างกันตามสิทธิ์**
ถาม *"employee salary bands?"* (handbook.md = confidential/hr):
```
anonymous / alice  → "I don't know"   (handbook ไม่อยู่ใน cache)
bob / carol        → ตอบจริง cite handbook.md   (อยู่ใน cache)
```

**3. Cache reuse** — bob ถาม 3 คำถาม: cache build ครั้งเดียว (~13ms) reuse ทุกครั้ง, ตอบ ~1s/query (ไม่มี embed/search)

---

## สถาปัตยกรรม

```
                 ┌──────────── BUILD TIME (ต่อ access-profile) ────────────┐
 principal ─────►│ governance.allowed_document_sources()  ──► Cerbos PDP    │
                 │        │ allowed sources (set)                           │
                 │        ▼                                                 │
                 │ load_full_documents() ──► assemble context ──► Cache     │
                 └──────────────────────────┬──────────────────────────────┘
                                            │ (memoize ตาม allowed-set)
                 ┌──────────── QUERY TIME ──▼──────────────────────────────┐
 question ──────►│ cag.generate.answer(question, cache, client)            │
                 │   = LLM( system + cache.context + question )            │
                 │   ไม่มี retrieval / ไม่มี gate (gate ทำไปแล้วตอน build)   │
                 └─────────────────────────────────────────────────────────┘
```

ชิ้นที่สร้าง (reuse ของเดิมทั้งหมด):
| ไฟล์ | หน้าที่ | reuse |
|------|--------|-------|
| `cag/cache.py` | build + memoize cache, gate ที่ build time | `governance.allowed_document_sources` (Cerbos) |
| `cag/generate.py` | ตอบจาก cache (ไม่มี retrieval) | `llm.client` wrapper |
| `scripts/cag_poc.py` | demo | corpus + principals เดิม |

---

## RAG vs CAG (เทียบ)

| | RAG (Phase 10) | CAG (PoC) |
|---|---|---|
| Governance ทำตอนไหน | ทุก query (retrieval time) | ครั้งเดียว (build time) ต่อ access-profile |
| ต้นทุน/query | embed + vector/BM25 + (rerank) | แค่ generate |
| Latency/query | สูงกว่า | ต่ำกว่า (PoC ~1s) |
| Corpus ขนาด | ใหญ่ได้ | จำกัด (ต้องใส่ context window) |
| Policy/PDP | Cerbos (เดียวกัน) | **Cerbos (เดียวกัน)** |

---

## ข้อจำกัด / ที่ยังไม่ทำ (ตรงไปตรงมา)

| ประเด็น | สถานะ | หมายเหตุ |
|--------|-------|---------|
| Cache แยกตาม tier (access-profile) | ✅ | principal สิทธิ์เท่ากัน share cache |
| KV-cache reuse ระดับ transformer จริง | ⚠️ | PoC พึ่ง Ollama prompt-cache (prefix คงที่); วัดจริงต้อง vLLM/llama.cpp prompt caching |
| **Cache invalidation** ตอนสิทธิ์/classification เปลี่ยน | ❌ | ต้อง clear cache เมื่อ policy/catalog/principal เปลี่ยน — สำคัญถ้าจะ prod |
| Corpus ใหญ่เกิน context | ❌ | premise ของ CAG; ทางออก = hybrid CAG (docs ร้อน) + RAG (ส่วนที่เหลือ) |
| Column/row masking (เหมือน SQL path) | ❌ | CAG เป็น doc-level; PII ระดับ field ต้องคิดต่อ |

---

## ข้อเสนอ / Next steps

1. **ถ้าจะลองจริง:** วัด latency CAG vs RAG บน corpus จริงให้เห็นตัวเลข + เติม cache invalidation (hook ตอน re-tag/แก้ policy)
2. **Scale:** ใช้ vLLM เพื่อ KV-prefix caching จริง (ได้ความเร็ว CAG เต็มที่) + พิจารณา hybrid CAG+RAG สำหรับ corpus ใหญ่
3. **Governance:** model เป็น doc-level อยู่แล้วพอสำหรับ RAG-style; ถ้าต้อง field-level masking ใน CAG ต้องออกแบบเพิ่ม

**บรรทัดเดียว:** CAG + Cerbos เข้ากันได้ดี — แค่ย้าย gate จาก retrieval-time → build-time, reuse policy/PDP เดิมทั้งหมด, ได้ความเร็วโดยไม่เสีย governance PoC พิสูจน์ end-to-end แล้ว เหลือ invalidation + การวัดผล scale ก่อนพิจารณา production

---

ref: [`governance-phase-10.md`](governance-phase-10.md) · [`governance-phase-10-demo.md`](governance-phase-10-demo.md)
