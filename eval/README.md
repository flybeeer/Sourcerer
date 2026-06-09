# Evaluation (Phase 3)

The eval harness measures retrieval and generation quality with numbers, so we
can prove which retrieval configuration is best — not guess.

## Files

- `corpus/` — a small synthetic document corpus (committed, non-sensitive) used
  for reproducible evaluation. Contains target docs (offices, SLAs, on-call, …)
  plus distractors (security, benefits, travel, …) so retrieval is non-trivial.
- `eval_set.jsonl` — the labelled question set (see format below).
- `results/` — generated comparison tables (gitignored).

## Eval set format (JSONL)

One JSON object per line:

```json
{
  "question": "Where is Sourcerer's headquarters located?",
  "expected_answer": "In Chiang Mai, Thailand.",
  "relevant_doc_ids": ["offices.md"]
}
```

- `question` — the user query.
- `expected_answer` — the ground-truth answer (reference for generation metrics).
- `relevant_doc_ids` — the **source file names** that should be retrieved. We
  evaluate retrieval at the document level (the chunk's `source`), because chunk
  database ids are not stable across re-ingests.

Expand this set freely — aim for 50–100 items for a robust signal.

## Metrics

**Retrieval** (over the final returned chunks, at k = `TOP_K_FINAL`):
- `recall@k` — fraction of a question's relevant docs that appear in the top-k.
- `MRR` — reciprocal rank of the first relevant doc.
- `hit_rate` — fraction of questions with at least one relevant doc in the top-k.

**Generation** (LLM-as-judge, `EVAL_JUDGE_MODEL`):
- `faithfulness` — is the answer supported by the retrieved context? (no hallucination)
- `answer_relevancy` — does the answer actually address the question?

## Running

```bash
python scripts/run_eval.py            # ingests eval/corpus, runs all 3 configs, saves a table
python scripts/run_eval.py --no-judge # retrieval metrics only (fast, no LLM judging)
```

Results are printed and saved under `eval/results/`.
