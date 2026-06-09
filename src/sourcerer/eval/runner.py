"""Eval runner: compare retrieval configs on the eval set, print + save a table.

Configs compared:
- vector-only    : Phase 1 path (vector search only).
- hybrid         : vector ∥ BM25 → RRF (no rerank).
- hybrid+rerank  : hybrid → reranker (RERANKER_TYPE, default 'llm' for this run).

For each config and eval item we measure retrieval metrics (recall@k, MRR, hit
rate) and, unless --no-judge, generation metrics (faithfulness, answer relevancy)
via the LLM judge.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from sourcerer.config import Settings, get_settings
from sourcerer.eval import report, retrieval_metrics
from sourcerer.eval.dataset import EvalItem, load_eval_set
from sourcerer.eval.generation_metrics import Judge
from sourcerer.generation import generator
from sourcerer.ingestion.pipeline import ingest_directory
from sourcerer.retrieval import retriever


@dataclass
class EvalConfig:
    name: str
    mode: str  # "vector" | "hybrid"
    reranker_type: str  # "none" | "llm" | "local" | "cohere"


def default_configs(rerank_with: str = "llm") -> list[EvalConfig]:
    return [
        EvalConfig("vector-only", "vector", "none"),
        EvalConfig("hybrid", "hybrid", "none"),
        EvalConfig("hybrid+rerank", "hybrid", rerank_with),
    ]


def _settings_for(base: Settings, cfg: EvalConfig) -> Settings:
    """A copy of settings with this config's mode/reranker applied."""
    return base.model_copy(update={"retrieval_mode": cfg.mode, "reranker_type": cfg.reranker_type})


def evaluate_config(
    cfg: EvalConfig,
    items: list[EvalItem],
    base: Settings,
    k: int,
    judge: Judge | None,
) -> dict:
    """Run one config over all eval items and return aggregated metrics."""
    settings = _settings_for(base, cfg)
    per_query: list[dict] = []
    faithfulness_scores: list[float] = []
    relevancy_scores: list[float] = []
    latencies: list[float] = []

    for item in items:
        # Time retrieval only — that's what differs between configs.
        started = time.perf_counter()
        chunks = retriever.retrieve(item.question, settings, mode=cfg.mode, top_k_final=k)
        latencies.append((time.perf_counter() - started) * 1000)

        sources = [c.source for c in chunks]
        per_query.append(retrieval_metrics.per_query_metrics(sources, item.relevant_doc_ids, k))

        if judge is not None:
            answer = generator.generate(item.question, chunks)
            context = "\n\n".join(c.content for c in chunks)
            faithfulness_scores.append(judge.faithfulness(answer.text, context))
            relevancy_scores.append(judge.answer_relevancy(item.question, answer.text))

    agg = retrieval_metrics.aggregate(per_query)
    row = {
        "config": cfg.name,
        "recall@k": agg["recall@k"],
        "mrr": agg["mrr"],
        "hit_rate": agg["hit_rate"],
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "faithfulness": (
            sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else None
        ),
        "answer_relevancy": (
            sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else None
        ),
    }
    return row


def run(
    configs: list[EvalConfig] | None = None,
    judge_enabled: bool = True,
    ingest: bool = True,
    k: int | None = None,
    limit: int | None = None,
) -> list[dict]:
    """Run the full comparison and save a report. Returns the result rows."""
    settings = get_settings()
    k = k or settings.top_k_final
    configs = configs or default_configs()

    if ingest:
        print(f"Ingesting eval corpus from {settings.eval_corpus_path} …")
        summary = ingest_directory(Path(settings.eval_corpus_path))
        print(f"  {summary['documents']} docs, {summary['chunks']} chunks")

    items = load_eval_set(settings.eval_set_path)
    if limit:
        items = items[:limit]
    print(f"Loaded {len(items)} eval items.")

    judge = Judge(settings) if judge_enabled else None

    rows = []
    for cfg in configs:
        print(f"Evaluating: {cfg.name} (mode={cfg.mode}, rerank={cfg.reranker_type}) …")
        rows.append(evaluate_config(cfg, items, settings, k, judge))

    table = report.render_table(rows, k)
    print("\n" + table + "\n")

    meta = {
        "n_items": len(items),
        "generator": settings.local_model,
        "judge": settings.eval_judge_model if judge_enabled else "(skipped)",
        "embedding": settings.embedding_model,
    }
    path = report.save_results(rows, k, meta, "eval/results")
    print(f"Saved report to {path}")
    return rows
