"""Compare GraphRAG global search vs hybrid RAG on *overview* questions.

Reuses the Phase 3 eval harness (LLM-as-judge) and reports both quality
(faithfulness, answer relevancy) and cost (tokens, latency) per approach. This is
the GraphRAG payoff question: on whole-corpus "what are the main themes?" queries,
does the graph beat vector RAG, and at what cost?

Requires a built index (scripts/graphrag_index.py) — it does NOT index here, but
it does make LLM calls for generation + judging, so it is not free.

    python scripts/graphrag_eval.py            # judged comparison
    python scripts/graphrag_eval.py --no-judge # latency/cost only
"""

from __future__ import annotations

import argparse
import time

from sourcerer.config import get_settings
from sourcerer.eval.dataset import load_eval_set
from sourcerer.eval.generation_metrics import Judge
from sourcerer.generation import generator
from sourcerer.graphrag import search as graph_search
from sourcerer.graphrag import store as graph_store
from sourcerer.llm.client import get_api_client, get_llm_client
from sourcerer.retrieval import retriever
from sourcerer.routing.pricing import estimate_cost


def _hybrid(question, settings, judge):
    started = time.perf_counter()
    chunks = retriever.retrieve(question, settings, mode="hybrid")
    ans = generator.generate(question, chunks, client=get_llm_client(settings))
    latency = (time.perf_counter() - started) * 1000
    context = "\n\n".join(c.content for c in chunks)
    cost = estimate_cost(settings.local_model, ans.input_tokens, ans.output_tokens)
    return _row(
        "hybrid RAG",
        ans.text,
        context,
        ans.input_tokens + ans.output_tokens,
        cost,
        latency,
        judge,
        question,
    )


def _graph(question, index, settings, judge):
    # Reduce on the API model when configured (map stays local) — matches /query.
    reduce_client = None
    name = "GraphRAG global"
    if settings.graphrag_reduce_with_api and settings.anthropic_api_key:
        reduce_client = get_api_client(settings)
        name = "GraphRAG (API reduce)"
    started = time.perf_counter()
    res = graph_search.global_search(
        question, index, get_llm_client(settings), reduce_client=reduce_client
    )
    latency = (time.perf_counter() - started) * 1000
    context = "\n\n".join(c.snippet for c in res.citations)
    # Only the reduce (answer) stage is priced; map + indexing are local/$0.
    cost = estimate_cost(res.model, res.answer_input_tokens, res.answer_output_tokens)
    return _row(
        name,
        res.text,
        context,
        res.input_tokens + res.output_tokens,
        cost,
        latency,
        judge,
        question,
    )


def _row(name, answer, context, tokens, cost, latency, judge, question):
    faith = judge.faithfulness(answer, context) if judge else None
    rel = judge.answer_relevancy(question, answer) if judge else None
    return {
        "approach": name,
        "faithfulness": faith,
        "answer_relevancy": rel,
        "tokens": tokens,
        "cost_usd": cost,
        "latency_ms": latency,
    }


def _avg(rows, key):
    vals = [r[key] for r in rows if r[key] is not None]
    return sum(vals) / len(vals) if vals else None


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="GraphRAG vs hybrid on overview questions.")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-judged quality.")
    args = parser.parse_args()

    if not graph_store.exists(settings.graphrag_root):
        raise SystemExit(
            f"No GraphRAG index at {settings.graphrag_root}. "
            "Build it first: python scripts/graphrag_index.py"
        )

    items = load_eval_set(settings.graphrag_overview_eval_set)
    index = graph_store.load(settings.graphrag_root)
    judge = None if args.no_judge else Judge(settings)
    print(f"Comparing on {len(items)} overview questions …\n")

    hybrid_rows, graph_rows = [], []
    for item in items:
        print(f"Q: {item.question}")
        hybrid_rows.append(_hybrid(item.question, settings, judge))
        graph_rows.append(_graph(item.question, index, settings, judge))

    def fmt(v, pct=False):
        if v is None:
            return "—"
        return f"{v:.3f}" if not pct else f"{v:.0f}"

    print("\n| Approach | Faithfulness | Answer rel. | Avg tokens | Avg cost | Avg latency ms |")
    print("|----------|-------------:|------------:|-----------:|---------:|---------------:|")
    for name, rows in (("hybrid RAG", hybrid_rows), ("GraphRAG global", graph_rows)):
        print(
            f"| {name} | {fmt(_avg(rows, 'faithfulness'))} | {fmt(_avg(rows, 'answer_relevancy'))} "
            f"| {fmt(_avg(rows, 'tokens'), pct=True)} | ${_avg(rows, 'cost_usd'):.5f} "
            f"| {fmt(_avg(rows, 'latency_ms'), pct=True)} |"
        )
    print(
        "\nNote: GraphRAG global makes N+1 LLM calls (one map per selected community "
        "+ one reduce) over short pre-computed summaries; hybrid makes one call over "
        "full chunks. Which wins on quality and cost is empirical — and depends "
        "heavily on extraction quality (a small extraction model yields a sparse "
        "graph and thin summaries). Read the numbers above, don't assume."
    )


if __name__ == "__main__":
    main()
