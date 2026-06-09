"""CLI entrypoint for the evaluation harness.

Usage:
    python scripts/run_eval.py                 # ingest eval corpus, run all configs, judge on
    python scripts/run_eval.py --no-judge      # retrieval metrics only (fast)
    python scripts/run_eval.py --no-ingest     # evaluate against the current DB contents
    python scripts/run_eval.py --rerank-with local   # use bge cross-encoder for hybrid+rerank
"""

from __future__ import annotations

import argparse
import logging

from sourcerer.config import get_settings
from sourcerer.eval.runner import default_configs, run


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness.")
    parser.add_argument(
        "--no-judge", action="store_true", help="Skip LLM-judged generation metrics."
    )
    parser.add_argument(
        "--no-ingest", action="store_true", help="Don't (re)ingest the eval corpus."
    )
    parser.add_argument("--k", type=int, default=None, help="Override top-k for retrieval metrics.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N items.")
    parser.add_argument(
        "--rerank-with",
        default="llm",
        choices=["llm", "local", "cohere"],
        help="Reranker backend for the hybrid+rerank config (default: llm).",
    )
    args = parser.parse_args()

    run(
        configs=default_configs(rerank_with=args.rerank_with),
        judge_enabled=not args.no_judge,
        ingest=not args.no_ingest,
        k=args.k,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
