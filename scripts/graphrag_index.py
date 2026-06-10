"""Build the GraphRAG index over the ingested corpus.

⚠️  THIS IS THE EXPENSIVE STEP. It runs the LOCAL model over every chunk
(entity extraction) plus once per community (summaries). On a small CPU model
that is minutes, not seconds. It is gated behind a confirmation prompt; pass
--yes to skip it (e.g. in CI).

    python scripts/graphrag_index.py            # prints estimate, asks to confirm
    python scripts/graphrag_index.py --yes      # run without prompting
    python scripts/graphrag_index.py --estimate # show the cost estimate and exit
"""

from __future__ import annotations

import argparse
import logging
import sys

from sourcerer import corpus
from sourcerer.config import get_settings
from sourcerer.graphrag import index as graph_index


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Build the GraphRAG index (expensive).")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    parser.add_argument("--estimate", action="store_true", help="Show the estimate and exit.")
    args = parser.parse_args()

    chunks = corpus.all_chunks()
    if not chunks:
        print("No chunks in the corpus. Run scripts/ingest.py first.")
        sys.exit(1)

    est = graph_index.estimate(chunks)
    print("\n⚠️  GraphRAG indexing — cost estimate")
    print(f"    chunks                : {est['chunks']}")
    print(f"    extraction LLM calls  : {est['extraction_calls']}")
    print(f"    summary LLM calls(est): {est['summary_calls_est']}")
    print(f"    TOTAL local LLM calls : ~{est['total_llm_calls_est']}")
    print(f"    extraction model      : {settings.graphrag_extraction_model}")
    print("    On a small CPU model this can take several minutes.\n")

    if args.estimate:
        return

    if not settings.graphrag_enabled:
        print("GRAPHRAG_ENABLED is false. Set it to true in .env to use the graph path.")

    if not args.yes:
        reply = input("Proceed with indexing? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted.")
            return

    idx = graph_index.build_index(chunks, settings)
    print(
        f"\nDone. {len(idx.entities)} entities, {len(idx.relationships)} relationships, "
        f"{len(idx.communities)} communities → {settings.graphrag_root}"
    )


if __name__ == "__main__":
    main()
