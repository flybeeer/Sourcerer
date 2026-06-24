"""CLI entrypoint for ingestion.

Usage:
    python scripts/ingest.py [path]
    python scripts/ingest.py data/hr --classification confidential --owner-team hr --pii salary

`path` defaults to ./data/raw. Loads every supported document under it, chunks,
embeds, and stores the vectors in pgvector. The optional governance flags
(Phase 10a) tag every ingested source in the asset catalog — metadata only;
enforcement arrives in 10b+.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sourcerer.config import get_settings
from sourcerer.ingestion.pipeline import ingest_directory


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Ingest documents into pgvector.")
    parser.add_argument(
        "path",
        nargs="?",
        default="data/raw",
        help="Directory of documents to ingest (default: data/raw).",
    )
    # Phase 10a — tag the ingested sources in the governance asset catalog. These
    # only record metadata; enforcement (the Cerbos gate) arrives in 10b+.
    parser.add_argument(
        "--classification",
        choices=("public", "internal", "confidential", "restricted"),
        help="Catalogue every ingested source at this sensitivity (governance).",
    )
    parser.add_argument(
        "--owner-team",
        help="Owning team recorded in the asset catalog (governance).",
    )
    parser.add_argument(
        "--pii",
        help="Comma-separated PII tags to record for the ingested sources (governance).",
    )
    args = parser.parse_args()

    root = Path(args.path)
    if not root.exists():
        raise SystemExit(f"Path does not exist: {root}")

    summary = ingest_directory(root)
    print(f"Done. Ingested {summary['documents']} document(s), {summary['chunks']} chunk(s).")

    if args.classification or args.owner_team or args.pii:
        # Lazy import keeps ingestion working without exercising the governance pillar.
        from sourcerer.governance.catalog import Asset, parse_pii_tags, upsert_asset

        pii_tags = parse_pii_tags(args.pii)
        for source in summary["sources"]:
            upsert_asset(
                Asset(
                    kind="document",
                    id=source,
                    classification=args.classification or "internal",
                    owner_team=args.owner_team,
                    pii_tags=pii_tags,
                )
            )
        print(f"Catalogued {len(summary['sources'])} source(s) in asset_catalog.")


if __name__ == "__main__":
    main()
