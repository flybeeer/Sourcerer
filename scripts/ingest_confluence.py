"""CLI entrypoint for ingesting Confluence pages as documents (optional source).

Usage:
    python scripts/ingest_confluence.py "space = ENG and type = page" \
        [--base-url https://yoursite.atlassian.net/wiki] [--email you@co.com] \
        [--token ...] [--classification internal --owner-team eng --pii email]

A CQL query selects the pages to pull in — the same idea as the SQL KB's
user-written SELECT (Path A). Each page becomes one document, then goes
through the normal chunk -> embed -> store pipeline into pgvector. Credentials
default to CONFLUENCE_BASE_URL / CONFLUENCE_EMAIL / CONFLUENCE_API_TOKEN in
.env; the optional governance flags (Phase 10a) tag every ingested page in the
asset catalog — metadata only, enforcement is the existing document gate.
"""

from __future__ import annotations

import argparse
import logging

from sourcerer.config import get_settings
from sourcerer.ingestion.pipeline import ingest_confluence


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Ingest Confluence pages into pgvector.")
    parser.add_argument(
        "cql", help='A CQL query choosing the pages to ingest, e.g. "space = ENG".'
    )
    parser.add_argument(
        "--base-url",
        default=settings.confluence_base_url,
        help="Confluence site wiki root (default: CONFLUENCE_BASE_URL).",
    )
    parser.add_argument(
        "--email",
        default=settings.confluence_email,
        help="Confluence Cloud account email (default: CONFLUENCE_EMAIL).",
    )
    parser.add_argument(
        "--token",
        default=settings.confluence_api_token,
        help="Confluence API token (default: CONFLUENCE_API_TOKEN).",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=settings.confluence_max_pages,
        help=f"Safety cap on pages pulled (default: {settings.confluence_max_pages}).",
    )
    # Phase 10a — tag the ingested pages in the governance asset catalog.
    parser.add_argument(
        "--classification",
        choices=("public", "internal", "confidential", "restricted"),
        help="Catalogue every ingested page at this sensitivity (governance).",
    )
    parser.add_argument(
        "--owner-team",
        help="Owning team recorded in the asset catalog (governance).",
    )
    parser.add_argument(
        "--pii",
        help="Comma-separated PII tags to record for the ingested pages (governance).",
    )
    args = parser.parse_args()

    if not (args.base_url and args.email and args.token):
        raise SystemExit(
            "Missing Confluence credentials: set CONFLUENCE_BASE_URL/CONFLUENCE_EMAIL/"
            "CONFLUENCE_API_TOKEN in .env, or pass --base-url/--email/--token."
        )

    summary = ingest_confluence(
        args.cql,
        base_url=args.base_url,
        email=args.email,
        api_token=args.token,
        max_pages=args.max_pages,
    )
    print(f"Done. Ingested {summary['documents']} page(s), {summary['chunks']} chunk(s).")

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
