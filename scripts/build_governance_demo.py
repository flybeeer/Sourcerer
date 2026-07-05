"""Seed a tiny governance demo: corpus + asset tags + principals (Phase 10b).

Inserts a handful of short documents straight into the `chunks` table with
placeholder embeddings (no Ollama needed) and real text, so the BM25/keyword leg
and the source-level gate can be exercised offline. Each doc is tagged in
`asset_catalog`, and a few principals are seeded — the matrix the leakage eval
(scripts/governance_eval.py) runs over.

Idempotent: re-running replaces these demo sources.

    python scripts/build_governance_demo.py
"""

from __future__ import annotations

from pathlib import Path

from sourcerer.config import get_settings
from sourcerer.db.session import connect, init_schema, to_vector_literal
from sourcerer.governance.catalog import Asset, init_catalog_schema, upsert_asset
from sourcerer.governance.principal import Principal, upsert_principal

# The CAG PoC (scripts/cag_poc.py) reads whole documents from disk, not chunks —
# so the same demo docs are also written here as files, under this dir. Point
# CAG_CORPUS_DIR at it (cag_poc.py does this) to run CAG on the seeded corpus.
DEMO_CORPUS_DIR = "data/governance_demo"

# (source, classification, owner_team, text). Text is themed so keyword search
# surfaces the right doc for the eval queries.
DOCS = [
    (
        "faq.md",
        "public",
        None,
        "Office opening hours: the building opens at 9am and closes at 6pm on weekdays.",
    ),
    (
        "security.md",
        "internal",
        "it",
        "Password policy: passwords must be rotated every 90 days and use MFA.",
    ),
    (
        "handbook.md",
        "confidential",
        "hr",
        "Employee salary bands: a senior engineer compensation range is defined here by HR.",
    ),
    (
        "board.md",
        "restricted",
        "exec",
        "Board meeting minutes: the confidential acquisition of a competitor was approved.",
    ),
]

PRINCIPALS = [
    Principal(id="anonymous", roles=[], clearance="public", teams=[]),
    Principal(id="alice", roles=["analyst"], clearance="internal", teams=["ops"], region="APAC"),
    Principal(id="bob", roles=["hr"], clearance="confidential", teams=["hr"], region="EMEA"),
    Principal(id="carol", roles=["admin"], clearance="restricted", teams=["hr", "exec"]),
]


def main() -> None:
    settings = get_settings()
    init_schema()
    init_catalog_schema()

    corpus_dir = Path(DEMO_CORPUS_DIR)
    corpus_dir.mkdir(parents=True, exist_ok=True)

    zero = to_vector_literal([0.0] * settings.embedding_dim)  # placeholder embedding
    with connect() as conn:
        for source, classification, owner_team, text in DOCS:
            # RAG path: seed chunks (for scripts/governance_eval.py).
            conn.execute("DELETE FROM chunks WHERE source = %s", (source,))
            conn.execute(
                "INSERT INTO chunks (source, chunk_index, content, embedding) "
                "VALUES (%s, 0, %s, %s::vector)",
                (source, text, zero),
            )
            # CAG path: write the same doc to disk (cag reads files, not chunks).
            (corpus_dir / source).write_text(text)
            upsert_asset(
                Asset(
                    kind="document", id=source, classification=classification, owner_team=owner_team
                )
            )

    for p in PRINCIPALS:
        upsert_principal(p)

    print(f"Seeded {len(DOCS)} demo doc(s) + {len(PRINCIPALS)} principal(s).")
    print(f"  chunks (RAG eval) + files on disk in {DEMO_CORPUS_DIR}/ (CAG)")
    print("Sources:", ", ".join(f"{s} ({c})" for s, c, _, _ in DOCS))


if __name__ == "__main__":
    main()
