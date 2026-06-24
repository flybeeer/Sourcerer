"""Asset catalog — governance metadata for sources/tables/columns (Phase 10a).

Maps an *asset* (a document `source`, a SQL `table`, or a `column`) to the
attributes the Cerbos gate will police from 10b on: a sensitivity
`classification`, the `owner_team`, and `pii_tags`. 10a only *records* this
metadata — nothing is enforced yet.

Default `CATALOG_BACKEND=local` keeps the catalog in the existing Postgres
(`asset_catalog` table); `openmetadata` is left as a production swap (same seam
as Ollama→vLLM, SQLite→DuckDB, JSON→Postgres graph store).

Pure helpers (classification ordering, tag parsing) live here too so the policy
math is unit-testable without a database.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sourcerer.db.session import connect

# Sensitivity lattice: public < internal < confidential < restricted. A principal
# may read an asset when their clearance rank >= the asset's classification rank
# (the actual decision is Cerbos' from 10b; the ordering is defined once here).
CLASSIFICATIONS: tuple[str, ...] = ("public", "internal", "confidential", "restricted")

# An asset can be a document source, a SQL table, or a single column.
AssetKind = str  # "document" | "table" | "column"


def classification_rank(level: str) -> int:
    """Rank a classification on the lattice. Unknown levels sort as most-restricted.

    Defaulting the unknown case high is fail-safe: a mis-tagged asset is treated
    as more sensitive, not less.
    """
    try:
        return CLASSIFICATIONS.index(level)
    except ValueError:
        return len(CLASSIFICATIONS)


def parse_pii_tags(raw: str | None) -> list[str]:
    """Parse a comma-separated PII tag string (e.g. "email, salary") to a list."""
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


@dataclass(frozen=True)
class Asset:
    """One catalogued asset and its governance attributes."""

    kind: AssetKind
    id: str
    classification: str = "internal"
    owner_team: str | None = None
    pii_tags: list[str] = field(default_factory=list)
    glossary: str | None = None

    def attr(self) -> dict:
        """Attribute bag shaped for a Cerbos `resource.attr` (used from 10b)."""
        return {
            "classification": self.classification,
            "owner_team": self.owner_team,
            "pii_tags": self.pii_tags,
        }


def init_catalog_schema() -> None:
    """Create the `asset_catalog` table if absent (idempotent).

    Kept out of `db.session.init_schema` so governance stays a cleanly separable,
    opt-in pillar — callers that tag or read the catalog ensure the table first.
    """
    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS asset_catalog (
                kind           TEXT        NOT NULL,
                asset_id       TEXT        NOT NULL,
                classification TEXT        NOT NULL DEFAULT 'internal',
                owner_team     TEXT,
                pii_tags       TEXT[]      NOT NULL DEFAULT '{}',
                glossary       TEXT,
                updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
                PRIMARY KEY (kind, asset_id)
            )
            """)


def upsert_asset(asset: Asset) -> None:
    """Insert or update one asset's governance metadata (keyed on kind+id)."""
    init_catalog_schema()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO asset_catalog
                (kind, asset_id, classification, owner_team, pii_tags, glossary, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, now())
            ON CONFLICT (kind, asset_id) DO UPDATE SET
                classification = EXCLUDED.classification,
                owner_team     = EXCLUDED.owner_team,
                pii_tags       = EXCLUDED.pii_tags,
                glossary       = EXCLUDED.glossary,
                updated_at     = now()
            """,
            (
                asset.kind,
                asset.id,
                asset.classification,
                asset.owner_team,
                asset.pii_tags,
                asset.glossary,
            ),
        )


def get_asset(kind: AssetKind, asset_id: str) -> Asset | None:
    """Return one catalogued asset, or None if it isn't tagged."""
    with connect() as conn:
        row = conn.execute(
            "SELECT kind, asset_id, classification, owner_team, pii_tags, glossary "
            "FROM asset_catalog WHERE kind = %s AND asset_id = %s",
            (kind, asset_id),
        ).fetchone()
    return _row_to_asset(row) if row else None


def list_assets(kind: AssetKind | None = None) -> list[Asset]:
    """List catalogued assets, optionally filtered by kind."""
    with connect() as conn:
        if kind is None:
            rows = conn.execute(
                "SELECT kind, asset_id, classification, owner_team, pii_tags, glossary "
                "FROM asset_catalog ORDER BY kind, asset_id"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT kind, asset_id, classification, owner_team, pii_tags, glossary "
                "FROM asset_catalog WHERE kind = %s ORDER BY asset_id",
                (kind,),
            ).fetchall()
    return [_row_to_asset(r) for r in rows]


def delete_asset(kind: AssetKind, asset_id: str) -> bool:
    """Remove an asset's catalog entry. Returns True if a row was deleted."""
    with connect() as conn:
        cur = conn.execute(
            "DELETE FROM asset_catalog WHERE kind = %s AND asset_id = %s",
            (kind, asset_id),
        )
        return cur.rowcount > 0


def _row_to_asset(row: tuple) -> Asset:
    return Asset(
        kind=row[0],
        id=row[1],
        classification=row[2],
        owner_team=row[3],
        pii_tags=list(row[4]) if row[4] else [],
        glossary=row[5],
    )
