"""Knowledge-base (corpus) management: list and delete ingested sources.

A "source" is one ingested document, identified by its file name (the `source`
column on the chunks table). Deleting a source removes all of its chunks.
"""

from __future__ import annotations

from sourcerer.db.session import connect


def list_sources() -> list[dict]:
    """Return each ingested source with its chunk count, newest first."""
    with connect() as conn:
        rows = conn.execute("""
            SELECT source, count(*) AS chunks, max(created_at) AS ingested
            FROM chunks
            GROUP BY source
            ORDER BY ingested DESC, source
            """).fetchall()
    return [{"source": r[0], "chunks": r[1], "ingested": r[2].isoformat()} for r in rows]


def delete_source(source: str) -> int:
    """Delete all chunks for a given source. Returns the number of rows removed."""
    with connect() as conn:
        cur = conn.execute("DELETE FROM chunks WHERE source = %s", (source,))
        return cur.rowcount
