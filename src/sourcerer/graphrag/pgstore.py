"""Postgres + pgvector backend for the GraphRAG index (production path).

Stores entities, relationships, and communities as tables instead of one JSON
file, and **embeds each community summary** so global search can rank communities
by similarity to the query (`summary_embedding <=> query`) rather than the JSON
store's top-N-by-size heuristic. At scale this is the difference between scanning
the whole graph in memory per query and fetching only the few relevant
communities via an HNSW index.

Tables are created in db.session.init_schema. Embeddings reuse the local bge-m3
client (same model as chunk embeddings), so no new model is involved.
"""

from __future__ import annotations

from sourcerer.config import Settings
from sourcerer.db.session import connect, to_vector_literal
from sourcerer.graphrag.types import Community, Entity, GraphIndex, Relationship


def exists(settings: Settings) -> bool:
    """True if a graph has been written to Postgres."""
    with connect() as conn:
        reg = conn.execute("SELECT to_regclass('graph_communities')").fetchone()[0]
        if reg is None:
            return False
        return conn.execute("SELECT EXISTS(SELECT 1 FROM graph_communities)").fetchone()[0]


def save(index: GraphIndex, settings: Settings) -> None:
    """Replace the stored graph; embed summarized communities into pgvector."""
    from sourcerer.ingestion.embeddings import embed_texts

    summarized = [c for c in index.communities if c.summary.strip()]
    vecs = embed_texts([c.summary for c in summarized]) if summarized else []
    emb_by_id = {id(c): v for c, v in zip(summarized, vecs, strict=True)}

    with connect() as conn:
        conn.execute("TRUNCATE graph_entities, graph_relationships, graph_communities")
        for e in index.entities:
            conn.execute(
                "INSERT INTO graph_entities (name, type, description, chunk_ids, sources) "
                "VALUES (%s, %s, %s, %s, %s)",
                (e.name, e.type, e.description, e.chunk_ids, getattr(e, "sources", [])),
            )
        for r in index.relationships:
            conn.execute(
                "INSERT INTO graph_relationships (source, target, description) VALUES (%s, %s, %s)",
                (r.source, r.target, r.description),
            )
        for c in index.communities:
            vec = emb_by_id.get(id(c))
            conn.execute(
                "INSERT INTO graph_communities "
                "(id, summary, entity_names, sources, summary_embedding) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    c.id,
                    c.summary,
                    c.entity_names,
                    c.sources,
                    to_vector_literal(vec) if vec is not None else None,
                ),
            )


def load(settings: Settings) -> GraphIndex:
    """Load the whole graph into a GraphIndex (used by rebuild — infrequent)."""
    with connect() as conn:
        ents = [
            Entity(r[0], r[1], r[2], list(r[3]))
            for r in conn.execute(
                "SELECT name, type, description, chunk_ids FROM graph_entities"
            ).fetchall()
        ]
        rels = [
            Relationship(r[0], r[1], r[2])
            for r in conn.execute(
                "SELECT source, target, description FROM graph_relationships"
            ).fetchall()
        ]
        comms = [
            Community(r[0], list(r[1]), r[2], list(r[3]))
            for r in conn.execute(
                "SELECT id, entity_names, summary, sources FROM graph_communities ORDER BY id"
            ).fetchall()
        ]
    return GraphIndex(entities=ents, relationships=rels, communities=comms)


def indexed_sources(settings: Settings) -> set[str]:
    """All source files referenced by the stored communities (for staleness check)."""
    with connect() as conn:
        rows = conn.execute("SELECT DISTINCT unnest(sources) FROM graph_communities").fetchall()
    return {r[0] for r in rows}


def top_communities(
    query: str, settings: Settings, live_sources: set[str] | None = None, k: int = 8
) -> list[Community]:
    """Return the communities most similar to the query (pgvector-ranked).

    Over-fetches, drops communities whose every source was deleted, then takes the
    top-k — semantic relevance instead of "largest by entity count".
    """
    from sourcerer.ingestion.embeddings import embed_query

    qvec = to_vector_literal(embed_query(query))
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, entity_names, summary, sources FROM graph_communities "
            "WHERE summary_embedding IS NOT NULL "
            "ORDER BY summary_embedding <=> %s LIMIT %s",
            (qvec, k * 3),
        ).fetchall()
    comms = [Community(r[0], list(r[1]), r[2], list(r[3])) for r in rows]
    if live_sources is not None:
        comms = [c for c in comms if not c.sources or any(s in live_sources for s in c.sources)]
    return comms[:k]
