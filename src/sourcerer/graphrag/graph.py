"""Build the entity graph and detect communities.

Entities extracted per-chunk are merged by normalized name; relationships become
edges. Communities are found with modularity-based clustering (networkx) — the
"global summary" structure GraphRAG is built around. A small corpus usually
yields a handful of communities, each a coherent topic cluster.
"""

from __future__ import annotations

from sourcerer.graphrag.types import Community, Entity, Relationship, normalize_name


def merge_entities(entities: list[Entity]) -> list[Entity]:
    """Deduplicate entities by normalized name, unioning descriptions + chunks."""
    merged: dict[str, Entity] = {}
    for e in entities:
        key = normalize_name(e.name)
        if key not in merged:
            merged[key] = Entity(
                name=e.name, type=e.type, description=e.description, chunk_ids=list(e.chunk_ids)
            )
            continue
        cur = merged[key]
        for cid in e.chunk_ids:
            if cid not in cur.chunk_ids:
                cur.chunk_ids.append(cid)
        # Keep the longest description (usually the most informative).
        if len(e.description) > len(cur.description):
            cur.description = e.description
    return list(merged.values())


def detect_communities(
    entities: list[Entity], relationships: list[Relationship]
) -> list[Community]:
    """Cluster entities into communities via greedy modularity (networkx).

    Falls back to one community per connected component if networkx is missing,
    and to a single all-entity community for tiny/edgeless graphs.
    """
    names = [e.name for e in entities]
    valid = {normalize_name(n) for n in names}
    edges = [
        (normalize_name(r.source), normalize_name(r.target))
        for r in relationships
        if normalize_name(r.source) in valid and normalize_name(r.target) in valid
    ]
    if not names:
        return []

    try:
        import networkx as nx

        g = nx.Graph()
        g.add_nodes_from(normalize_name(n) for n in names)
        g.add_edges_from(edges)
        try:
            groups = nx.community.greedy_modularity_communities(g)
        except (ZeroDivisionError, nx.NetworkXError):
            groups = list(nx.connected_components(g))
    except ImportError:
        groups = _connected_components(valid, edges)

    by_key = {normalize_name(e.name): e.name for e in entities}
    communities: list[Community] = []
    for i, group in enumerate(groups):
        members = [by_key[k] for k in group if k in by_key]
        if members:
            communities.append(Community(id=i, entity_names=members))
    return communities


def _connected_components(nodes: set[str], edges: list[tuple[str, str]]) -> list[set[str]]:
    """Pure-Python connected components (networkx fallback)."""
    parent = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        parent[find(a)] = find(b)
    groups: dict[str, set[str]] = {}
    for n in nodes:
        groups.setdefault(find(n), set()).add(n)
    return list(groups.values())
