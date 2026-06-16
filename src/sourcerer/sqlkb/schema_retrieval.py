"""Schema retrieval: pick the tables a question actually needs.

Dumping every table's DDL into the Text-to-SQL prompt breaks down on a wide
schema (hundreds of tables) — it blows the context window and buries the relevant
tables in noise, so the model writes worse SQL. Instead we *retrieve* the tables,
reusing the project's signature move: rank by **lexical** overlap and **embedding**
similarity, fuse the two rankings with RRF (same idea as document retrieval), and
keep the top-K. Lexical is deterministic and free; the embedding leg (bge-m3, which
is multilingual) catches semantic/cross-language matches a Thai question needs.

Only table *signatures* (name + columns) are ranked — cheap to build for every
table; the expensive full descriptions (with sample rows) are built afterwards for
just the survivors, in `schema.describe_schema(..., tables=...)`.
"""

from __future__ import annotations

import re
from pathlib import Path

from sourcerer.config import Settings, get_settings
from sourcerer.llm.client import LLMClient
from sourcerer.sqlkb import schema

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MIN_TOKEN = 3  # ignore 1-2 char tokens ("id", "of") — too noisy to match on


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= _MIN_TOKEN}


def _lexical_ranking(question: str, signatures: dict[str, str]) -> list[str]:
    """Order tables by how many question tokens appear in each signature.

    A token matches when it contains, or is contained by, a signature token (so
    "employees" hits the `employee_id` column / `employees` table). Ties keep the
    original (stable) table order.
    """
    q_tokens = _tokens(question)
    order = list(signatures)

    def score(table: str) -> int:
        sig_tokens = _tokens(signatures[table])
        return sum(any(q in s or s in q for s in sig_tokens) for q in q_tokens)

    return sorted(order, key=lambda t: (-score(t), order.index(t)))


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _embedding_ranking(
    question: str, signatures: dict[str, str], client: LLMClient
) -> list[str] | None:
    """Order tables by embedding cosine to the question, or None if embeddings fail.

    A degraded embedding backend must never break Text-to-SQL — we just fall back
    to the lexical ranking, so this swallows errors and returns None.
    """
    names = list(signatures)
    try:
        vectors = client.embed([question] + [signatures[n] for n in names])
    except Exception:  # noqa: BLE001 - embeddings are best-effort here
        return None
    q_vec, table_vecs = vectors[0], vectors[1:]
    scored = sorted(
        zip(names, table_vecs, strict=True),
        key=lambda nv: _cosine(q_vec, nv[1]),
        reverse=True,
    )
    return [name for name, _ in scored]


def _rrf(rankings: list[list[str]], k: int) -> list[str]:
    """Fuse name rankings by Reciprocal Rank Fusion (rank-based, scale-free)."""
    scores: dict[str, float] = {}
    seen_at: dict[str, int] = {}
    for ranking in rankings:
        for rank, name in enumerate(ranking, start=1):
            scores[name] = scores.get(name, 0.0) + 1.0 / (k + rank)
            seen_at.setdefault(name, len(seen_at))
    return sorted(scores, key=lambda n: (-scores[n], seen_at[n]))


def select_tables(
    question: str,
    signatures: dict[str, str],
    settings: Settings,
    embed_client: LLMClient | None,
) -> list[str]:
    """Return the top-K most relevant table names for `question`."""
    top_k = settings.sql_kb_schema_top_k
    rankings = [_lexical_ranking(question, signatures)]
    if embed_client is not None:
        embedding = _embedding_ranking(question, signatures, embed_client)
        if embedding is not None:
            rankings.append(embedding)
    fused = _rrf(rankings, settings.rrf_k) if len(rankings) > 1 else rankings[0]
    return fused[:top_k]


def select_schema(
    question: str,
    db_path: str | Path,
    settings: Settings | None = None,
    embed_client: LLMClient | None = None,
) -> str:
    """Schema description for the prompt, narrowed to the tables `question` needs.

    Below the `sql_kb_schema_top_k` threshold (or with it set to 0) every table is
    described — identical to the old behaviour. Above it, only the retrieved tables
    are. The embedding client defaults to the shared bge-m3 client and is built
    lazily, so a small schema never touches the embedding backend.
    """
    settings = settings or get_settings()
    top_k = settings.sql_kb_schema_top_k
    signatures = schema.table_signatures(db_path, settings)

    if top_k <= 0 or len(signatures) <= top_k:
        return schema.describe_schema(db_path, settings)

    if embed_client is None:
        from sourcerer.llm.client import get_embedding_client

        embed_client = get_embedding_client(settings)

    tables = select_tables(question, signatures, settings, embed_client)
    return schema.describe_schema(db_path, settings, tables)
