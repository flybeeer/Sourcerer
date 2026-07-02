"""Data-quality report over the ingested Confluence corpus.

See docs/confluence-data-quality.md for the full design. Two layers:

  Stage 1 (default, SQL/numpy only, no LLM call): metadata checks over
  `document_metadata` (staleness, orphaned ownership, unlabeled,
  never-reviewed, orphan page, bad status) + content checks over `chunks`
  that need no model call — stub detection by word count, and near-duplicate
  detection via the embeddings already sitting in pgvector (no re-embedding).

  Stage 2 (--with-llm): adds two regex content checks (residual HTML
  tags/entities that `storage_to_text` failed to strip; staleness markers
  like TODO/deprecated in the body) plus an LLM-as-judge clarity/currency
  score, using the same "ask for a single number" pattern as the Phase 3
  eval judge (src/sourcerer/eval/generation_metrics.py).

  Note on the markup-residue check vs. the design doc's original sketch:
  chunk storage is whitespace-joined (chunking.chunk_text does `text.split()`
  then `" ".join(...)`), so newlines never survive into `chunks.content` —
  a line-length heuristic can't see them. Residual tags/entities don't
  depend on newlines and directly catch extraction bugs, so that's the
  Stage 2 markup check implemented here instead.

Scope: only sources with a `document_metadata` row are considered — today
that's Confluence pages exclusively (directory/SQL ingestion don't populate
per-document metadata), matching the design doc's title.

By default this is read-only. `--persist` writes each document's dq object
into `document_metadata.metadata->'dq'`, queryable via plain SQL (DBeaver
etc.) without re-running the report. Each run fully replaces the prior 'dq'
object — a --persist run without --with-llm drops the Stage 2 fields rather
than merging, so the stored dq always matches exactly what that run printed.

Usage:
    python scripts/dq_report.py
    python scripts/dq_report.py --with-llm
    python scripts/dq_report.py --with-llm --persist
    python scripts/dq_report.py --stale-years 1 --dup-threshold 0.9 --min-words 30
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
import psycopg

from sourcerer.config import get_settings
from sourcerer.db.session import connect

logger = logging.getLogger(__name__)

_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")
_STALENESS_MARKER_RE = re.compile(r"\b(TODO|TBD|deprecated|obsolete|outdated)\b", re.IGNORECASE)
_RESIDUAL_TAG_RE = re.compile(r"</?[a-zA-Z][^>]{0,30}>")
_RESIDUAL_ENTITY_RE = re.compile(r"&[a-zA-Z]+;|&#\d+;")

_JUDGE_PROMPT = """You are a strict documentation reviewer. Read the DOCUMENT below and rate its
overall quality as an internal knowledge-base source, considering clarity, whether it is
self-contained (understandable without extra context), and whether it reads as still current
(not obviously superseded or abandoned).

- 1.0 = clear, self-contained, reads as current and trustworthy.
- 0.0 = confusing, missing context, or reads as clearly outdated/abandoned.

DOCUMENT:
{content}

Respond with ONLY a number between 0 and 1 (e.g. 0.0, 0.5, 1.0)."""

_JUDGE_CONTENT_CHARS = 6000  # cap prompt size for large multi-chunk documents


@dataclass
class DocScore:
    """One document's data-quality flags. `words`/`llm_score` are informational,
    not counted issues on their own — `stub`/`markup_residue`/etc. carry the verdicts."""

    source: str
    title: str
    stale: bool = False
    orphaned_owner: bool = False
    unlabeled: bool = False
    never_reviewed: bool = False
    orphan_page: bool = False
    bad_status: bool = False
    stub: bool = False
    words: int = 0
    duplicate_of: list[str] = field(default_factory=list)
    markup_residue: bool = False
    staleness_marker: bool = False
    llm_score: float | None = None

    @property
    def issue_count(self) -> int:
        return sum(
            [
                self.stale,
                self.orphaned_owner,
                self.unlabeled,
                self.never_reviewed,
                self.orphan_page,
                self.bad_status,
                self.stub,
                bool(self.duplicate_of),
                self.markup_residue,
                self.staleness_marker,
            ]
        )


# --- Stage 1: metadata + cheap content checks --------------------------------


def _metadata_checks(conn: psycopg.Connection, stale_years: float) -> dict[str, DocScore]:
    """One DocScore per source with a `document_metadata` row, all six metadata flags set."""
    rows = conn.execute(
        """
        SELECT source,
               coalesce(metadata->>'title', source) AS title,
               (metadata->>'last_modified')::timestamptz
                 < now() - (%s * interval '1 year')                  AS stale,
               metadata->>'last_modified_by' ~* '\\(deleted\\)|\\(unlicensed\\)' AS orphaned_owner,
               coalesce(jsonb_array_length(metadata->'labels'), 0) = 0    AS unlabeled,
               coalesce((metadata->>'version')::int, 1) = 1              AS never_reviewed,
               coalesce(jsonb_array_length(metadata->'ancestors'), 0) = 0 AS orphan_page,
               coalesce(metadata->>'status', 'current') <> 'current'     AS bad_status
        FROM document_metadata
        ORDER BY source
        """,
        (stale_years,),
    ).fetchall()
    scores = {}
    for row in rows:
        (
            source,
            title,
            stale,
            orphaned_owner,
            unlabeled,
            never_reviewed,
            orphan_page,
            bad_status,
        ) = row
        scores[source] = DocScore(
            source=source,
            title=title,
            stale=bool(stale),
            orphaned_owner=bool(orphaned_owner),
            unlabeled=bool(unlabeled),
            never_reviewed=bool(never_reviewed),
            orphan_page=bool(orphan_page),
            bad_status=bool(bad_status),
        )
    return scores


def _word_counts(conn: psycopg.Connection, sources: set[str]) -> dict[str, int]:
    """Total words per source across its chunks — cheap stub-page detector."""
    rows = conn.execute(
        """
        SELECT source, sum(array_length(regexp_split_to_array(trim(content), '\\s+'), 1))
        FROM chunks
        WHERE source = ANY(%s)
        GROUP BY source
        """,
        (list(sources),),
    ).fetchall()
    return {source: (words or 0) for source, words in rows}


def _mean_embeddings(conn: psycopg.Connection, sources: set[str]) -> dict[str, np.ndarray]:
    """Per-source mean chunk embedding — reuses embeddings already in pgvector."""
    rows = conn.execute(
        "SELECT source, embedding FROM chunks WHERE source = ANY(%s)",
        (list(sources),),
    ).fetchall()
    by_source: dict[str, list[np.ndarray]] = {}
    for source, embedding in rows:
        by_source.setdefault(source, []).append(np.asarray(embedding, dtype=float))
    return {source: np.mean(vectors, axis=0) for source, vectors in by_source.items()}


def _duplicate_pairs(embeddings: dict[str, np.ndarray], threshold: float) -> dict[str, list[str]]:
    """Sources whose mean embedding is >= threshold cosine-similar to another source's.

    O(n^2) — fine at the corpus sizes this project keeps while iterating
    (project rule: keep the corpus small before any expensive indexing).
    """
    sources = list(embeddings)
    dup_of: dict[str, list[str]] = {s: [] for s in sources}
    for i in range(len(sources)):
        a = embeddings[sources[i]]
        for j in range(i + 1, len(sources)):
            b = embeddings[sources[j]]
            sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
            if sim >= threshold:
                dup_of[sources[i]].append(sources[j])
                dup_of[sources[j]].append(sources[i])
    return dup_of


# --- Stage 2: regex content checks + LLM judge (--with-llm) -----------------


def _content_by_source(conn: psycopg.Connection, sources: set[str]) -> dict[str, str]:
    rows = conn.execute(
        """
        SELECT source, string_agg(content, ' ' ORDER BY chunk_index)
        FROM chunks
        WHERE source = ANY(%s)
        GROUP BY source
        """,
        (list(sources),),
    ).fetchall()
    return dict(rows)


def _markup_residue(text: str) -> bool:
    return bool(_RESIDUAL_TAG_RE.search(text) or _RESIDUAL_ENTITY_RE.search(text))


def _staleness_marker(text: str) -> bool:
    return bool(_STALENESS_MARKER_RE.search(text))


def _parse_judge_score(reply: str) -> float:
    match = _NUMBER_RE.search(reply)
    if not match:
        return 0.5  # neutral fallback when the judge doesn't return a parseable number
    return max(0.0, min(1.0, float(match.group(1))))


def _judge_score(client, text: str) -> float:
    prompt = _JUDGE_PROMPT.format(content=text[:_JUDGE_CONTENT_CHARS])
    reply = client.chat([{"role": "user", "content": prompt}]).text
    return _parse_judge_score(reply)


# --- Report rendering ---------------------------------------------------------

_FLAG_COLUMNS = [
    ("stale", "S"),
    ("orphaned_owner", "O"),
    ("unlabeled", "U"),
    ("never_reviewed", "N"),
    ("orphan_page", "P"),
    ("bad_status", "B"),
    ("stub", "T"),
]
_LLM_FLAG_COLUMNS = [("markup_residue", "M"), ("staleness_marker", "K")]


def _flags_string(score: DocScore, with_llm: bool) -> str:
    columns = _FLAG_COLUMNS + (_LLM_FLAG_COLUMNS if with_llm else [])
    chars = [letter if getattr(score, attr) else "." for attr, letter in columns]
    chars.append("D" if score.duplicate_of else ".")
    return "".join(chars)


def render_report(scores: list[DocScore], with_llm: bool) -> str:
    if not scores:
        return "No documents to report on."

    ordered = sorted(scores, key=lambda s: (-s.issue_count, s.source))
    legend = (
        "S=stale O=orphaned-owner U=unlabeled N=never-reviewed "
        "P=orphan-page B=bad-status T=stub D=duplicate"
    )
    if with_llm:
        legend += " M=markup-residue K=staleness-marker"

    lines = [legend, ""]
    header = f"{'source':<26} {'title':<40} {'flags':<10} {'words':>6}"
    if with_llm:
        header += f" {'llm':>5}"
    header += f" {'issues':>6}"
    lines.append(header)
    lines.append("-" * len(header))

    for s in ordered:
        title = s.title[:38]
        row = f"{s.source:<26} {title:<40} {_flags_string(s, with_llm):<10} {s.words:>6}"
        if with_llm:
            row += f" {s.llm_score if s.llm_score is not None else float('nan'):>5.2f}"
        row += f" {s.issue_count:>6}"
        lines.append(row)

    n = len(scores)
    lines.append("")
    lines.append(f"{n} document(s):")
    for attr, letter in _FLAG_COLUMNS + (_LLM_FLAG_COLUMNS if with_llm else []):
        count = sum(1 for s in scores if getattr(s, attr))
        lines.append(f"  {letter} {attr:<18} {count:>3}/{n}  ({count / n:.0%})")
    dup_count = sum(1 for s in scores if s.duplicate_of)
    lines.append(f"  D {'duplicate':<18} {dup_count:>3}/{n}  ({dup_count / n:.0%})")
    if with_llm:
        judged = [s.llm_score for s in scores if s.llm_score is not None]
        if judged:
            lines.append(f"  avg llm quality score: {sum(judged) / len(judged):.2f}")
    avg_issues = sum(s.issue_count for s in scores) / n
    lines.append(f"  avg issues/document: {avg_issues:.1f}")
    return "\n".join(lines)


# --- Persistence ---------------------------------------------------------------


def _dq_dict(score: DocScore, with_llm: bool, checked_at: str) -> dict:
    """Shape a DocScore into the object stored at metadata->'dq'.

    Only includes the Stage 2 keys (markup_residue/staleness_marker/llm_score)
    when this run actually computed them — a --persist run without --with-llm
    won't silently wipe those fields with False, it just leaves them out (see
    _persist_scores: each run fully replaces the prior 'dq' object, so a
    Stage-1-only run does drop stale Stage 2 results rather than merge with
    them — the stored dq always reflects exactly what the last run printed).
    """
    d = {
        "checked_at": checked_at,
        "stale": score.stale,
        "orphaned_owner": score.orphaned_owner,
        "unlabeled": score.unlabeled,
        "never_reviewed": score.never_reviewed,
        "orphan_page": score.orphan_page,
        "bad_status": score.bad_status,
        "stub": score.stub,
        "words": score.words,
        "duplicate_of": score.duplicate_of,
    }
    if with_llm:
        d["markup_residue"] = score.markup_residue
        d["staleness_marker"] = score.staleness_marker
        d["llm_score"] = score.llm_score
    d["issue_count"] = score.issue_count
    return d


def _persist_scores(conn: psycopg.Connection, scores: dict[str, DocScore], with_llm: bool) -> None:
    """Write each document's dq object into document_metadata.metadata->'dq'."""
    checked_at = datetime.now(UTC).isoformat()
    with conn.cursor() as cur:
        cur.executemany(
            """
            UPDATE document_metadata
            SET metadata = jsonb_set(metadata, '{dq}', %s::jsonb),
                updated_at = now()
            WHERE source = %s
            """,
            [
                (json.dumps(_dq_dict(score, with_llm, checked_at)), source)
                for source, score in scores.items()
            ],
        )


# --- CLI -----------------------------------------------------------------------


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Data-quality report over the ingested Confluence corpus."
    )
    parser.add_argument(
        "--stale-years",
        type=float,
        default=2.0,
        help="Flag documents not modified within this many years (default: 2).",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=50,
        help="Flag documents with fewer total words as stubs (default: 50).",
    )
    parser.add_argument(
        "--dup-threshold",
        type=float,
        default=0.95,
        help="Cosine similarity above which two documents are flagged as near-duplicates "
        "(default: 0.95).",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Also run markup-residue/staleness-marker regex checks and an LLM quality judge.",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Write the computed dq object back into document_metadata.metadata->'dq' "
        "(queryable via SQL/DBeaver). Off by default — this tool is read-only unless asked.",
    )
    args = parser.parse_args()

    with connect() as conn:
        scores = _metadata_checks(conn, args.stale_years)
        sources = set(scores)
        if not sources:
            print(
                "No documents in document_metadata yet. Ingest some Confluence pages first:\n"
                '  python scripts/ingest_confluence.py "space = ENG and type = page"'
            )
            return

        word_counts = _word_counts(conn, sources)
        for source, words in word_counts.items():
            scores[source].words = words
            scores[source].stub = words < args.min_words

        embeddings = _mean_embeddings(conn, sources)
        for source, dups in _duplicate_pairs(embeddings, args.dup_threshold).items():
            scores[source].duplicate_of = dups

        if args.with_llm:
            from sourcerer.llm.client import get_llm_client

            client = get_llm_client(settings)
            content = _content_by_source(conn, sources)
            for source, text in content.items():
                s = scores[source]
                s.markup_residue = _markup_residue(text)
                s.staleness_marker = _staleness_marker(text)
                s.llm_score = _judge_score(client, text)

        if args.persist:
            _persist_scores(conn, scores, with_llm=args.with_llm)

    print(render_report(list(scores.values()), with_llm=args.with_llm))
    if args.persist:
        print(f"\nPersisted dq scores for {len(scores)} document(s) to document_metadata.")


if __name__ == "__main__":
    main()
