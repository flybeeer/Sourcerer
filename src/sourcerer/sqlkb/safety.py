"""Validate LLM-generated SQL before it touches the database.

Defence in depth: the connection is already read-only (`connection.connect_ro`),
but we also refuse anything that isn't a single read-only SELECT so a model that
hallucinates a `DELETE` (or smuggles a second statement) never gets that far, and
we always cap the result size.

`safe_select(raw, max_rows)` returns a cleaned, LIMIT-bounded SELECT, or raises
`UnsafeSQLError` with a reason. The caller turns that into an "I don't know".
"""

from __future__ import annotations

import re

# Statement-level keywords that must never appear — any of these means the model
# tried to write, change structure, or reach outside the current database.
_FORBIDDEN = (
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "truncate",
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "reindex",
    "grant",
    "revoke",
)

_FENCE_RE = re.compile(r"^```(?:sql)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_WORD_RE = re.compile(r"[a-z_]+")
# The start of the actual query, so we can shed any prose the model prefixes
# ("Here is the SQL: SELECT …"). The extracted statement is still fully validated.
_SELECT_START_RE = re.compile(r"\b(select|with)\b", re.IGNORECASE)


class UnsafeSQLError(ValueError):
    """Raised when generated SQL is not a single read-only SELECT."""


def _strip(raw: str) -> str:
    """Reduce a chatty model reply to a single candidate statement.

    Small models wrap the query in markdown and/or explain it before *and* after
    the SQL. We (1) drop markdown fences, (2) slice from the first SELECT/WITH so a
    leading sentence is shed, and (3) keep only up to the first ';' so a trailing
    explanation (or a tacked-on second statement) is dropped rather than rejected.
    The single statement that remains still goes through every safety check, and
    the connection is read-only regardless — so dropping extra statements is safe.
    """
    text = _FENCE_RE.sub("", raw).strip()
    match = _SELECT_START_RE.search(text)
    if match:
        text = text[match.start() :]
    return text.split(";", 1)[0].strip()


def safe_select(raw: str, max_rows: int) -> str:
    """Return a validated SELECT bounded by `max_rows`, or raise UnsafeSQLError."""
    sql = _strip(raw)
    if not sql:
        raise UnsafeSQLError("empty SQL")

    lowered = sql.lower()
    first_word = _WORD_RE.search(lowered)
    first = first_word.group(0) if first_word else None
    if first not in ("select", "with"):
        raise UnsafeSQLError(f"not a SELECT (starts with {first!r})")

    # Word-boundary match so columns like `created_at` don't trip "create".
    forbidden = [kw for kw in _FORBIDDEN if re.search(rf"\b{kw}\b", lowered)]
    if forbidden:
        raise UnsafeSQLError(f"forbidden keyword(s): {', '.join(forbidden)}")

    # Enforce a row cap. Respect an existing LIMIT only if it's already tighter.
    limit_match = re.search(r"\blimit\s+(\d+)\s*$", lowered)
    if limit_match:
        if int(limit_match.group(1)) > max_rows:
            sql = re.sub(r"\blimit\s+\d+\s*$", f"LIMIT {max_rows}", sql, flags=re.IGNORECASE)
    else:
        sql = f"{sql} LIMIT {max_rows}"
    return sql
