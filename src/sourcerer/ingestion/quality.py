"""Ingest-time data-quality gate over Confluence page metadata.

`metadata_issues()` is the pure-Python twin of the metadata checks in
scripts/dq_report.py (which runs the same six checks as SQL over
document_metadata *after* ingestion): same rules, same missing-field
semantics, so a page scores identically whether it is checked before ingest
or reported on afterwards.

Used by pipeline.ingest_confluence when `max_issues` is set: a page failing
more than `max_issues` checks never reaches chunk/embed/store — the same
"untrusted content never enters the candidate set" idea as the Phase 10
governance gate, applied to quality instead of authorization. Its metadata
is still stored in full (with the gate verdict), so nothing is lost.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

_ORPHANED_OWNER_RE = re.compile(r"\((deleted|unlicensed)\)", re.IGNORECASE)


def metadata_issues(
    metadata: dict, *, stale_years: float = 2.0, now: datetime | None = None
) -> list[str]:
    """Names of the metadata checks this page fails (empty list = fully trusted).

    Missing-field semantics mirror the SQL in scripts/dq_report.py (where a
    NULL comparison is falsy and coalesce supplies the default): a missing
    `last_modified` or `last_modified_by` cannot flag stale/orphaned_owner,
    while missing `labels`/`version`/`ancestors`/`status` fall back to the
    most-common default and *can* flag their checks.
    """
    now = now or datetime.now(UTC)
    issues: list[str] = []

    last_modified = metadata.get("last_modified")
    if last_modified:
        modified_at = datetime.fromisoformat(last_modified)
        if modified_at < now - timedelta(days=stale_years * 365.25):
            issues.append("stale")
    if _ORPHANED_OWNER_RE.search(metadata.get("last_modified_by") or ""):
        issues.append("orphaned_owner")
    if not metadata.get("labels"):
        issues.append("unlabeled")
    if (metadata.get("version") or 1) == 1:
        issues.append("never_reviewed")
    if not metadata.get("ancestors"):
        issues.append("orphan_page")
    if (metadata.get("status") or "current") != "current":
        issues.append("bad_status")
    return issues
