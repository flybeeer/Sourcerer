"""Pure-logic tests for the ingest-time quality gate (ingestion/quality.py).

metadata_issues must stay in parity with the SQL metadata checks in
scripts/dq_report.py — same six checks, same missing-field semantics.
"""

from datetime import UTC, datetime

from sourcerer.ingestion.quality import metadata_issues

_NOW = datetime(2026, 7, 2, tzinfo=UTC)


def _fresh_page(**overrides) -> dict:
    """A page that passes every check; override fields to trip specific ones."""
    page = {
        "last_modified": "2026-01-01T00:00:00+00:00",
        "last_modified_by": "Alice",
        "labels": ["qa"],
        "version": 3,
        "ancestors": [{"id": "1", "title": "Home"}],
        "status": "current",
    }
    page.update(overrides)
    return page


def test_clean_page_has_no_issues():
    assert metadata_issues(_fresh_page(), now=_NOW) == []


def test_stale_page_flagged_beyond_window():
    page = _fresh_page(last_modified="2020-03-24T17:24:00.000Z")
    assert "stale" in metadata_issues(page, now=_NOW)


def test_stale_window_configurable():
    page = _fresh_page(last_modified="2025-06-01T00:00:00+00:00")  # ~13 months before _NOW
    assert "stale" not in metadata_issues(page, stale_years=2.0, now=_NOW)
    assert "stale" in metadata_issues(page, stale_years=1.0, now=_NOW)


def test_missing_last_modified_cannot_flag_stale():
    page = _fresh_page(last_modified=None)
    assert "stale" not in metadata_issues(page, now=_NOW)


def test_orphaned_owner_deleted_and_unlicensed():
    deleted = _fresh_page(last_modified_by="ผู้ใช้เดิม (Deleted)")
    unlicensed = _fresh_page(last_modified_by="Somchai (Unlicensed)")
    active = _fresh_page(last_modified_by="Alice")
    assert "orphaned_owner" in metadata_issues(deleted, now=_NOW)
    assert "orphaned_owner" in metadata_issues(unlicensed, now=_NOW)
    assert "orphaned_owner" not in metadata_issues(active, now=_NOW)


def test_missing_owner_cannot_flag_orphaned():
    page = _fresh_page(last_modified_by=None)
    assert "orphaned_owner" not in metadata_issues(page, now=_NOW)


def test_unlabeled_on_empty_or_missing_labels():
    assert "unlabeled" in metadata_issues(_fresh_page(labels=[]), now=_NOW)
    assert "unlabeled" in metadata_issues(_fresh_page(labels=None), now=_NOW)
    assert "unlabeled" not in metadata_issues(_fresh_page(), now=_NOW)


def test_never_reviewed_on_version_one_or_missing():
    assert "never_reviewed" in metadata_issues(_fresh_page(version=1), now=_NOW)
    assert "never_reviewed" in metadata_issues(_fresh_page(version=None), now=_NOW)
    assert "never_reviewed" not in metadata_issues(_fresh_page(version=2), now=_NOW)


def test_orphan_page_on_empty_ancestors():
    assert "orphan_page" in metadata_issues(_fresh_page(ancestors=[]), now=_NOW)
    assert "orphan_page" not in metadata_issues(_fresh_page(), now=_NOW)


def test_bad_status_on_non_current():
    assert "bad_status" in metadata_issues(_fresh_page(status="draft"), now=_NOW)
    assert "bad_status" not in metadata_issues(_fresh_page(status=None), now=_NOW)


def test_real_qa_corpus_shape_scores_three_issues():
    # The dominant shape in the live QA-space batch: stale + orphaned owner +
    # unlabeled, but versioned, parented, and current → 3 issues.
    page = _fresh_page(
        last_modified="2019-11-05T21:14:03.000Z",
        last_modified_by="Sutipa Kerdchue [X] (Unlicensed)",
        labels=[],
    )
    assert metadata_issues(page, now=_NOW) == ["stale", "orphaned_owner", "unlabeled"]
