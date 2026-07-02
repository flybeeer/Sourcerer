"""Pure-logic tests for scripts/dq_report.py — no DB, no LLM.

SQL-touching functions (_metadata_checks, _word_counts, _mean_embeddings,
_content_by_source) need a live Postgres and aren't unit tested here, same as
other DB-fetching functions in this codebase; they were verified against the
real ingested QA-space corpus.
"""

import numpy as np
import pytest

from scripts.dq_report import (
    DocScore,
    _duplicate_pairs,
    _markup_residue,
    _parse_judge_score,
    _staleness_marker,
    render_report,
)


def test_docscore_issue_count_sums_boolean_flags():
    score = DocScore(source="s", title="t", stale=True, unlabeled=True)
    assert score.issue_count == 2


def test_docscore_issue_count_counts_duplicate_as_one_regardless_of_matches():
    score = DocScore(source="s", title="t", duplicate_of=["a", "b", "c"])
    assert score.issue_count == 1


def test_docscore_issue_count_zero_when_clean():
    assert DocScore(source="s", title="t").issue_count == 0


def test_markup_residue_detects_leftover_tag():
    assert _markup_residue("some text <div class='x'> more text")


def test_markup_residue_detects_leftover_entity():
    assert _markup_residue("Q&amp;A session notes")


def test_markup_residue_false_on_clean_text():
    assert not _markup_residue("Plain prose with no markup residue at all.")


def test_staleness_marker_case_insensitive():
    assert _staleness_marker("this step is Deprecated, use the new flow")
    assert _staleness_marker("TODO: update this section")
    assert not _staleness_marker("This process is fully documented and current.")


def test_parse_judge_score_extracts_and_clamps():
    assert _parse_judge_score("0.8") == 0.8
    assert _parse_judge_score("I'd say 1.5 out of 1") == 1.0
    assert _parse_judge_score("score: 3") == 1.0


def test_parse_judge_score_falls_back_to_neutral_when_unparseable():
    assert _parse_judge_score("I cannot rate this document.") == 0.5


def test_duplicate_pairs_flags_similar_vectors_both_ways():
    embeddings = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([1.0, 0.001]),  # ~cosine 1.0, near-duplicate of a
        "c": np.array([0.0, 1.0]),  # orthogonal, not a duplicate of anything
    }
    dup_of = _duplicate_pairs(embeddings, threshold=0.95)
    assert dup_of["a"] == ["b"]
    assert dup_of["b"] == ["a"]
    assert dup_of["c"] == []


def test_duplicate_pairs_respects_threshold():
    embeddings = {"a": np.array([1.0, 0.0]), "b": np.array([0.5, 0.5])}
    assert _duplicate_pairs(embeddings, threshold=0.99) == {"a": [], "b": []}


def test_render_report_empty():
    assert render_report([], with_llm=False) == "No documents to report on."


def test_render_report_includes_source_and_summary_counts():
    scores = [
        DocScore(source="confluence:1", title="Stale doc", stale=True, unlabeled=True),
        DocScore(source="confluence:2", title="Clean doc"),
    ]
    report = render_report(scores, with_llm=False)
    assert "confluence:1" in report
    assert "confluence:2" in report
    assert "2 document(s):" in report
    assert "S stale" in report
    assert "avg issues/document: 1.0" in report


def test_render_report_with_llm_includes_score_column_and_average():
    scores = [
        DocScore(source="confluence:1", title="A", llm_score=0.8),
        DocScore(source="confluence:2", title="B", llm_score=0.4),
    ]
    report = render_report(scores, with_llm=True)
    assert "llm" in report
    assert "avg llm quality score: 0.60" in report


def test_render_report_sorts_worst_first():
    clean = DocScore(source="confluence:clean", title="Clean")
    bad = DocScore(source="confluence:bad", title="Bad", stale=True, unlabeled=True, stub=True)
    report = render_report([clean, bad], with_llm=False)
    assert report.index("confluence:bad") < report.index("confluence:clean")


@pytest.mark.parametrize("text", ["", "no markup or entities here"])
def test_markup_residue_various_clean_inputs(text):
    assert not _markup_residue(text)
