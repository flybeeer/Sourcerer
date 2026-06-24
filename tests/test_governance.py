"""Tests for the governance catalog + principal logic (Phase 10a, no I/O).

DB-backed CRUD is exercised against a live Postgres elsewhere; here we cover the
pure policy math and the fail-closed identity resolution that the Cerbos gate
will build on from 10b.
"""

from sourcerer import governance
from sourcerer.config import Settings
from sourcerer.governance import principal as principal_mod
from sourcerer.governance.catalog import (
    Asset,
    classification_rank,
    parse_pii_tags,
)
from sourcerer.governance.principal import ANONYMOUS, Principal


def _settings(**overrides) -> Settings:
    return Settings(**overrides)


# ---------- classification lattice ----------


def test_classification_order_is_monotonic():
    ranks = [classification_rank(c) for c in ("public", "internal", "confidential", "restricted")]
    assert ranks == sorted(ranks)
    assert ranks == [0, 1, 2, 3]


def test_unknown_classification_sorts_most_restrictive():
    # Fail-safe: a mis-tagged level is treated as more sensitive than any known one.
    assert classification_rank("bogus") > classification_rank("restricted")


def test_principal_clearance_rank_tracks_lattice():
    assert Principal(id="x", clearance="internal").clearance_rank == classification_rank("internal")


# ---------- PII tag parsing ----------


def test_parse_pii_tags_trims_and_drops_blanks():
    assert parse_pii_tags(" email , salary ,, ") == ["email", "salary"]


def test_parse_pii_tags_empty_is_empty_list():
    assert parse_pii_tags(None) == []
    assert parse_pii_tags("") == []


# ---------- attribute bags (shape the Cerbos input from 10b) ----------


def test_asset_attr_bag_carries_governance_fields():
    a = Asset(
        kind="table",
        id="employees",
        classification="confidential",
        owner_team="hr",
        pii_tags=["salary"],
    )
    assert a.attr() == {
        "classification": "confidential",
        "owner_team": "hr",
        "pii_tags": ["salary"],
    }


def test_principal_attr_bag_carries_abac_fields():
    p = Principal(id="alice", roles=["analyst"], clearance="internal", teams=["ops"], region="APAC")
    assert p.attr() == {"clearance": "internal", "teams": ["ops"], "region": "APAC"}


# ---------- fail-closed identity resolution ----------


def test_resolve_principal_none_when_governance_disabled():
    settings = _settings(governance_enabled=False)
    assert governance.resolve_principal("alice", settings) is None


def test_resolve_missing_header_is_anonymous():
    settings = _settings(governance_enabled=True)
    assert governance.resolve_principal(None, settings) is ANONYMOUS
    assert governance.resolve_principal("", settings) is ANONYMOUS


def test_resolve_unknown_principal_is_anonymous(monkeypatch):
    monkeypatch.setattr(principal_mod, "get_principal", lambda _id: None)
    settings = _settings(governance_enabled=True)
    assert governance.resolve_principal("nobody", settings) is ANONYMOUS


def test_resolve_known_principal_returns_it(monkeypatch):
    alice = Principal(id="alice", roles=["analyst"], clearance="internal")
    monkeypatch.setattr(principal_mod, "get_principal", lambda _id: alice)
    settings = _settings(governance_enabled=True)
    assert governance.resolve_principal(" alice ", settings) is alice


def test_resolve_is_fail_closed_when_lookup_errors(monkeypatch):
    # Catalog not provisioned (table missing) → least privilege, never an error.
    def boom(_id):
        raise RuntimeError('relation "principals" does not exist')

    monkeypatch.setattr(principal_mod, "get_principal", boom)
    settings = _settings(governance_enabled=True)
    assert governance.resolve_principal("alice", settings) is ANONYMOUS
