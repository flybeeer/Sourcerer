"""Tests for the governance catalog, principal, and gate logic (no I/O).

DB-backed CRUD is exercised against a live Postgres elsewhere; here we cover the
pure policy math, the fail-closed identity resolution (10a), the access policy +
PDP selection the document gate uses (10b), and the GraphRAG community gate (10d).
"""

from sourcerer import governance
from sourcerer.config import Settings
from sourcerer.governance import gate as gate_mod
from sourcerer.governance import principal as principal_mod
from sourcerer.governance.catalog import (
    Asset,
    classification_rank,
    parse_pii_tags,
)
from sourcerer.governance.gate import CerbosPDP, LocalPDP, get_pdp
from sourcerer.governance.policy import can_read
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


# ---------- access policy (Phase 10b) ----------

_CONFIDENTIAL_HR = Asset(
    kind="document", id="handbook.md", classification="confidential", owner_team="hr"
)
_PUBLIC = Asset(kind="document", id="faq.md", classification="public")
_RESTRICTED = Asset(kind="document", id="board.md", classification="restricted")


def test_clearance_at_or_above_classification_allows():
    alice = Principal(id="alice", clearance="confidential")
    assert can_read(alice, _CONFIDENTIAL_HR)
    assert can_read(alice, _PUBLIC)


def test_lower_clearance_without_team_is_denied():
    alice = Principal(id="alice", clearance="internal", teams=["ops"])
    assert not can_read(alice, _CONFIDENTIAL_HR)


def test_owner_team_overrides_low_clearance():
    # Low clearance but on the owning team → may read its own team's docs.
    hr_junior = Principal(id="dave", clearance="internal", teams=["hr"])
    assert can_read(hr_junior, _CONFIDENTIAL_HR)


def test_admin_role_reads_anything():
    carol = Principal(id="carol", roles=["admin"], clearance="internal", teams=[])
    assert can_read(carol, _RESTRICTED)


def test_anonymous_sees_only_public():
    assert can_read(ANONYMOUS, _PUBLIC)
    assert not can_read(ANONYMOUS, _CONFIDENTIAL_HR)
    assert not can_read(ANONYMOUS, _RESTRICTED)


# ---------- local PDP + selection ----------


def test_local_pdp_filters_to_readable_ids():
    assets = [_PUBLIC, _CONFIDENTIAL_HR, _RESTRICTED]
    alice = Principal(id="alice", clearance="confidential")
    assert LocalPDP().readable(alice, assets) == {"faq.md", "handbook.md"}


def test_get_pdp_selects_backend():
    assert isinstance(get_pdp(_settings(governance_pdp="local")), LocalPDP)
    assert isinstance(get_pdp(_settings(governance_pdp="cerbos")), CerbosPDP)


def test_get_pdp_rejects_unknown_backend():
    import pytest

    with pytest.raises(ValueError):
        get_pdp(_settings(governance_pdp="opa"))


# ---------- allowed_document_sources (gate planner, corpus/catalog stubbed) ----------


def test_allowed_sources_none_when_governance_off(monkeypatch):
    settings = _settings(governance_enabled=False)
    assert gate_mod.allowed_document_sources(Principal(id="x"), settings) is None


def test_allowed_sources_none_when_no_principal():
    settings = _settings(governance_enabled=True)
    assert gate_mod.allowed_document_sources(None, settings) is None


def test_allowed_sources_excludes_forbidden_and_defaults_untagged(monkeypatch):
    # Corpus has 3 sources; only handbook.md is tagged (confidential/hr). The
    # other two are untagged → default public. Anonymous sees only the public two.
    monkeypatch.setattr(
        gate_mod.corpus,
        "list_sources",
        lambda: [{"source": "handbook.md"}, {"source": "faq.md"}, {"source": "notes.md"}],
    )
    monkeypatch.setattr(gate_mod.catalog, "list_assets", lambda kind: [_CONFIDENTIAL_HR])
    settings = _settings(governance_enabled=True, governance_default_classification="public")

    allowed = gate_mod.allowed_document_sources(ANONYMOUS, settings)
    assert allowed == {"faq.md", "notes.md"}  # handbook.md (confidential) excluded

    bob = Principal(id="bob", clearance="public", teams=["hr"])  # on the hr team
    assert gate_mod.allowed_document_sources(bob, settings) == {"handbook.md", "faq.md", "notes.md"}


# ---------- GraphRAG community gate (Phase 10d) ----------


class _Comm:
    """Minimal stand-in for a GraphRAG Community (duck-typed on .sources)."""

    def __init__(self, sources):
        self.sources = sources


def _stub_corpus(monkeypatch, sources, tagged):
    monkeypatch.setattr(gate_mod.corpus, "list_sources", lambda: [{"source": s} for s in sources])
    monkeypatch.setattr(gate_mod.catalog, "list_assets", lambda kind: tagged)


def test_communities_unfiltered_when_governance_off():
    comms = [_Comm(["a.md"]), _Comm(["secret.md"])]
    kept, denied = gate_mod.filter_readable_communities(
        comms, ANONYMOUS, _settings(governance_enabled=False)
    )
    assert kept == comms and denied == 0


def test_community_dropped_when_any_source_forbidden(monkeypatch):
    # board.md is restricted; a community touching it is hidden from anonymous even
    # though its other source (faq.md) is public — a summary can fuse both.
    _stub_corpus(
        monkeypatch,
        ["faq.md", "board.md"],
        [Asset(kind="document", id="board.md", classification="restricted")],
    )
    settings = _settings(governance_enabled=True, governance_default_classification="public")
    comms = [_Comm(["faq.md"]), _Comm(["faq.md", "board.md"]), _Comm(["board.md"])]

    kept, denied = gate_mod.filter_readable_communities(comms, ANONYMOUS, settings)
    assert [c.sources for c in kept] == [["faq.md"]]  # only the all-public community
    assert denied == 2
