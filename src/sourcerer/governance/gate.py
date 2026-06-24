"""Governance gate — decide which assets a principal may read (Phase 10b).

The gate authorizes at *retrieval time*: it resolves the set of document sources
a principal is allowed to read, and the caller pushes that set down into the
vector/BM25 SQL as `WHERE source = ANY(:allowed)` — so forbidden chunks never
enter the candidate set (let alone the LLM's context).

The decision is delegated to a Policy Decision Point (PDP), swappable via
`GOVERNANCE_PDP` (mirrors LOCAL_BACKEND / SQL_KB_BACKEND):

  local  — the reference policy (policy.can_read) evaluated in-process. No infra;
           this is what the eval + tests exercise.
  cerbos — the externalised Cerbos PDP sidecar over the same policy, expressed as
           versioned YAML. Production path; needs the [governance] extra + sidecar.

Both return the *same* allowed-source set for the same inputs (parity is the
point of keeping policy.py as the written reference).
"""

from __future__ import annotations

from typing import Protocol

from sourcerer import corpus
from sourcerer.config import Settings
from sourcerer.governance import catalog, policy
from sourcerer.governance.catalog import Asset
from sourcerer.governance.principal import Principal


class PDP(Protocol):
    """A policy decision point: which of these assets may the principal read?"""

    def readable(self, principal: Principal, assets: list[Asset]) -> set[str]: ...


class LocalPDP:
    """Reference PDP — evaluates policy.can_read in-process (no infra)."""

    def readable(self, principal: Principal, assets: list[Asset]) -> set[str]:
        return policy.readable_assets(principal, assets)


class CerbosPDP:
    """Production PDP — delegates the decision to a Cerbos sidecar.

    Batches the candidate assets into one CheckResources call; returns the ids
    Cerbos allows the `read` action on. The YAML policy under `policies/` mirrors
    policy.can_read, so this yields the same set as LocalPDP.
    """

    def __init__(self, endpoint: str):
        self._endpoint = endpoint

    def readable(self, principal: Principal, assets: list[Asset]) -> set[str]:
        # Lazy import: the cerbos SDK is the [governance] extra, only needed here.
        from cerbos.sdk.client import CerbosClient
        from cerbos.sdk.model import Principal as CPrincipal
        from cerbos.sdk.model import Resource, ResourceList

        cprincipal = CPrincipal(
            principal.id,
            roles=principal.roles or ["user"],
            attr=principal.attr(),
        )
        resources = ResourceList()
        for a in assets:
            # a.kind ("document" | "table" | "column") = the Cerbos resource kind,
            # so one PDP governs all three retrieval paths.
            resources.add(Resource(a.id, a.kind, attr=a.attr()), actions={"read"})

        with CerbosClient(self._endpoint) as client:
            resp = client.check_resources(cprincipal, resources)
        return {
            a.id
            for a in assets
            if resp.get_resource(a.id, lenient=True) and resp.get_resource(a.id).is_allowed("read")
        }


def get_pdp(settings: Settings) -> PDP:
    """Return the configured PDP (local reference or Cerbos sidecar)."""
    backend = settings.governance_pdp.lower()
    if backend == "local":
        return LocalPDP()
    if backend == "cerbos":
        return CerbosPDP(settings.cerbos_endpoint)
    raise ValueError(f"Unknown GOVERNANCE_PDP: {settings.governance_pdp!r} (local | cerbos)")


def _document_assets(settings: Settings) -> list[Asset]:
    """Build an Asset for every source in the corpus (catalogued or defaulted).

    Untagged sources take GOVERNANCE_DEFAULT_CLASSIFICATION (default public), so
    enabling governance only restricts what you've explicitly tagged.
    """
    tagged = {a.id: a for a in catalog.list_assets("document")}
    default = settings.governance_default_classification
    return [
        tagged.get(s["source"], Asset(kind="document", id=s["source"], classification=default))
        for s in corpus.list_sources()
    ]


def allowed_document_sources(principal: Principal | None, settings: Settings) -> set[str] | None:
    """Sources `principal` may read, or None when governance is disabled.

    None = no filter (governance off → unchanged behaviour). A set (possibly
    empty) = restrict retrieval to exactly those sources; an empty set correctly
    yields no results rather than "no filter".
    """
    if not settings.governance_enabled or principal is None:
        return None
    return get_pdp(settings).readable(principal, _document_assets(settings))
