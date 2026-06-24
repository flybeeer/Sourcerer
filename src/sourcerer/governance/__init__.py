"""Data governance (Phase 10) — authorize at retrieval/execution time.

10a lays the foundation: an `asset_catalog` of governance metadata and a
`principal` identity on every request. Nothing is enforced yet; the Cerbos gate
(SQLGlot parse → CheckResources, PlanResources→WHERE) arrives from 10b on.
"""

from __future__ import annotations

from sourcerer.config import Settings
from sourcerer.governance import catalog, principal
from sourcerer.governance.principal import ANONYMOUS, Principal

__all__ = ["catalog", "principal", "Principal", "ANONYMOUS", "resolve_principal"]


def resolve_principal(header_value: str | None, settings: Settings) -> Principal | None:
    """Resolve the request's principal, or None when governance is disabled.

    Returns None (governance off) so callers can record/enforce only when
    `GOVERNANCE_ENABLED`; with it on, an absent/unknown header resolves to the
    least-privileged `ANONYMOUS` principal (fail-closed).
    """
    if not settings.governance_enabled:
        return None
    return principal.resolve(header_value)
