"""The access policy, written explicitly (Phase 10b).

This is the reference implementation of Sourcerer's authorization rules — the
single source of truth the local PDP evaluates and the Cerbos YAML mirrors.
Keeping the rule as a plain, tested Python function (the project's "explicit
first, framework second" ethos) means the policy is reviewable and the
externalised Cerbos policy can be checked for parity against it.

Rule (RBAC + ABAC, per the Phase 10 blueprint):

    ALLOW read  if  principal is an admin
                OR  principal.clearance >= asset.classification   (clearance lattice)
                OR  asset.owner_team in principal.teams           (team ownership)

`admin` is the one role that shortcuts the attribute checks; everything else is
attribute-based (clearance/team). The lattice ordering lives in catalog.py so
ranks are defined once.
"""

from __future__ import annotations

from sourcerer.governance.catalog import Asset, classification_rank
from sourcerer.governance.principal import Principal

# A role that may read any asset regardless of classification/ownership.
ADMIN_ROLE = "admin"


def can_read(principal: Principal, asset: Asset) -> bool:
    """Return True if `principal` may read `asset` under the policy above."""
    if ADMIN_ROLE in principal.roles:
        return True
    if principal.clearance_rank >= classification_rank(asset.classification):
        return True
    if asset.owner_team and asset.owner_team in principal.teams:
        return True
    return False


def readable_assets(principal: Principal, assets: list[Asset]) -> set[str]:
    """Return the ids of the assets `principal` may read (reference filter)."""
    return {a.id for a in assets if can_read(principal, a)}
