"""PlanResources variant of the document gate — predicate pushdown, not enumeration.

`gate.CerbosPDP.readable` asks Cerbos *CheckResources*: it serialises every
candidate document asset into one request and keeps the ids Cerbos allows. That
is correct and easy to audit, but the payload grows with the corpus — every
cache rebuild (CAG) or query (RAG) ships N resources across the wire.

This module is the O(1)-call alternative. Cerbos *PlanResources* answers a
different question — "given this principal, what *condition* makes a `document`
readable?" — and returns a query plan (an AST over `R.attr.*`) regardless of how
many documents exist. We translate that plan into a SQL `WHERE` over a normalised
catalog view and let Postgres do the matching. One Cerbos round-trip, one SQL
query; cost is independent of corpus size. This is the same predicate-pushdown
idea as the Phase 7 SQL work and `retrieval/filter.source_clause`, taken one
layer up: the *policy itself* becomes the predicate.

Trade-off vs CheckResources: the plan AST must be translated faithfully, so the
translator (`plan_to_sql`) is the part that needs tests against the *live*
Cerbos output for `policies/document.yaml` — the operator set and the exact
attribute paths Cerbos emits depend on the policy and the SDK version. Treat the
mapping below as the contract to verify, not as guaranteed-stable.

Drop-in for CAG: `allowed_document_sources_plan` returns the same
`set[str] | None` shape as `gate.allowed_document_sources`, so `cag.cache`
can call either behind a flag.
"""

from __future__ import annotations

from typing import Any

from sourcerer.config import Settings
from sourcerer.db.session import connect
from sourcerer.governance.principal import Principal

# Cerbos attribute path  →  column in the governed view below. The plan AST
# references resource attributes as e.g. "request.resource.attr.classification";
# we only police the two the document policy reads.
_COLUMN_MAP: dict[str, str] = {
    "request.resource.attr.classification": "classification",
    "request.resource.attr.owner_team": "owner_team",
    # Some SDK versions strip the request prefix; accept both spellings.
    "resource.attr.classification": "classification",
    "resource.attr.owner_team": "owner_team",
}

# Cerbos plan operators  →  SQL. Comparisons + boolean connectives + set
# membership cover what document.yaml emits (lattice compare, team ownership).
_BINARY_OPS = {"eq": "=", "ne": "<>", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}
_BOOL_OPS = {"and": " AND ", "or": " OR "}


# The universe to match against: every corpus source, left-joined to its catalog
# entry, with untagged sources taking GOVERNANCE_DEFAULT_CLASSIFICATION (mirrors
# gate._document_assets). Built as a CTE so the translated predicate slots into
# the outer WHERE.
_GOVERNED_VIEW = """
WITH governed AS (
    SELECT
        c.source                              AS asset_id,
        COALESCE(a.classification, %s)        AS classification,
        a.owner_team                          AS owner_team
    FROM (SELECT DISTINCT source FROM chunks) c
    LEFT JOIN asset_catalog a
        ON a.kind = 'document' AND a.asset_id = c.source
)
SELECT asset_id FROM governed WHERE {predicate}
"""


def allowed_document_sources_plan(
    principal: Principal | None, settings: Settings
) -> set[str] | None:
    """Sources `principal` may read, computed via PlanResources + SQL pushdown.

    Same contract as `gate.allowed_document_sources`: None when governance is off
    (no filter), else the exact readable set (empty set = nothing visible, never
    "no filter"). Fail-closed: any translation/SDK error denies everything.
    """
    if not settings.governance_enabled or principal is None:
        return None

    predicate, params = _resource_plan_predicate("document", principal, settings)
    if predicate == "FALSE":
        return set()  # always-denied → nothing visible

    sql = _GOVERNED_VIEW.format(predicate=predicate)
    bound = (settings.governance_default_classification, *params)
    with connect() as conn:
        rows = conn.execute(sql, bound).fetchall()
    return {r[0] for r in rows}


def _resource_plan_predicate(
    kind: str, principal: Principal, settings: Settings
) -> tuple[str, tuple]:
    """Call Cerbos PlanResources for `read` on `kind`, return (SQL predicate, params).

    ALWAYS_ALLOWED → "TRUE"; ALWAYS_DENIED → "FALSE"; CONDITIONAL → translated AST.
    """
    # Lazy import: the cerbos SDK is the [governance] extra, only needed here.
    from cerbos.sdk.client import CerbosClient
    from cerbos.sdk.model import Principal as CPrincipal
    from cerbos.sdk.model import ResourceDesc

    cprincipal = CPrincipal(
        principal.id,
        roles=principal.roles or ["user"],
        attr=principal.attr(),
    )
    with CerbosClient(settings.cerbos_endpoint) as client:
        plan = client.plan_resources("read", cprincipal, ResourceDesc(kind))

    flt = plan.filter
    kind_str = str(getattr(flt, "kind", "")).upper()
    if "ALWAYS_ALLOWED" in kind_str:
        return "TRUE", ()
    if "ALWAYS_DENIED" in kind_str:
        return "FALSE", ()
    params: list[Any] = []
    return _translate(flt.condition, params), tuple(params)


# --- Plan AST → SQL ---------------------------------------------------------
#
# An operand is one of: an expression (operator + nested operands), a variable
# (an attribute path), or a literal value. The SDK may surface these as proto
# objects or dicts depending on transport, so every accessor tolerates both.


def _translate(operand: Any, params: list) -> str:
    """Recursively translate a Cerbos plan operand into a parameterised SQL fragment."""
    expr = _expression(operand)
    if expr is None:
        raise ValueError(f"expected an expression operand, got: {operand!r}")

    op = _operator(expr)
    operands = _operands(expr)

    if op in _BOOL_OPS:
        parts = [_translate(o, params) for o in operands]
        return "(" + _BOOL_OPS[op].join(parts) + ")"

    if op == "not":
        return f"(NOT {_translate(operands[0], params)})"

    if op in _BINARY_OPS:
        col = _column(operands[0])
        params.append(_literal(operands[1]))
        return f"{col} {_BINARY_OPS[op]} %s"

    if op == "in":
        # `R.attr.owner_team in P.attr.teams` → owner_team = ANY(:teams). The
        # principal side is a constant list baked into the plan at plan time.
        col = _column(operands[0])
        params.append(list(_literal(operands[1])))
        return f"{col} = ANY(%s)"

    raise ValueError(f"unsupported plan operator: {op!r}")


def _expression(operand: Any) -> Any | None:
    """The `.expression` of an operand (operator + operands), or None."""
    if isinstance(operand, dict):
        return operand.get("expression")
    return getattr(operand, "expression", None)


def _operator(expr: Any) -> str:
    return expr["operator"] if isinstance(expr, dict) else expr.operator


def _operands(expr: Any) -> list:
    return list(expr["operands"] if isinstance(expr, dict) else expr.operands)


def _column(operand: Any) -> str:
    """Map a variable operand (attribute path) to its governed-view column."""
    if isinstance(operand, dict):
        name = operand.get("variable")
    else:
        name = getattr(operand, "variable", None)
    if name in _COLUMN_MAP:
        return _COLUMN_MAP[name]
    raise ValueError(f"plan references an ungoverned attribute: {name!r}")


def _literal(operand: Any) -> Any:
    """Extract a literal value operand (the principal-side constant)."""
    if isinstance(operand, dict):
        return operand.get("value")
    return getattr(operand, "value", None)
