"""CLI for the governance catalog + principal directory (Phase 10a).

Records governance metadata — nothing is enforced yet (the Cerbos gate lands in
10b+). Two object types:

  assets      — documents / tables / columns → classification, owner_team, pii_tags
  principals  — stub IdP identities → roles, clearance, teams, region

Usage:
    python scripts/catalog.py set-asset --kind document --id handbook.md \
        --classification confidential --owner-team hr --pii salary,email
    python scripts/catalog.py list-assets
    python scripts/catalog.py set-principal --id alice \
        --roles analyst --clearance internal --teams ops --region APAC
    python scripts/catalog.py list-principals
    python scripts/catalog.py seed-demo
"""

from __future__ import annotations

import argparse
import logging

from sourcerer.config import get_settings
from sourcerer.governance.catalog import (
    Asset,
    list_assets,
    parse_pii_tags,
    upsert_asset,
)
from sourcerer.governance.principal import (
    Principal,
    list_principals,
    upsert_principal,
)


def _csv(raw: str | None) -> list[str]:
    """Parse a comma-separated flag value into a clean list."""
    return [x.strip() for x in raw.split(",")] if raw else []


def _cmd_set_asset(args: argparse.Namespace) -> None:
    upsert_asset(
        Asset(
            kind=args.kind,
            id=args.id,
            classification=args.classification,
            owner_team=args.owner_team,
            pii_tags=parse_pii_tags(args.pii),
            glossary=args.glossary,
        )
    )
    print(f"Catalogued {args.kind}:{args.id} ({args.classification}).")


def _cmd_list_assets(args: argparse.Namespace) -> None:
    assets = list_assets(args.kind)
    if not assets:
        print("(no catalogued assets)")
        return
    for a in assets:
        pii = f" pii={','.join(a.pii_tags)}" if a.pii_tags else ""
        team = f" team={a.owner_team}" if a.owner_team else ""
        print(f"{a.kind:9} {a.id:32} {a.classification}{team}{pii}")


def _cmd_set_principal(args: argparse.Namespace) -> None:
    upsert_principal(
        Principal(
            id=args.id,
            roles=_csv(args.roles),
            clearance=args.clearance,
            teams=_csv(args.teams),
            region=args.region,
        )
    )
    print(f"Saved principal {args.id} (clearance={args.clearance}).")


def _cmd_list_principals(_args: argparse.Namespace) -> None:
    principals = list_principals()
    if not principals:
        print("(no principals)")
        return
    for p in principals:
        roles = ",".join(p.roles) or "-"
        teams = ",".join(p.teams) or "-"
        print(
            f"{p.id:16} roles={roles:16} clearance={p.clearance:12} "
            f"teams={teams:12} region={p.region or '-'}"
        )


def _cmd_seed_demo(_args: argparse.Namespace) -> None:
    """Seed a small demo matrix of principals (assets are tagged at ingest time)."""
    demo = [
        Principal(
            id="alice", roles=["analyst"], clearance="internal", teams=["ops"], region="APAC"
        ),
        Principal(id="bob", roles=["hr"], clearance="confidential", teams=["hr"], region="EMEA"),
        Principal(
            id="carol",
            roles=["admin"],
            clearance="restricted",
            teams=["hr", "finance"],
            region="APAC",
        ),
    ]
    for p in demo:
        upsert_principal(p)
    print(f"Seeded {len(demo)} demo principal(s): {', '.join(p.id for p in demo)}.")


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Manage the governance catalog + principals.")
    sub = parser.add_subparsers(dest="command", required=True)

    sa = sub.add_parser("set-asset", help="Catalogue one asset's governance metadata.")
    sa.add_argument("--kind", required=True, choices=("document", "table", "column"))
    sa.add_argument("--id", required=True, help="Asset id (source / table / table.column).")
    sa.add_argument(
        "--classification",
        default="internal",
        choices=("public", "internal", "confidential", "restricted"),
    )
    sa.add_argument("--owner-team")
    sa.add_argument("--pii", help="Comma-separated PII tags (e.g. email,salary).")
    sa.add_argument("--glossary", help="Optional human description.")
    sa.set_defaults(func=_cmd_set_asset)

    la = sub.add_parser("list-assets", help="List catalogued assets.")
    la.add_argument("--kind", choices=("document", "table", "column"), help="Filter by kind.")
    la.set_defaults(func=_cmd_list_assets)

    sp = sub.add_parser("set-principal", help="Add/update a principal in the stub IdP.")
    sp.add_argument("--id", required=True)
    sp.add_argument("--roles", help="Comma-separated roles (e.g. analyst,admin).")
    sp.add_argument(
        "--clearance",
        default="public",
        choices=("public", "internal", "confidential", "restricted"),
    )
    sp.add_argument("--teams", help="Comma-separated teams.")
    sp.add_argument("--region")
    sp.set_defaults(func=_cmd_set_principal)

    lp = sub.add_parser("list-principals", help="List principals.")
    lp.set_defaults(func=_cmd_list_principals)

    sd = sub.add_parser("seed-demo", help="Seed a small demo set of principals.")
    sd.set_defaults(func=_cmd_seed_demo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
