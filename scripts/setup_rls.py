"""Provision the Postgres row-level security backstop (Phase 10d, defense-in-depth).

Run once (as the owner/superuser) to:
  1. define `governance_rank(level)` — the classification lattice in SQL, matching
     catalog.classification_rank (unknown → 99, fail-safe);
  2. create a restricted, non-superuser reader role (subject to RLS);
  3. enable RLS on `chunks` and install a policy that *recomputes* the access rule
     in the database from `asset_catalog` + `principals` + the `sourcerer.principal`
     session setting — the same rule as policy.can_read.

Because the policy is evaluated by Postgres for the reader role, a bug in the
app-level gate cannot widen what the database returns: the reader only ever sees
rows the principal is allowed to read. The owner role (used for ingest/admin) is a
superuser and bypasses RLS, so the no-governance path is unaffected.

Idempotent — safe to re-run.

    python scripts/setup_rls.py
"""

from __future__ import annotations

from psycopg import sql

from sourcerer.config import get_settings
from sourcerer.db.session import connect, init_schema
from sourcerer.governance.catalog import init_catalog_schema
from sourcerer.governance.principal import init_principal_schema


def main() -> None:
    settings = get_settings()
    init_schema()  # chunks + query_log
    init_catalog_schema()  # asset_catalog
    init_principal_schema()  # principals
    reader = settings.rls_reader_user
    default_class = settings.governance_default_classification

    with connect() as conn:
        # 1) The classification lattice as a SQL function (parity with Python).
        conn.execute("""
            CREATE OR REPLACE FUNCTION governance_rank(level text) RETURNS int AS $$
              SELECT CASE level
                WHEN 'public' THEN 0
                WHEN 'internal' THEN 1
                WHEN 'confidential' THEN 2
                WHEN 'restricted' THEN 3
                ELSE 99           -- unknown → most-restrictive (fail-safe)
              END;
            $$ LANGUAGE sql IMMUTABLE;
            """)

        # 2) Restricted reader role (created once; password set below where %s binds —
        #    a placeholder can't bind inside a DO block body).
        conn.execute(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{reader}') THEN
                    CREATE ROLE {reader} LOGIN NOSUPERUSER NOBYPASSRLS;
                END IF;
            END
            $$;
            """)
        # PASSWORD takes a literal, not a bind param — compose it safely.
        conn.execute(
            sql.SQL("ALTER ROLE {} WITH PASSWORD {}").format(
                sql.Identifier(reader), sql.Literal(settings.rls_reader_password)
            )
        )
        conn.execute(f"GRANT USAGE ON SCHEMA public TO {reader}")
        conn.execute(f"GRANT SELECT ON chunks, asset_catalog, principals TO {reader}")

        # 3) Enable RLS on chunks + the policy that recomputes policy.can_read. The
        #    reader (non-owner, non-superuser) is subject to it; the owner bypasses.
        conn.execute("ALTER TABLE chunks ENABLE ROW LEVEL SECURITY")
        conn.execute("DROP POLICY IF EXISTS chunks_reader_acl ON chunks")
        # DDL can't bind params — inline the default classification as a literal.
        conn.execute(sql.SQL("""
            CREATE POLICY chunks_reader_acl ON chunks FOR SELECT
            USING (
                current_setting('sourcerer.principal', true) IS NOT NULL
                AND current_setting('sourcerer.principal', true) <> ''
                AND EXISTS (
                    SELECT 1 FROM principals p
                    LEFT JOIN asset_catalog a
                        ON a.kind = 'document' AND a.asset_id = chunks.source
                    WHERE p.id = current_setting('sourcerer.principal', true)
                      AND (
                        'admin' = ANY(p.roles)
                        OR governance_rank(p.clearance)
                             >= governance_rank(COALESCE(a.classification, {default}))
                        OR (a.owner_team IS NOT NULL AND a.owner_team = ANY(p.teams))
                      )
                )
            )
            """).format(default=sql.Literal(default_class)))

    print(f"RLS provisioned: reader role '{reader}', policy chunks_reader_acl on chunks.")
    print("Enable the backstop with GOVERNANCE_RLS_ENABLED=true (+ GOVERNANCE_ENABLED=true).")


if __name__ == "__main__":
    main()
