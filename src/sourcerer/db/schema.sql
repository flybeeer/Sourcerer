-- Runs once on first Postgres init (mounted into docker-entrypoint-initdb.d).
-- Only enables the pgvector extension here; the application creates the tables
-- via db.session.init_schema() so the embedding dimension can come from config.
CREATE EXTENSION IF NOT EXISTS vector;
