"""SQL knowledge base (Phase 7, optional).

Use a SQLite database as a knowledge source via two independent paths:

- **Ingestion (RAG)** — `loader` pulls rows with a user-written SELECT and turns
  each row into a document for the normal chunk→embed→store pipeline. Good for
  *content/semantic* questions ("what do customers complain about?").
- **Text-to-SQL** — `text_to_sql` answers *analytical/aggregation* questions
  ("total sales last year") by generating a read-only SELECT and executing it.
  Retrieval can never SUM thousands of rows; running real SQL gives exact numbers.

The SQLite file is always opened read-only; generated SQL is validated by `safety`
before it ever touches the database.
"""
