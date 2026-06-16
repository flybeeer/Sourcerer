"""Prompt construction for grounded, cited answers.

The system prompt enforces the project's hard rule: answer only from the
provided sources, and say "I don't know" rather than guessing.
"""

from __future__ import annotations

from sourcerer.retrieval.types import RetrievedChunk

NO_ANSWER = "I don't know based on the provided documents."

_RULES = [
    "Use ONLY the information in the numbered sources below. Do not use outside knowledge.",
    "Cite the sources you rely on inline using bracketed numbers like [1] or [2].",
    f'If the sources do not contain the answer, reply with exactly: "{NO_ANSWER}"',
    "Be concise and factual.",
]

SYSTEM_PROMPT = (
    "You are Sourcerer, a knowledge assistant that answers strictly "
    "from a set of provided sources.\n\nRules:\n" + "\n".join(f"- {rule}" for rule in _RULES)
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as a numbered source list for the prompt."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] (source: {chunk.source}, chunk {chunk.chunk_index})"
        blocks.append(f"{header}\n{chunk.content}")
    return "\n\n".join(blocks)


def build_messages(query: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """Build the chat messages for a grounded answer."""
    context = format_context(chunks)
    user_content = (
        f"Sources:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the sources above, with inline citations."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


# --- Text-to-SQL (Phase 7) --------------------------------------------------

_SQL_RULES = [
    "Write ONE read-only SQLite SELECT statement that answers the question.",
    "Use ONLY the tables and columns in the schema. Do not invent names.",
    "Never write to the database: no INSERT/UPDATE/DELETE/CREATE/DROP/ALTER/PRAGMA.",
    "Prefer aggregates (SUM, COUNT, AVG, GROUP BY) for analytical questions.",
    "Output ONLY the SQL — no prose, no markdown fences, no explanation.",
]

SQL_SYSTEM_PROMPT = (
    "You are a careful analytics engineer that translates a natural-language "
    "question into a single SQLite SELECT query.\n\nRules:\n"
    + "\n".join(f"- {rule}" for rule in _SQL_RULES)
)


def build_sql_messages(query: str, schema: str) -> list[dict]:
    """Build the chat messages that ask the model for a SELECT query."""
    user_content = f"Database schema:\n{schema}\n\nQuestion: {query}\n\nSQL:"
    return [
        {"role": "system", "content": SQL_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


SQL_ANSWER_SYSTEM_PROMPT = (
    "You are Sourcerer, a knowledge assistant. You are given a question, the SQL "
    "query that was run to answer it, and the query's result rows. State the answer "
    "in one or two sentences, grounded strictly in the result. Quote the relevant "
    "numbers. If the result is empty, say the data has no matching records."
)


def build_sql_answer_messages(query: str, sql: str, result_table: str) -> list[dict]:
    """Build the chat messages that turn SQL results into a natural-language answer."""
    user_content = (
        f"Question: {query}\n\n"
        f"SQL executed:\n{sql}\n\n"
        f"Result:\n{result_table}\n\n"
        "Answer the question using only this result."
    )
    return [
        {"role": "system", "content": SQL_ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
