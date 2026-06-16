"""Synthesize a cited natural-language answer from a Text-to-SQL execution.

Reuses `generator.Answer` / `generator.Citation` so the API response shape is
identical to the RAG path. The citation *is* the executed SQL plus the first rows
— the answer is transparent and verifiable: anyone can re-run the query.
"""

from __future__ import annotations

from pathlib import Path

from sourcerer.config import Settings
from sourcerer.generation.generator import Answer, Citation
from sourcerer.generation.prompts import NO_ANSWER, build_sql_answer_messages
from sourcerer.llm.client import LLMClient
from sourcerer.sqlkb import text_to_sql

_SNIPPET_CHARS = 400


def _citation(execution: text_to_sql.SQLExecution, db_stem: str) -> Citation:
    """Build the single citation: the SQL that ran + a peek at the result."""
    table = text_to_sql.render_table(execution.columns, execution.rows, max_rows=5)
    snippet = f"{execution.sql}\n\n{table}"
    if len(snippet) > _SNIPPET_CHARS:
        snippet = snippet[:_SNIPPET_CHARS] + "…"
    return Citation(n=1, source=f"{db_stem} (SQL)", chunk_index=0, score=1.0, snippet=snippet)


def answer(
    query: str, settings: Settings, sql_client: LLMClient, answer_client: LLMClient
) -> Answer:
    """Run Text-to-SQL and synthesize a grounded, cited answer.

    `sql_client` writes the query (local by default, API when configured);
    `answer_client` phrases the result (follows the router's local/api decision).
    Token usage from both calls is summed onto the returned Answer.
    """
    db_stem = Path(settings.sql_kb_path).stem
    execution = text_to_sql.run(query, settings, sql_client)

    # Generation failed validation/execution → no answer (guardrail, never guess).
    if not execution.ok:
        return Answer(
            text=NO_ANSWER,
            citations=[],
            model=execution.model or None,
            input_tokens=execution.input_tokens,
            output_tokens=execution.output_tokens,
        )

    table = text_to_sql.render_table(execution.columns, execution.rows)
    reply = answer_client.chat(build_sql_answer_messages(query, execution.sql, table))

    return Answer(
        text=reply.text,
        citations=[_citation(execution, db_stem)],
        model=reply.model or execution.model,
        input_tokens=execution.input_tokens + reply.input_tokens,
        output_tokens=execution.output_tokens + reply.output_tokens,
    )
