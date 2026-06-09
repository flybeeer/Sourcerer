"""Generation metrics via LLM-as-judge.

Two scores in [0, 1], judged by EVAL_JUDGE_MODEL through the LLM client wrapper:
- faithfulness     : is the answer supported by the retrieved context? (no hallucination)
- answer_relevancy : does the answer actually address the question?

The judge is asked for a single number, which is more robust to parse from a
small local model than structured JSON. A separate (ideally stronger) judge model
reduces self-evaluation bias versus the generator.
"""

from __future__ import annotations

import re

from sourcerer.config import Settings
from sourcerer.llm.ollama_client import OllamaClient

_NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")

_FAITHFULNESS_PROMPT = """You are a strict evaluator. Given a CONTEXT and an ANSWER, rate how
well the ANSWER is supported by the CONTEXT.

- 1.0 = every claim in the answer is directly supported by the context.
- 0.0 = the answer contradicts the context or makes up unsupported claims.
- Saying "I don't know" when the context lacks the answer is fully faithful (1.0).

CONTEXT:
{context}

ANSWER:
{answer}

Respond with ONLY a number between 0 and 1 (e.g. 0.0, 0.5, 1.0)."""

_RELEVANCY_PROMPT = """You are a strict evaluator. Given a QUESTION and an ANSWER, rate how well the
ANSWER addresses the QUESTION.

- 1.0 = the answer directly and completely addresses the question.
- 0.0 = the answer is off-topic or does not address the question.
- An honest "I don't know" to an answerable question is only partially relevant.

QUESTION:
{question}

ANSWER:
{answer}

Respond with ONLY a number between 0 and 1 (e.g. 0.0, 0.5, 1.0)."""


class Judge:
    """LLM-as-judge using EVAL_JUDGE_MODEL through the Ollama client wrapper."""

    def __init__(self, settings: Settings) -> None:
        self._client = OllamaClient(
            base_url=settings.ollama_base_url,
            chat_model=settings.eval_judge_model,
            embedding_model=settings.embedding_model,
        )

    def _score(self, prompt: str) -> float:
        reply = self._client.chat([{"role": "user", "content": prompt}])
        match = _NUMBER_RE.search(reply)
        if not match:
            return 0.0
        return max(0.0, min(1.0, float(match.group(1))))

    def faithfulness(self, answer: str, context: str) -> float:
        return self._score(_FAITHFULNESS_PROMPT.format(context=context, answer=answer))

    def answer_relevancy(self, question: str, answer: str) -> float:
        return self._score(_RELEVANCY_PROMPT.format(question=question, answer=answer))
