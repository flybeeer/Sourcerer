"""Eval set loading.

The eval set is JSONL — one {question, expected_answer, relevant_doc_ids} per line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalItem:
    question: str
    expected_answer: str
    relevant_doc_ids: list[str]  # source file names that should be retrieved


def load_eval_set(path: str | Path) -> list[EvalItem]:
    """Load eval items from a JSONL file, skipping blank lines."""
    items: list[EvalItem] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        items.append(
            EvalItem(
                question=obj["question"],
                expected_answer=obj.get("expected_answer", ""),
                relevant_doc_ids=list(obj.get("relevant_doc_ids", [])),
            )
        )
    return items
