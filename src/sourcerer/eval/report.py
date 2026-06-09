"""Format and persist the eval comparison table."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

# Columns shown in the comparison table, in order.
_COLUMNS = [
    ("config", "Config"),
    ("recall@k", "Recall@k"),
    ("mrr", "MRR"),
    ("hit_rate", "Hit rate"),
    ("faithfulness", "Faithfulness"),
    ("answer_relevancy", "Answer rel."),
    ("avg_latency_ms", "Latency ms"),
]


def _fmt(key: str, value) -> str:
    if value is None:
        return "—"
    if key == "config":
        return str(value)
    if key == "avg_latency_ms":
        return f"{value:.0f}"
    return f"{value:.3f}"


def render_table(rows: list[dict], k: int) -> str:
    """Render the results rows as a GitHub-flavoured Markdown table."""
    headers = [label for _, label in _COLUMNS]
    lines = [
        f"Retrieval metrics evaluated at k={k}.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        cells = [_fmt(key, row.get(key)) for key, _ in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_results(rows: list[dict], k: int, meta: dict, results_dir: str | Path) -> Path:
    """Write a timestamped Markdown report (+ JSON) and return the Markdown path."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    table = render_table(rows, k)
    md = (
        f"# Eval comparison — {stamp}\n\n"
        f"- Eval items: {meta.get('n_items')}\n"
        f"- Generator: `{meta.get('generator')}`  ·  Judge: `{meta.get('judge')}`\n"
        f"- Embedding: `{meta.get('embedding')}`\n\n"
        f"{table}\n"
    )
    md_path = out_dir / f"comparison-{stamp}.md"
    md_path.write_text(md, encoding="utf-8")
    (out_dir / f"comparison-{stamp}.json").write_text(
        json.dumps({"k": k, "meta": meta, "rows": rows}, indent=2), encoding="utf-8"
    )
    return md_path
