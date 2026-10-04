"""Shared helpers for experiment notebooks.

Extracted from the bf16-era notebooks (32/40/43/45/46) during the 2026-07
cleanup. Existing notebooks keep their inline copies untouched (their code is
the record of what produced the logged results); new notebooks should import
from here instead of redefining.

Deliberately NOT extracted: ``make_methods`` and ``rows_from`` — both are
experiment-specific (each notebook defines its own method grid / result
reshaping), so a shared version would misrepresent the per-notebook variants.
"""

import json
from pathlib import Path

import pandas as pd


def llama_chat(user: str, assistant: str = "") -> str:
    """Llama-3 chat-format prompt with an optional assistant prefix."""
    return (
        f"<|start_header_id|>user<|end_header_id|>{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>{assistant}"
    )


def gemma_chat(user: str, assistant: str = "") -> str:
    """Gemma-2 chat-format prompt with an optional assistant prefix."""
    return (
        f"<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n{assistant}"
    )


def chat_fn(model_key: str):
    """Chat formatter for a model key (substring match, as in notebook 43)."""
    if "llama" in model_key:
        return llama_chat
    if "gemma" in model_key:
        return gemma_chat
    raise ValueError(model_key)


def answer(text: str) -> str:
    """The assistant part of a Llama-3 chat-format generation."""
    marker = "assistant<|end_header_id|>"
    return text.split(marker)[-1].strip() if marker in text else text


def read_jsonl(path: str | Path) -> list[dict]:
    """Parse an EvaluationHarness JSONL log into a list of records."""
    return [json.loads(line) for line in Path(path).read_text().strip().split("\n")]


def pareto_front(
    points: pd.DataFrame, cost: str = "ppl_ratio", gain: str = "joint_z"
) -> pd.DataFrame:
    """Rows not dominated by any other (lower ``cost`` AND higher ``gain``)."""
    keep = []
    for _, row in points.iterrows():
        dominated = (
            (points[cost] <= row[cost])
            & (points[gain] >= row[gain])
            & ((points[cost] < row[cost]) | (points[gain] > row[gain]))
        ).any()
        keep.append(not dominated)
    return points[keep].sort_values(cost)
