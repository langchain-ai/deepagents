"""Oolong answer scoring.

Ported from the rlming benchmark harness. A prediction is correct if any of:
- exact_match: canonicalized prediction equals gold answer
- normalized_match: case/whitespace-normalized equality
- contains_match: normalized gold appears inside normalized prediction
- numeric_match: first extracted number from each side matches
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

_ANSWER_PREFIXES = ("final answer:", "answer:", "label:", "user:")


def _normalize_text(value: str) -> str:
    """Collapse whitespace and lowercase."""
    return " ".join(value.strip().lower().split())


def _strip_markdown(value: str) -> str:
    """Strip markdown bold/italic markers."""
    text = value.strip()
    while text.startswith("**") and text.endswith("**"):
        text = text[2:-2].strip()
    while text.startswith("*") and text.endswith("*") and not text.startswith("**"):
        text = text[1:-1].strip()
    return text


def _first_number(value: str) -> str | None:
    """Extract the first number from a string, ignoring commas."""
    match = _NUMERIC_RE.search(value.replace(",", ""))
    return match.group(0) if match else None


def canonical_prediction(value: str) -> str:
    """Parse raw model output into a canonical prediction string.

    Scans lines in reverse looking for known answer prefixes
    (`final answer:`, `answer:`, etc.), then falls back to the last
    non-empty line.
    """
    text = value.strip()
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        lower = _strip_markdown(stripped).lower()
        for prefix in _ANSWER_PREFIXES:
            if lower.startswith(prefix):
                result = _strip_markdown(stripped)
                return _strip_markdown(result[len(prefix) :].strip())
    lowered = text.lower()
    for prefix in _ANSWER_PREFIXES:
        if lowered.startswith(prefix):
            return _strip_markdown(text[len(prefix) :].strip())
    if "\n" in text:
        last_line = text.splitlines()[-1].strip()
        if last_line:
            return _strip_markdown(last_line)
    return _strip_markdown(text)


def parse_gold(raw: str | list[str]) -> str:
    """Normalize a gold answer field into a single string.

    Handles plain strings, Python-style list strings (e.g. `"['spam']"`),
    JSON arrays, and numeric values.
    """
    gold = raw
    if isinstance(gold, list):
        gold = gold[0] if gold else ""
    gold = str(gold).strip()
    if gold.startswith("[") and gold.endswith("]"):
        try:
            parsed = json.loads(gold.replace("'", '"'))
            if isinstance(parsed, list) and parsed:
                gold = str(parsed[0]).strip()
        except (json.JSONDecodeError, IndexError):
            pass
    return gold


@dataclass(frozen=True)
class Score:
    """Scoring result for a single Oolong task."""

    pred: str
    gold: str
    correct: bool
    exact_match: bool
    normalized_match: bool
    contains_match: bool
    numeric_match: bool


def score_output(*, output: str, gold_answer: str) -> Score:
    """Score a model output against a gold answer.

    Args:
        output: Raw model output text.
        gold_answer: Parsed gold answer string.

    Returns:
        A `Score` with match details.
    """
    pred = canonical_prediction(output)
    exact_match = pred.strip() == gold_answer.strip()
    normalized_match = _normalize_text(pred) == _normalize_text(gold_answer)
    contains_match = _normalize_text(gold_answer) in _normalize_text(pred)
    gold_num = _first_number(gold_answer)
    pred_num = _first_number(pred)
    numeric_match = gold_num is not None and pred_num is not None and gold_num == pred_num
    correct = exact_match or normalized_match or contains_match or numeric_match
    return Score(
        pred=pred,
        gold=gold_answer,
        correct=correct,
        exact_match=exact_match,
        normalized_match=normalized_match,
        contains_match=contains_match,
        numeric_match=numeric_match,
    )
