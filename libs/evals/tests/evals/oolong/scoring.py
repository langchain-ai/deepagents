"""Scoring for the Oolong benchmark.

Direct port of ``evals/oolong/scoring.ts`` in ``langchain-ai/deepagentsjs``
so Python-side numbers are comparable to what the JS eval suite reports.
The JS port itself is a port of the paper harness
(``oolong_benchmark/src/eval/eval_helpers.py::synth_process_response``) —
which means this file is two hops from the paper. Most of the strategy
is literal:

1. ``canonical_prediction`` extracts a scalar from the model's raw text.
2. ``score_output`` applies one of four strategies keyed on the example's
   ``answer_type``: exact/normalized string match, comparison-phrase
   containment, numeric partial credit (``0.75 ** |diff|``), or flexible
   date match.

See `tests/evals/oolong/README.md` (forthcoming) for the "JS parity"
note. One deviation worth calling out: the JS version splits on lines
and takes the last non-empty one *before* splitting on ``:``; the
paper's Python splits on ``:`` only. We follow JS-parity here so Python
numbers line up with what the team has been reading on the JS side.
Rerun against the paper's harness if you need a paper-to-paper number.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from datetime import date, datetime

_NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")

_COMPARISON_PHRASES: tuple[str, ...] = (
    "more common than",
    "less common than",
    "same frequency as",
)

# Python `datetime.date(y, m, d)` literal — appears in gold answers for
# DATE-typed rows because the upstream dataset was generated from Python.
_PY_DATE_RE = re.compile(r"datetime\.date\(\s*(\d{4})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})\s*\)")

# Common non-ISO date formats we try in order. Flexible parsing is
# deliberate — the paper says "flexible date parsing comparison" and
# the JS side leans on JavaScript's loose ``new Date(...)``.
_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d %b %Y",
    "%d %B %Y",
    "%Y-%m-%dT%H:%M:%S",
)


@dataclass(frozen=True)
class Score:
    """Result of scoring one prediction against one gold answer."""

    pred: str
    """Canonical prediction extracted from model output."""
    gold: str
    """Gold answer as a trimmed scalar string."""
    score: float
    """Score in [0, 1]. 1 for exact/comparison/date matches, 0.75^|diff|
    for numeric partial credit, 0 otherwise."""
    correct: bool
    """Whether ``score >= 1``. LangSmith feedback key for pass/fail."""
    exact_match: bool
    """Trimmed string equality after canonical parsing."""
    normalized_match: bool
    """Lowercased, whitespace-collapsed equality."""
    contains_match: bool
    """Comparison-phrase containment (only meaningful for ``ANSWER_TYPE.COMPARISON``)."""
    numeric_match: bool
    """Exact numeric match (only meaningful for ``ANSWER_TYPE.NUMERIC``)."""


def _normalize_text(value: str) -> str:
    """Collapse whitespace and lowercase for fuzzy comparison."""
    return " ".join(value.strip().lower().split())


def _strip_markdown(value: str) -> str:
    """Strip outer ``**...**`` and ``*...*`` wrappers.

    Mirrors the JS ``stripMarkdown``: peels matched bold first, then
    matched italic. Unbalanced markers are left alone.
    """
    text = value.strip()
    while text.startswith("**") and text.endswith("**") and len(text) >= 4:
        text = text[2:-2].strip()
    while (
        text.startswith("*") and text.endswith("*") and not text.startswith("**") and len(text) >= 2
    ):
        text = text[1:-1].strip()
    return text


def _first_number(value: str) -> float | None:
    """Extract the first numeric literal from a string, ignoring commas."""
    match = _NUMERIC_RE.search(value.replace(",", ""))
    if match is None:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def canonical_prediction(value: str) -> str:
    """Reduce raw model output to a canonical scalar answer.

    We intentionally diverge from the paper's ``synth_attempt_answer_parse``
    by first taking the last non-empty line before splitting on ``:``.

    Pipeline:

    1. Trim.
    2. If multi-line, take the last non-empty line.
    3. If no ``:`` and length < 20, return after markdown strip.
    4. If no ``:`` and length >= 20, return the last word after markdown strip.
    5. Otherwise split on ``:`` and take the last segment, strip
       markdown asterisks and ``[``/``]`` wrappers.
    """
    text = value.strip()

    if "\n" in text:
        lines = [line for line in text.split("\n") if line.strip()]
        if lines:
            text = lines[-1].strip()

    if ":" not in text:
        if len(text) < 20:
            return _strip_markdown(text)
        words = text.split()
        return _strip_markdown(words[-1] if words else text)

    candidate = text.split(":")[-1].strip()
    # this is not a markdown strip, just three chars the agent tends to sprinkle
    # around answers (e.g. `The answer is: **[spam]**`).
    candidate = candidate.replace("*", "").replace("[", "").replace("]", "")
    return candidate.strip()


def parse_gold(raw: object) -> str:
    """Normalize a gold answer to a scalar string.

    The Oolong dataset ships gold answers as Python-repr list strings
    like ``"['spam']"`` or ``"[4]"``. Handles:

    - Already-a-list Python objects → take first element.
    - JSON/Python list strings → parse via ``ast.literal_eval`` (which
      accepts single quotes, matching the JS quote-swap-then-JSON
      approach).
    - Plain scalars → ``str``-cast and trim.
    """
    gold: object = raw
    if isinstance(gold, list):
        gold = gold[0] if gold else ""
    text = str(gold).strip()

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            # Fall back to JSON after quote-swap — covers edge cases
            # ``literal_eval`` rejects (e.g. nested True/False shapes).
            try:
                parsed = json.loads(text.replace("'", '"'))
            except json.JSONDecodeError:
                return text
        if isinstance(parsed, list) and parsed:
            return str(parsed[0]).strip()
        return text
    return text


def _try_parse_date(value: str) -> date | None:
    """Flexibly parse a date string; return ``None`` on no match.

    Recognizes Python ``datetime.date(y, m, d)`` literals first
    (upstream gold answers use this) then walks a list of common
    formats. We don't use ``dateutil`` to keep the deps lean — the
    Oolong date shapes are narrow.
    """
    m = _PY_DATE_RE.search(value)
    if m is not None:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    trimmed = value.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(trimmed, fmt).date()
        except ValueError:
            continue
    return None


def score_output(output: str, gold_answer: str, answer_type: str) -> Score:
    """Score a raw model output against a gold answer.

    Follows the layered paper strategy: exact → normalized → type-
    specific. Returns a populated :class:`Score` so the test harness
    can emit the granular ``exact_match`` / ``normalized_match`` /
    ``contains_match`` / ``numeric_match`` feedback keys alongside the
    bool ``correct`` and numeric ``score``.

    Args:
        output: Raw text from the agent's final message.
        gold_answer: Trimmed scalar gold string (already passed through
            :func:`parse_gold`).
        answer_type: The dataset's ``answer_type`` column, e.g.
            ``"ANSWER_TYPE.LABEL"``. The three dataset suites in this
            directory (trec_coarse, multinli, metaphors) all produce
            ``ANSWER_TYPE.LABEL``, but the harness also handles
            ``COMPARISON``, ``NUMERIC``, and ``DATE`` so the same code
            works if the team extends to other Oolong datasets.
    """
    pred = canonical_prediction(output)

    exact_match = pred.strip() == gold_answer.strip()
    normalized_match = _normalize_text(pred) == _normalize_text(gold_answer)

    contains_match = False
    if answer_type == "ANSWER_TYPE.COMPARISON":
        norm_pred = _normalize_text(pred)
        norm_gold = _normalize_text(gold_answer)
        for phrase in _COMPARISON_PHRASES:
            if phrase in norm_gold and phrase in norm_pred:
                contains_match = True
                break

    numeric_match = False
    numeric_score = 0.0
    if answer_type == "ANSWER_TYPE.NUMERIC":
        gold_num = _first_number(gold_answer)
        pred_num = _first_number(pred)
        if gold_num is not None and pred_num is not None:
            numeric_score = 0.75 ** abs(gold_num - pred_num)
            numeric_match = numeric_score >= 1.0

    date_match = False
    if answer_type == "ANSWER_TYPE.DATE":
        gold_date = _try_parse_date(gold_answer)
        pred_date = _try_parse_date(pred)
        if gold_date is not None and pred_date is not None:
            date_match = gold_date == pred_date

    if exact_match or normalized_match or contains_match or date_match:
        score = 1.0
    elif answer_type == "ANSWER_TYPE.NUMERIC" and numeric_score > 0:
        score = numeric_score
    else:
        score = 0.0

    return Score(
        pred=pred,
        gold=gold_answer,
        score=score,
        correct=score >= 1.0,
        exact_match=exact_match,
        normalized_match=normalized_match,
        contains_match=contains_match,
        numeric_match=numeric_match,
    )


__all__ = [
    "Score",
    "canonical_prediction",
    "parse_gold",
    "score_output",
]
