"""Unit tests for the Oolong scorer.

These are the parity-guard tests. Each case here should produce the same
numbers the JS ``scoring.ts`` port does — if someone edits
``canonical_prediction`` or ``score_output`` and breaks that parity,
we want to catch it here rather than waiting for a diverged LangSmith
dashboard.

Cases are organized by the piece they exercise:

- ``canonical_prediction`` — multi-line, colon-split, markdown strip,
  last-word fallback.
- ``parse_gold`` — the Python-repr list quirk.
- ``score_output`` — one scenario per answer type.
"""

from __future__ import annotations

import math

import pytest

from tests.evals.oolong.scoring import (
    canonical_prediction,
    parse_gold,
    score_output,
)

# ---- canonical_prediction ----------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Short, no colon → pass-through with markdown strip.
        ("spam", "spam"),
        ("  spam  ", "spam"),
        ("**spam**", "spam"),
        ("*spam*", "spam"),
        # Long, no colon → last word, markdown-stripped.
        (
            "this is a long explanation without a colon ending in **spam**",
            "spam",
        ),
        # Has a colon → last segment, stripped of * and [ ].
        ("The answer is: spam", "spam"),
        ("Answer: **[spam]**", "spam"),
        ("Step 1: analyzed\nFinal answer: spam", "spam"),
        # Multi-line → take last non-empty line before the rest of the pipeline.
        ("first line\n\nsecond line: foo", "foo"),
        ("first: bar\n\nlast: baz", "baz"),
        # Last word when long and no colon.
        ("one two three four five six seven eight nine ten answer", "answer"),
    ],
)
def test_canonical_prediction(raw: str, expected: str) -> None:
    assert canonical_prediction(raw) == expected


def test_canonical_prediction_empty_input() -> None:
    """Defensive: empty string stays empty, doesn't crash."""
    assert canonical_prediction("") == ""


def test_canonical_prediction_only_whitespace_lines() -> None:
    """All lines blank → fall through to the no-colon branch on the trimmed whole."""
    assert canonical_prediction("   \n   \n   ") == ""


# ---- parse_gold --------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Plain scalar.
        ("spam", "spam"),
        # Python-repr list (the common Oolong shape).
        ("['spam']", "spam"),
        ("['less common than']", "less common than"),
        # Numeric inside a list.
        ("[4]", "4"),
        # JSON list with double quotes.
        ('["spam"]', "spam"),
        # Actual Python list object passed through.
        (["spam"], "spam"),
        # Empty list.
        ([], ""),
        # Nested objects — `ast.literal_eval` handles them; we just take [0].
        ("['a', 'b']", "a"),
    ],
)
def test_parse_gold(raw: object, expected: str) -> None:
    assert parse_gold(raw) == expected


def test_parse_gold_unparseable_bracket_string_returns_as_is() -> None:
    """If the [...] can't parse as a list, keep the raw bracketed text.

    Mirrors the JS ``// keep as-is`` fallback — never silently eat
    characters the test suite might be depending on.
    """
    assert parse_gold("[not a valid literal") == "[not a valid literal"


# ---- score_output: LABEL ------------------------------------------


def test_label_exact_match() -> None:
    s = score_output("The answer is: spam", "spam", "ANSWER_TYPE.LABEL")
    assert s.correct is True
    assert s.score == 1.0
    assert s.exact_match is True


def test_label_normalized_match_only() -> None:
    """Different case / whitespace → normalized-match wins, score still 1."""
    s = score_output("  SPAM  ", "spam", "ANSWER_TYPE.LABEL")
    assert s.correct is True
    assert s.exact_match is False
    assert s.normalized_match is True


def test_label_mismatch() -> None:
    s = score_output("The answer is: ham", "spam", "ANSWER_TYPE.LABEL")
    assert s.correct is False
    assert s.score == 0.0


# ---- score_output: COMPARISON -----------------------------------


def test_comparison_phrase_containment() -> None:
    """The phrase appears wrapped in extra words on both sides → still scores."""
    s = score_output(
        "My conclusion: 'dog' is more common than 'cat' here",
        "more common than",
        "ANSWER_TYPE.COMPARISON",
    )
    assert s.correct is True
    assert s.contains_match is True


def test_comparison_different_phrase_mismatches() -> None:
    s = score_output(
        "My conclusion: more common than expected",
        "less common than",
        "ANSWER_TYPE.COMPARISON",
    )
    assert s.correct is False
    assert s.contains_match is False


# ---- score_output: NUMERIC ---------------------------------------


def test_numeric_exact() -> None:
    s = score_output("Count: 42", "42", "ANSWER_TYPE.NUMERIC")
    assert s.correct is True
    assert s.score == 1.0


def test_numeric_off_by_one_partial_credit() -> None:
    """0.75 ** 1 = 0.75."""
    s = score_output("Count: 43", "42", "ANSWER_TYPE.NUMERIC")
    assert s.correct is False
    assert math.isclose(s.score, 0.75, rel_tol=1e-9)


def test_numeric_off_by_four_partial_credit() -> None:
    """0.75 ** 4 ≈ 0.316."""
    s = score_output("Count: 46", "42", "ANSWER_TYPE.NUMERIC")
    assert math.isclose(s.score, 0.75**4, rel_tol=1e-9)


def test_numeric_ignores_commas_in_number() -> None:
    s = score_output("Total: 1,234", "1234", "ANSWER_TYPE.NUMERIC")
    assert s.correct is True


def test_numeric_no_number_in_output() -> None:
    s = score_output("I give up", "42", "ANSWER_TYPE.NUMERIC")
    assert s.correct is False
    assert s.score == 0.0


# ---- score_output: DATE ------------------------------------------


def test_date_python_literal_gold() -> None:
    """Gold comes in the ``datetime.date(...)`` shape the dataset uses."""
    s = score_output(
        "Final date: 2024-01-15",
        "datetime.date(2024, 1, 15)",
        "ANSWER_TYPE.DATE",
    )
    assert s.correct is True


def test_date_cross_format_match() -> None:
    s = score_output(
        "The answer: Jan 15, 2024",
        "2024-01-15",
        "ANSWER_TYPE.DATE",
    )
    assert s.correct is True


def test_date_mismatch() -> None:
    s = score_output(
        "The answer: 2024-02-15",
        "2024-01-15",
        "ANSWER_TYPE.DATE",
    )
    assert s.correct is False
    assert s.score == 0.0


def test_date_unparseable_output_mismatches() -> None:
    s = score_output(
        "I'm not sure",
        "2024-01-15",
        "ANSWER_TYPE.DATE",
    )
    assert s.correct is False


# ---- score_output: cross-answer-type guards ----------------------


def test_numeric_flag_only_meaningful_for_numeric_type() -> None:
    """``numeric_match`` is always False for non-NUMERIC types, even when
    the strings happen to contain numbers."""
    s = score_output("The answer: 42", "42", "ANSWER_TYPE.LABEL")
    assert s.correct is True
    assert s.numeric_match is False


def test_contains_flag_only_meaningful_for_comparison_type() -> None:
    """``contains_match`` is always False for non-COMPARISON types."""
    s = score_output(
        "The answer: more common than expected",
        "more common than",
        "ANSWER_TYPE.LABEL",
    )
    # Exact string "more common than" doesn't appear as a pred standalone
    # after canonical parsing on this shape — so this is a real miss under
    # LABEL scoring. Point of the test: LABEL doesn't secretly pick up
    # COMPARISON's substring logic.
    assert s.contains_match is False
