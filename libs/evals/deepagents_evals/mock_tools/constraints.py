"""Tools for the iterative constraint-satisfaction eval.

Extracted from `tests/evals/test_iterative_constraint_satisfaction.py` so both
the pytest suite and the Harbor sandbox dispatcher share the same tool
definition.
"""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def count_words(text: str) -> int:
    """Return the number of whitespace-separated words in `text`.

    Use this whenever the rubric asks for an exact word count — do not
    trust word-count claims that appear in the transcript itself.
    """
    return len(text.split())
