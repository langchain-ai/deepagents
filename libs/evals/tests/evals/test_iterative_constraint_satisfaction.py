"""Eval test for iterative constraint satisfaction.

Asks a deep agent (with whatever `--model` is selected) to produce a
paragraph under two simultaneous hard constraints: exact word count AND
every sentence must start with a vowel. The agent has access to planning,
scratchpad, and the agent loop itself — room to draft, count, and revise
until both constraints are met. The test passes when the final paragraph
satisfies the rubric.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from deepagents import create_deep_agent

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

from tests.evals.utils import (
    AgentTrajectory,
    SuccessAssertion,
    TrajectoryScorer,
    run_agent,
)

pytestmark = [pytest.mark.eval_category("conversation")]


# All vowel characters (upper and lowercase) used to check sentence starts.
_VOWELS = frozenset("aeiouAEIOU")

# Regex that splits text into sentences on any sequence of sentence-ending punctuation - periods, exclamation points, or question marks.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")

# Characters stripped from the edges of the extracted paragraph (whitespace, quotes, markdown). Done to allow grading to be robust to models that wrap their final answer in quotes, markdown, or code fences.
_WRAPPER_CHARS = " \t\n\r\"'`*_"


def _extract_paragraph(answer: str) -> str:
    """Extract the last paragraph from the model's reply.

    Splits the answer on blank lines and returns the final non-empty chunk,
    stripped of surrounding markdown / quotes / fences. This makes grading
    robust to models that prefix their answer with reasoning, drafts, or
    summaries — only the trailing paragraph is graded.
    """
    stripped = answer.strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", stripped) if p.strip()]
    last = paragraphs[-1] if paragraphs else stripped
    if last.startswith("```"):
        # Drop a fenced code block: take the inside, ignore an optional language tag.
        inner = last.strip("`")
        if "\n" in inner:
            inner = inner.split("\n", 1)[1]
        last = inner.rsplit("```", 1)[0]
    return last.strip(_WRAPPER_CHARS)


def _sentences(paragraph: str) -> list[str]:
    """Split into sentences and drop empty fragments from trailing punctuation."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]


def _grade_text(text: str, target_words: int) -> tuple[bool, list[str]]:
    """Grade a paragraph against both constraints.

    Returns `(passed, problems)` where `problems` is a human-readable list
    of constraint violations (empty when ``passed`` is True).
    """
    paragraph = _extract_paragraph(text)
    word_count = len(paragraph.split())
    sentences = _sentences(paragraph)
    non_vowel = [s for s in sentences if not s or s[0] not in _VOWELS]

    problems: list[str] = []
    if word_count != target_words:
        problems.append(f"word count {word_count} != {target_words}")
    if not sentences:
        problems.append("no sentences detected")
    if non_vowel:
        preview = "; ".join(s[:40] for s in non_vowel)
        problems.append(f"{len(non_vowel)} sentence(s) did not start with a vowel: {preview}")
    return not problems, problems


@dataclass(frozen=True)
class ExactWordCountAndVowelStarts(SuccessAssertion):
    """The final paragraph must be exactly ``target_words`` long and every
    sentence must begin with a vowel.
    """

    target_words: int

    def check(self, trajectory: AgentTrajectory) -> bool:
        passed, _ = _grade_text(trajectory.answer, self.target_words)
        return passed

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        _, problems = _grade_text(trajectory.answer, self.target_words)
        return "constraint violation: " + "; ".join(problems)


TARGET_WORDS = 72

_QUERY = """Write a story about a bear that's 72 words and has every sentence starting with a vowel. Output paragraph only."""


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_exact_word_count_and_vowel_starts(model: BaseChatModel) -> None:
    """Deep agent satisfies two interacting hard constraints by iterating.

    The agent has access to planning, scratchpad, and the agent loop, which
    lets it draft, count, and revise. The test passes when the final
    paragraph hits the exact word count AND every sentence starts with a
    vowel.
    """
    agent = create_deep_agent(model=model)
    run_agent(
        agent,
        model=model,
        query=_QUERY,
        scorer=TrajectoryScorer().success(ExactWordCountAndVowelStarts(target_words=TARGET_WORDS)),
    )
