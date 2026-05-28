"""Eval test for iterative constraint satisfaction.

Asks a deep agent to produce a paragraph under two simultaneous hard
constraints: exact word count AND every sentence must start with the
phrase `Zebra protocol`. The agent is wired with `RubricMiddleware`, so
a separate grader sub-agent evaluates each draft against the rubric and
loops the main agent back with feedback until the rubric is satisfied
(or the iteration cap is hit).

The programmatic scorer slides a window across every contiguous span of
sentences in the final response and passes on the first span that hits
the exact word count with every sentence starting with the phrase
`Zebra protocol` (case-insensitive). This makes the test robust to
models that wrap the answer with reasoning, a word-count tally, or a
self-congratulatory summary line — without having to teach the model
about the grading strategy.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from deepagents import RubricMiddleware, create_deep_agent
from langchain_core.tools import tool

from tests.evals.utils import (
    AgentTrajectory,
    SuccessAssertion,
    TrajectoryScorer,
    run_agent,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


@tool
def count_words(text: str) -> int:
    """Return the number of whitespace-separated words in ``text``.

    Use this whenever the rubric asks for an exact word count — do not
    trust word-count claims that appear in the transcript itself.
    """
    return len(text.split())


pytestmark = [pytest.mark.eval_category("conversation")]


# Phrase every sentence must begin with (case-insensitive match).
_START_PHRASE = "Zebra protocol"
_START_PHRASE_LOWER = _START_PHRASE.lower()


def _starts_with_phrase(sentence: str) -> bool:
    """Case-insensitive check that ``sentence`` begins with ``_START_PHRASE``."""
    return sentence.lower().startswith(_START_PHRASE_LOWER)


# Splits text into sentences on any run of sentence-ending punctuation,
# blank-line paragraph breaks, or a colon directly followed by a newline
# (which models often use to introduce the answer block, e.g.
# "Here's the story:\n\nZeke ..."). Including those as sentence enders
# stops a preamble line from gluing itself to the first story sentence.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+|:\s*\n+|\n\s*\n+")

# Stripped from the edges of each paragraph so wrapping in quotes, markdown
# emphasis, code fences, or list-marker dashes doesn't trip the grader.
_WRAPPER_CHARS = " \t\n\r\"'`*_-"


def _sentences(text: str) -> list[str]:
    """Split into sentences and drop empty fragments + wrapper chars."""
    return [
        cleaned
        for cleaned in (s.strip().strip(_WRAPPER_CHARS) for s in _SENTENCE_SPLIT_RE.split(text))
        if cleaned
    ]


def _grade_response(answer: str, target_words: int) -> tuple[bool, list[str]]:
    """Slide a window of contiguous phrase-starting sentences and return on first match.

    The model's final response often wraps the answer paragraph with
    preamble ("Here you go:"), postamble ("This is 147 words exactly."),
    or both — sometimes on separate lines, sometimes inline. We can't
    reliably point at "the answer paragraph" up front, so we walk the
    sentence list and consider every maximal run of consecutive sentences
    starting with ``_START_PHRASE`` as a candidate. Within each such run,
    every contiguous sub-window is checked for an exact ``target_words``
    match; if none hits, we fast-forward past any non-matching sentences
    (preamble, "Word count: 147" interjections, postamble) to the next
    qualifying run rather than restarting from each index — that lets the
    scan recover from noise sitting between the story and the model's
    commentary without ever accepting a non-matching sentence as part of
    the answer.
    """
    sentences = _sentences(answer)
    if not sentences:
        return False, ["no sentences detected in response"]

    idx = 0
    while idx < len(sentences):
        # Advance past any sentence that doesn't begin with the required phrase.
        if not _starts_with_phrase(sentences[idx]):
            idx += 1
            continue
        # Find the maximal contiguous run of phrase-starting sentences from idx.
        run_end = idx
        while run_end < len(sentences) and _starts_with_phrase(sentences[run_end]):
            run_end += 1
        # Sliding window over (start, end) within this run; early-break
        # whenever the running word count would exceed target_words.
        for start in range(idx, run_end):
            word_count = 0
            for end in range(start, run_end):
                word_count += len(sentences[end].split())
                if word_count == target_words:
                    return True, []
                if word_count > target_words:
                    break
        idx = run_end

    total_words = sum(len(s.split()) for s in sentences)
    non_phrase = sum(1 for s in sentences if not _starts_with_phrase(s))
    return False, [
        f"no contiguous span starting with {_START_PHRASE!r} summed to {target_words} words",
        f"response stats: sentences={len(sentences)}, total_words={total_words}, non-phrase starts={non_phrase}",
    ]


@dataclass(frozen=True)
class ExactWordCountAndPhraseStarts(SuccessAssertion):
    """At least one paragraph in the response must be exactly ``target_words``
    long with every sentence beginning with the phrase ``_START_PHRASE``
    (case-insensitive).
    """

    target_words: int

    def check(self, trajectory: AgentTrajectory) -> bool:
        passed, _ = _grade_response(trajectory.answer, self.target_words)
        return passed

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        _, problems = _grade_response(trajectory.answer, self.target_words)
        return "no paragraph satisfied both constraints: " + " | ".join(problems)


TARGET_WORDS = 147

_QUERY = (
    f"Write a story about a bear that's {TARGET_WORDS} words and has every "
    'sentence starting with the phrase "Zebra protocol".'
)

_RUBRIC = (
    f"The story must be {TARGET_WORDS} words and every sentence must start "
    'with the phrase "Zebra protocol".'
)


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_exact_word_count_and_z_starts(model: BaseChatModel) -> None:
    """Deep agent satisfies two interacting hard constraints by iterating.

    `RubricMiddleware` grades each draft against `_RUBRIC` and loops the
    main agent back with feedback until the rubric is satisfied or the
    iteration cap is reached. The programmatic scorer below is the final
    test-pass gate and guards against grader hallucinations: it slides a
    window across the response's sentences and accepts any contiguous
    z-starting span that totals exactly ``TARGET_WORDS`` words.
    """
    agent = create_deep_agent(
        model=model,
        middleware=[
            RubricMiddleware(
                model=model,
                tools=[count_words],
                max_iterations=5,
            )
        ],
    )
    run_agent(
        agent,
        model=model,
        query=_QUERY,
        extra_state={"rubric": _RUBRIC},
        scorer=TrajectoryScorer().success(ExactWordCountAndPhraseStarts(target_words=TARGET_WORDS)),
    )
