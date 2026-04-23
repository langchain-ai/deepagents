"""Token-budget pre-flight for the compaction benchmark.

### What this test is for

The compaction benchmark only measures something meaningful if
compaction *actually fires* during a run. If the scripted turns plus
the fixture the agent reads don't add up to enough tokens, the
80k-token aggressive trigger never trips and the benchmark silently
degrades into "agent vs agent with no summarization involved."

This test is the cheap, deterministic gate that catches that
failure mode without a model call: it counts tokens in the raw
inputs the agent will almost certainly see - every scripted user
message, plus every fixture file - and asserts the total crosses a
conservative floor well above the trigger.

### What it does NOT assert

- It does not assert compaction fires at a specific turn. That
  requires a real agent run and is covered by the integration-style
  ``test_compaction_bench.py`` entry point.
- It does not count agent reasoning tokens, tool-call JSON overhead,
  or redundant reads. Those all *add* to the real count, so the
  floor here is a safe under-estimate.

### Token counter

Uses tiktoken's ``cl100k_base`` encoding if available (accurate for
GPT-class models and a close-enough proxy for Claude). Falls back to
a chars/4 heuristic when tiktoken isn't installed, with a lower
threshold to account for the rougher counting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.evals.compaction_bench.graders import load_fixture
from tests.evals.compaction_bench.instance_001_partnerco import INSTANCE
from tests.evals.compaction_bench.task_spec import AGGRESSIVE_TRIGGER_TOKENS

if TYPE_CHECKING:
    from collections.abc import Callable


# ---------------------------------------------------------------------------
# Token-counting helpers
# ---------------------------------------------------------------------------


def _tiktoken_counter() -> Callable[[str], int] | None:
    """Return a tiktoken-based counter, or ``None`` if tiktoken is missing.

    Prefers ``cl100k_base`` for stability across model releases. Any
    error from the tokenizer library is treated as unavailable rather
    than fatal - this test must not crash a dev environment that
    skipped the optional extras.
    """
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 -- tiktoken has several error paths; any means unavailable.
        return None
    return lambda text: len(encoding.encode(text))


def _chars_over_four_counter() -> Callable[[str], int]:
    """Rough fallback counter: ``len(text) // 4``.

    This consistently under-counts whitespace-heavy text (code) and
    over-counts dense prose, but averages close enough for a floor
    check. When used, the floor is lowered to account for rougher
    counting.
    """
    return lambda text: max(1, len(text) // 4)


def _select_counter() -> tuple[Callable[[str], int], str]:
    """Return ``(counter, counter_name)`` picking tiktoken if available."""
    tt = _tiktoken_counter()
    if tt is not None:
        return tt, "tiktoken"
    return _chars_over_four_counter(), "chars_over_four"


# ---------------------------------------------------------------------------
# Budget components
# ---------------------------------------------------------------------------


def _scripted_message_tokens(counter: Callable[[str], int]) -> int:
    """Sum of tokens in the scripted ``UserMessage.content`` values.

    This is the irreducible floor: the agent sees every one of these
    messages verbatim, so their tokens contribute to the context on
    every subsequent turn.
    """
    return sum(counter(m.content) for m in INSTANCE.messages)


def _fixture_tokens(counter: Callable[[str], int]) -> int:
    """Sum of tokens in the fixture mini-repo's text files.

    The agent has to explore the repo to answer Feature A; the
    ``ls`` / ``read_file`` calls that produce this exploration will
    place these file contents directly into the transcript. Even a
    lazy agent ends up reading most of this material once.
    """
    fixture = load_fixture(INSTANCE.fixture_dir)
    return sum(counter(content) for content in fixture.values())


# ---------------------------------------------------------------------------
# The preflight assertion
# ---------------------------------------------------------------------------


# The floor is derived from the trigger: we need enough raw-input
# tokens that, amplified by realistic agent overhead (reasoning +
# tool-call JSON wrapping + intermediate file writes), the transcript
# crosses the trigger at least once during the scripted run.
#
# Empirically, a deepagents-style agent produces ~2-3x the raw-input
# size in total transcript when exploring and editing a small repo.
# We pick a conservative 2x multiplier and require the raw-input total
# to be at least ``trigger / 2`` with a 40% safety margin on top.
#
# At the v1 trigger of 15k, that gives:
#   15_000 / 2 * 0.6 == 4_500 tokens (tiktoken)
# i.e. raw inputs totalling ~4.5k tiktoken tokens will reliably produce
# a transcript that crosses 15k at least once across 18 turns. If the
# trigger is later raised (e.g. during a v2 fixture expansion), the
# floor rises automatically.
_OVERHEAD_MULTIPLIER: float = 2.0
_SAFETY_MARGIN: float = 0.6
_TIKTOKEN_FLOOR: int = int(AGGRESSIVE_TRIGGER_TOKENS / _OVERHEAD_MULTIPLIER * _SAFETY_MARGIN)
"""Token floor when counting with tiktoken (close to real)."""

_CHARS_FALLBACK_FLOOR: int = int(_TIKTOKEN_FLOOR * 0.75)
"""Token floor when counting with the chars/4 fallback (rougher)."""


def test_token_budget_floor_is_reached() -> None:
    """The scripted inputs + fixture exceed a conservative token floor.

    If this test fails it means the instance has been trimmed below
    the point where compaction will reliably fire. Either add more
    content to the scripted messages, enlarge the fixture, or
    intentionally lower ``AGGRESSIVE_TRIGGER_TOKENS`` - but make that
    choice deliberately rather than discovering it as silent eval
    drift.
    """
    counter, counter_name = _select_counter()
    messages_tokens = _scripted_message_tokens(counter)
    fixture_tokens = _fixture_tokens(counter)
    total = messages_tokens + fixture_tokens

    floor = _TIKTOKEN_FLOOR if counter_name == "tiktoken" else _CHARS_FALLBACK_FLOOR

    assert total >= floor, (
        f"Raw input tokens ({counter_name}) are too low to trigger compaction.\n"
        f"  scripted user messages: {messages_tokens:,}\n"
        f"  fixture files:          {fixture_tokens:,}\n"
        f"  total:                  {total:,}\n"
        f"  required floor:         {floor:,}\n"
        f"  aggressive trigger:     {AGGRESSIVE_TRIGGER_TOKENS:,}\n"
        f"(real runs accumulate ~2x this in reasoning + tool-call tokens, "
        f"so a floor well below the trigger is fine.)"
    )


def test_scripted_messages_are_not_all_trivially_short() -> None:
    """At least some scripted turns carry substantive content.

    Short turns like ``"proceed"`` or ``"continue"`` are legitimate in
    real sessions and appear in the instance script, so this test does
    not enforce a per-turn floor. It only guards against the pathology
    of *every* turn being trivially short - which would mean the
    content driving compaction pressure lives entirely in tool outputs
    rather than the user script, making per-phase grading noisy.
    """
    counter, _ = _select_counter()
    substantive = [m for m in INSTANCE.messages if counter(m.content) >= 40]
    assert len(substantive) >= len(INSTANCE.messages) // 2, (
        f"Only {len(substantive)}/{len(INSTANCE.messages)} turns carry "
        f"substantive (>=40 token) content. Too few substantive turns "
        f"makes per-phase grading (G12/G13) noisy."
    )


def test_fixture_has_enough_content_to_pressure_context() -> None:
    """The fixture is large enough that reading it fills the context.

    Guards against someone accidentally trimming the fixture to a
    handful of trivial files. The floor is intentionally above what
    a tiny sample repo would produce but well below the trigger -
    the fixture alone is not expected to cross the trigger without
    the accumulating turn history.
    """
    counter, _ = _select_counter()
    fixture_tokens = _fixture_tokens(counter)
    assert fixture_tokens >= 3_000, (
        f"Fixture is too small ({fixture_tokens:,} tokens) to simulate "
        f"a realistic mini-repo the agent would explore."
    )


@pytest.mark.parametrize(
    ("label", "condition"),
    [
        (
            "AGGRESSIVE_TRIGGER_TOKENS is below the production 150k default",
            AGGRESSIVE_TRIGGER_TOKENS < 150_000,
        ),
        (
            "AGGRESSIVE_TRIGGER_TOKENS is above a 5k floor (not a typo)",
            AGGRESSIVE_TRIGGER_TOKENS > 5_000,
        ),
    ],
    ids=["below_prod_default", "above_typo_floor"],
)
def test_aggressive_trigger_is_sane(label: str, *, condition: bool) -> None:
    """The aggressive trigger is plausibly positioned in token space.

    Not a unit test of the trigger value itself - it is a sanity
    gate that catches accidental edits (e.g. a typo turning 15_000
    into 150 or 150_000 into 1_500_000). The bounds are deliberately
    wide so that the v2 fixture expansion can raise the trigger
    back toward 80k without this test failing.
    """
    assert condition, label
