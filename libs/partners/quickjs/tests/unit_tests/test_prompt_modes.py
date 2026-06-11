"""Unit tests for mode-specific prompt rendering."""

from typing import Literal

import pytest

from langchain_quickjs._prompt import (
    render_eval_tool_code_doc,
    render_eval_tool_description,
    render_repl_system_prompt,
)


@pytest.mark.parametrize(
    ("mode", "expected_fragment"),
    [
        (
            "thread",
            "persists across tool calls and across multiple turns for this "
            "conversation thread",
        ),
        (
            "turn",
            "persists across tool calls within a single turn of conversation",
        ),
        (
            "call",
            "runs JavaScript in a fresh sandboxed REPL for each invocation",
        ),
    ],
)
def test_render_repl_system_prompt_mode_specific(
    mode: Literal["thread", "turn", "call"], expected_fragment: str
) -> None:
    prompt = render_repl_system_prompt(
        tool_name="eval",
        timeout=5.0,
        memory_limit_mb=64,
        mode=mode,
    )
    assert expected_fragment in prompt
    assert "Timeout: 5.0s per call. Memory: 64 MB total." in prompt
    assert "### Dispatching Subagents with `task`" not in prompt


@pytest.mark.parametrize(
    ("mode", "expected_fragment"),
    [
        (
            "thread",
            "State persists across calls and across turns in this conversation.",
        ),
        (
            "turn",
            "State persists across calls within a turn, but resets between turns.",
        ),
        ("call", "Each call runs in a fresh REPL environment (no cross-call state)."),
    ],
)
def test_render_eval_tool_code_doc_mode_specific(
    mode: Literal["thread", "turn", "call"], expected_fragment: str
) -> None:
    doc = render_eval_tool_code_doc(mode=mode)
    assert expected_fragment in doc


@pytest.mark.parametrize(
    ("mode", "expected_fragment"),
    [
        (
            "thread",
            "Persistent state is enabled: variables and functions defined in one "
            "call are visible to subsequent calls in this conversation.",
        ),
        (
            "turn",
            "Persistent state is enabled within a single turn: variables and "
            "functions defined in one call are visible to later calls within the "
            "same turn, but reset between turns.",
        ),
        (
            "call",
            "Each call runs in a fresh sandboxed REPL with no state carried over.",
        ),
    ],
)
def test_render_eval_tool_description_mode_specific(
    mode: Literal["thread", "turn", "call"], expected_fragment: str
) -> None:
    description = render_eval_tool_description(mode=mode)
    assert expected_fragment in description
    assert description.startswith("Execute JavaScript in a sandboxed REPL.")
