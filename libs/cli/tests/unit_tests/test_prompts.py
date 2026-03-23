"""Tests for prompts module."""

from deepagents_cli.prompts import REMEMBER_PROMPT


def test_remember_prompt_is_nonempty_string() -> None:
    """`REMEMBER_PROMPT` should be a non-empty string."""
    assert isinstance(REMEMBER_PROMPT, str)
    assert len(REMEMBER_PROMPT) > 0
