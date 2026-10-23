"""Tests for cached LangSmith secret redaction."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import pytest
from langsmith.anonymizer import (
    DEFAULT_SECRET_RULES,
    create_anonymizer,
    create_secret_anonymizer,
)

from deepagents_code._trace_redaction import (
    CachingRuleProcessor,
    create_cached_secret_anonymizer,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_SECRET = "sk-ant-" + "A" * 40


def _payload(turn: int) -> dict[str, object]:
    """Build a traced-run payload whose history grows by one turn per call."""
    messages: list[object] = [{"role": "system", "content": "You are helpful."}]
    for i in range(turn):
        user = {"role": "user", "content": f"step {i}: export API_KEY={_SECRET}"}
        command = f"curl -H 'Authorization: Bearer {_SECRET}'"
        assistant = {
            "role": "assistant",
            "content": [{"type": "text", "text": "Done."}],
            "tool_calls": [{"args": {"cmd": command}}],
        }
        tool = {"role": "tool", "content": "plain output " * 20}
        messages.extend([user, assistant, tool])
    return {"messages": messages, "nested": {"a": [{"b": [[f"token={_SECRET}"]]}]}}


def _assert_matches_stock(anonymize: Callable[[Any], Any]) -> None:
    stock = create_secret_anonymizer()
    for turn in (1, 2, 3, 3, 5):
        expected = stock(copy.deepcopy(_payload(turn)))
        actual = anonymize(copy.deepcopy(_payload(turn)))
        assert actual == expected
        assert _SECRET not in str(actual)


def test_cached_output_matches_stock_anonymizer() -> None:
    """Repeated and growing payloads redact exactly like the stock anonymizer."""
    _assert_matches_stock(create_cached_secret_anonymizer())


@pytest.mark.parametrize(
    ("max_entries", "max_chars"),
    [(1, 1_000_000), (1_000, 20), (2, 50)],
)
def test_cache_bounds_do_not_change_output(max_entries: int, max_chars: int) -> None:
    """Eviction and oversized strings still produce identical redaction."""
    processor = CachingRuleProcessor(
        list(DEFAULT_SECRET_RULES),
        max_entries=max_entries,
        max_chars=max_chars,
    )
    _assert_matches_stock(create_anonymizer(processor, max_depth=24))
