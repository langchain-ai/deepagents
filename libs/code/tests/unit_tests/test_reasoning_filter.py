"""Regression tests for filtering model-internal reasoning content blocks."""

from __future__ import annotations

import pytest

from deepagents_code.textual_adapter import _is_reasoning_block


class TestIsReasoningBlock:
    """Verify reasoning/thinking blocks are recognized regardless of shape."""

    def test_plain_text_block_is_not_reasoning(self) -> None:
        assert _is_reasoning_block({"type": "text", "text": "Hello."}) is False

    @pytest.mark.parametrize("block_type", ["reasoning", "thinking"])
    def test_explicit_reasoning_type(self, block_type: str) -> None:
        block = {"type": block_type, "reasoning": "Need continue fallback runnable."}
        assert _is_reasoning_block(block) is True

    @pytest.mark.parametrize(
        "fragment",
        [
            "Need continue fallback runnable.",
            "Need offset earlier.",
            "Need class maybe start near top.",
        ],
    )
    def test_mislabeled_text_with_is_reasoning_flag(self, fragment: str) -> None:
        block = {"type": "text", "text": fragment, "is_reasoning": True}
        assert _is_reasoning_block(block) is True

    def test_text_block_with_reasoning_field(self) -> None:
        block = {
            "type": "text",
            "text": "Need continue fallback runnable.",
            "reasoning": "Need continue fallback runnable.",
        }
        assert _is_reasoning_block(block) is True

    def test_reasoning_summary_block_shape(self) -> None:
        # Shape emitted by langchain-core's openai responses translator after
        # exploding a reasoning summary.
        block = {
            "type": "reasoning",
            "reasoning": "Need offset earlier.",
            "id": "rs_abc",
        }
        assert _is_reasoning_block(block) is True


class TestFilterLoop:
    """Simulate the streaming loop's block filter and assert output."""

    def _filter(self, blocks: list[dict[str, object]]) -> list[str]:
        appended: list[str] = []
        for block in blocks:
            if _is_reasoning_block(block):
                continue
            if block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    appended.append(str(text))
        return appended

    def test_only_user_visible_text_is_appended(self) -> None:
        blocks: list[dict[str, object]] = [
            {"type": "text", "text": "Hello."},
            {"type": "reasoning", "reasoning": "Need continue fallback runnable."},
            {
                "type": "text",
                "text": "Need offset earlier.",
                "is_reasoning": True,
            },
            {
                "type": "text",
                "text": "Need class maybe start near top.",
                "reasoning": "Need class maybe start near top.",
            },
            {"type": "thinking", "text": "internal scratchpad"},
        ]
        assert self._filter(blocks) == ["Hello."]
