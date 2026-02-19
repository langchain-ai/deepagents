"""CLI-specific tests for compact_conversation tool (HITL gating, display).

Core compact tool logic tests live in the SDK at
`libs/deepagents/tests/unit_tests/middleware/test_compact_tool.py`.
"""

from __future__ import annotations

from unittest.mock import patch

from deepagents_cli.tool_display import format_tool_display


class TestHITLGating:
    """Test that compact_conversation HITL gating respects the constant."""

    def test_hitl_gating_when_enabled(self) -> None:
        """With REQUIRE_COMPACT_TOOL_APPROVAL=True, tool should be gated."""
        with patch("deepagents_cli.agent.REQUIRE_COMPACT_TOOL_APPROVAL", True):
            from deepagents_cli.agent import _add_interrupt_on

            result = _add_interrupt_on()
            assert "compact_conversation" in result


class TestDisplayFormatting:
    """Test tool display formatting for compact_conversation."""

    def test_display_formatting(self) -> None:
        """format_tool_display should return the expected string."""
        result = format_tool_display("compact_conversation", {})
        assert "compact_conversation()" in result
