"""CLI-specific tests for compact_conversation tool (HITL gating, display).

Core compact tool logic tests live in the SDK at
`libs/deepagents/tests/unit_tests/middleware/test_compact_tool.py`.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage

from deepagents_code.offload_middleware import CLICompactionMiddleware
from deepagents_code.tool_display import format_tool_display


class TestHITLGating:
    """Test that compact_conversation HITL gating respects the constant."""

    def test_hitl_gating_when_enabled(self) -> None:
        """With REQUIRE_COMPACT_TOOL_APPROVAL=True, tool should be gated."""
        with patch("deepagents_code.agent.REQUIRE_COMPACT_TOOL_APPROVAL", True):
            from deepagents_code.agent import _add_interrupt_on

            result = _add_interrupt_on()
            assert "compact_conversation" in result

    def test_hitl_gating_when_disabled(self) -> None:
        """With REQUIRE_COMPACT_TOOL_APPROVAL=False, tool should NOT be gated."""
        with patch("deepagents_code.agent.REQUIRE_COMPACT_TOOL_APPROVAL", False):
            from deepagents_code.agent import _add_interrupt_on

            result = _add_interrupt_on()
            assert "compact_conversation" not in result


class TestDisplayFormatting:
    """Test tool display formatting for compact_conversation."""

    def test_display_formatting(self) -> None:
        """format_tool_display should return the expected string."""
        result = format_tool_display("compact_conversation", {})
        assert "compact_conversation()" in result


class TestCLICompactionMiddleware:
    """Cover dcode's explicit `/offload` behavior layered over the SDK tool."""

    @staticmethod
    def _summarization() -> MagicMock:
        summarization = MagicMock()
        summarization._backend = object()
        summarization._apply_event_to_messages.side_effect = lambda messages, _event: (
            messages
        )
        summarization._determine_cutoff_index.return_value = 2
        summarization._partition_messages.side_effect = lambda messages, cutoff: (
            messages[:cutoff],
            messages[cutoff:],
        )
        summarization._acreate_summary = AsyncMock(return_value="Summary")
        summarization._aoffload_to_backend = AsyncMock(
            return_value="/conversation_history/thread.md"
        )
        summarization._build_new_messages_with_path.return_value = [
            HumanMessage(content="Summary")
        ]
        summarization._compute_state_cutoff.return_value = 2
        return summarization

    async def test_force_bypasses_sdk_eligibility_gate(self) -> None:
        """Forced compaction partitions directly even below the proactive gate."""
        summarization = self._summarization()
        middleware = CLICompactionMiddleware(summarization)
        runtime = MagicMock()
        runtime.context = None
        runtime.state = {"messages": [HumanMessage("one"), HumanMessage("two")]}
        runtime.tool_call_id = "tool-call"

        result = await middleware._arun_forced_compact(runtime)

        summarization._is_eligible_for_compaction.assert_not_called()
        summarization._acreate_summary.assert_awaited_once()
        assert result.update is not None
        assert result.update["_summarization_event"]["cutoff_index"] == 2

    def test_runtime_model_builds_matching_summarizer(self) -> None:
        """A `/model` override selects the summarizer used by `/offload`."""
        startup = self._summarization()
        middleware = CLICompactionMiddleware(startup)
        runtime = MagicMock()
        runtime.context = {
            "model": "provider:active-model",
            "model_params": {"temperature": 0},
        }
        active_model = object()
        result = SimpleNamespace(model=active_model)
        selected = MagicMock()

        with (
            patch(
                "deepagents_code.config.create_model", return_value=result
            ) as create_model,
            patch(
                "deepagents_code.offload_middleware.create_summarization_middleware",
                return_value=selected,
            ) as create_summarization,
        ):
            actual = middleware._summarization_for_runtime(runtime)

        assert actual is selected
        create_model.assert_called_once_with(
            "provider:active-model", extra_kwargs={"temperature": 0}
        )
        create_summarization.assert_called_once_with(active_model, startup._backend)
