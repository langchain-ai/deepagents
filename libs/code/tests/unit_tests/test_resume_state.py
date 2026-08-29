"""Tests for resume-state persistence and token display callbacks."""

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code._session_stats import SessionStats
from deepagents_code.app import DeepAgentsApp
from deepagents_code.resume_state import (
    ResumeStateMiddleware,
    _extract_context_tokens,
    coerce_goal_status,
    coerce_model_spec,
)


def _runtime(context: dict[str, str | None] | None) -> SimpleNamespace:
    """Build a stand-in `Runtime` exposing only `.context`."""
    return SimpleNamespace(context=context)


class TestCoerceRubricModelSpec:
    """Tests for checkpointed grader model coercion."""

    def test_accepts_nonblank_string(self) -> None:
        assert coerce_model_spec(" openai:gpt-5.5 ") == "openai:gpt-5.5"

    @pytest.mark.parametrize("value", [None, "", "   ", 1, {}])
    def test_rejects_malformed_values(self, value: object) -> None:
        assert coerce_model_spec(value) is None


class TestCoerceGoalStatus:
    """Tests for `coerce_goal_status`."""

    def test_returns_known_statuses(self) -> None:
        assert coerce_goal_status("active") == "active"
        assert coerce_goal_status("paused") == "paused"
        assert coerce_goal_status("blocked") == "blocked"
        assert coerce_goal_status("complete") == "complete"


class TestCoerceGoalProposalKind:
    """Tests for persisted pending-review mode coercion."""


class TestExtractContextTokens:
    """Tests for `_extract_context_tokens`."""

    def test_prefers_input_plus_output(self) -> None:
        msg = AIMessage(
            content="hi",
            usage_metadata={
                "input_tokens": 100,
                "output_tokens": 25,
                "total_tokens": 200,  # deliberately inconsistent
            },
        )
        assert _extract_context_tokens(msg) == 125

    def test_falls_back_to_total_tokens(self) -> None:
        msg = AIMessage(
            content="hi",
            usage_metadata={
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 999,
            },
        )
        assert _extract_context_tokens(msg) == 999

    def test_returns_none_without_usage_metadata(self) -> None:
        msg = AIMessage(content="hi")
        assert _extract_context_tokens(msg) is None

    def test_returns_none_for_zero_usage(self) -> None:
        msg = AIMessage(
            content="hi",
            usage_metadata={
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )
        assert _extract_context_tokens(msg) is None


class TestAfterModelHook:
    """Tests for the `after_model` persistence hook."""

    async def test_writes_context_tokens_from_last_ai_message(self) -> None:
        middleware = ResumeStateMiddleware()
        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(
                    content="response",
                    usage_metadata={
                        "input_tokens": 1500,
                        "output_tokens": 200,
                        "total_tokens": 1700,
                    },
                ),
            ],
        }
        result = middleware.after_model(state, _runtime(None))  # ty: ignore
        assert result == {"_context_tokens": 1700}

    async def test_does_not_write_model_spec_from_context(self) -> None:
        """Model metadata is written by ConfigurableModelMiddleware."""
        middleware = ResumeStateMiddleware()
        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(
                    content="response",
                    usage_metadata={
                        "input_tokens": 1500,
                        "output_tokens": 200,
                        "total_tokens": 1700,
                    },
                ),
            ],
        }
        runtime = _runtime({"model": "openai:gpt-5.1"})
        result = middleware.after_model(state, runtime)  # ty: ignore
        assert result == {"_context_tokens": 1700}

    async def test_returns_none_when_no_ai_message(self) -> None:
        middleware = ResumeStateMiddleware()
        state: dict[str, Any] = {"messages": [HumanMessage(content="hi")]}
        result = middleware.after_model(state, _runtime(None))  # ty: ignore
        assert result is None

    async def test_returns_none_when_last_ai_lacks_usage(self) -> None:
        middleware = ResumeStateMiddleware()
        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content="no usage info"),
            ],
        }
        result = middleware.after_model(state, _runtime(None))  # ty: ignore
        assert result is None

    async def test_handles_empty_messages(self) -> None:
        middleware = ResumeStateMiddleware()
        result = middleware.after_model({"messages": []}, _runtime(None))  # ty: ignore
        assert result is None

    async def test_skips_intervening_tool_messages(self) -> None:
        """Picks up the most recent AIMessage even when followed by tool turns."""
        from langchain_core.messages import ToolMessage

        middleware = ResumeStateMiddleware()
        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(
                    content="older",
                    usage_metadata={
                        "input_tokens": 100,
                        "output_tokens": 10,
                        "total_tokens": 110,
                    },
                ),
                ToolMessage(content="tool out", tool_call_id="t1"),
                AIMessage(
                    content="newer",
                    usage_metadata={
                        "input_tokens": 500,
                        "output_tokens": 50,
                        "total_tokens": 550,
                    },
                ),
            ],
        }
        result = middleware.after_model(state, _runtime(None))  # ty: ignore
        assert result == {"_context_tokens": 550}


class TestTokenDisplayCallbacks:
    """Verify the callback-based token tracking that replaced TextualTokenTracker."""


class TestCostDisplayCallbacks:
    """Verify persisted thread cost is restored and accumulated in the TUI."""

    def test_unreported_pricing_health_leaves_the_last_value(self) -> None:
        """A checkpoint read says nothing about pricing and must not erase it."""
        app = DeepAgentsApp()
        app._lc_thread_id = "thread-1"
        app._set_session_cost(1.0, thread_id="thread-1", pricing_ok=False)

        app._set_session_cost(2.0)

        assert app._pricing_is_broken() is True

    def test_committed_state_lowers_an_optimistic_display(self) -> None:
        """The client defers to the checkpoint instead of pushing its own total."""
        app = DeepAgentsApp()
        app._set_session_cost(1.0)
        app._add_provisional_cost(0.5)

        app._sync_session_cost_from_state({"_session_cost_usd": 1.1})

        assert app._displayed_cost_usd == pytest.approx(1.1)

    async def test_resumed_zero_cost_usage_is_not_reported_as_unused(self) -> None:
        """Checkpoint history preserves usage when its total cannot prove it."""
        from deepagents_code.app import _ThreadHistoryPayload

        app = DeepAgentsApp(thread_id="thread-1")
        payload = _ThreadHistoryPayload(
            messages=[],
            context_tokens=0,
            model_spec="",
            session_cost_usd=0.0,
            transcript_messages=(AIMessage(content=""),),
        )

        await app._load_thread_history(
            thread_id="thread-1",
            preloaded_payload=payload,
        )

        assert app._thread_stats.request_count == 0
        assert app._format_cost_summary() == (
            "Cost estimate unavailable\n\n"
            "Earlier model usage was restored for this thread, but its request "
            "and pricing details were not persisted. This does not mean the usage "
            "was free."
        )

    def test_cost_summary_warns_when_current_details_are_incomplete(self) -> None:
        """Checkpoint spend missing from streamed stats is called out."""
        stats = SessionStats()
        stats.record_request(
            "gpt-5.5",
            1_000,
            100,
            provider="openai",
            cost_usd=0.32,
        )
        app = DeepAgentsApp()
        app._reset_thread_usage(1.0)
        app._thread_stats = stats
        # The graph charged another $0.10 that the client message stream did
        # not expose, in addition to the represented $0.32.
        app._set_session_cost(1.42)

        summary = app._format_cost_summary()

        assert "Estimated thread cost: $1.42" in summary
        assert "openai:gpt-5.5: $0.32" in summary
        assert (
            "Some current-session usage is included only in the total because "
            "detailed usage metadata was unavailable."
        ) in summary

    async def test_checkpoint_reconcile_never_writes_cost(self) -> None:
        """The client reads the graph's total; it never back-fills the channel."""
        from unittest.mock import AsyncMock

        app = DeepAgentsApp()
        app._agent = object()
        app._lc_thread_id = "thread-1"
        app._set_session_cost(1.0)
        app._add_provisional_cost(0.25)
        app._get_thread_state_values = AsyncMock(
            return_value={"_session_cost_usd": 1.5}
        )
        app._aupdate_thread_state = AsyncMock()

        await app._sync_session_cost_from_checkpoint()

        assert app._displayed_cost_usd == pytest.approx(1.5)
        app._aupdate_thread_state.assert_not_awaited()

    async def test_failed_state_read_keeps_the_displayed_cost(self) -> None:
        from unittest.mock import AsyncMock

        app = DeepAgentsApp()
        app._agent = object()
        app._lc_thread_id = "thread-1"
        app._set_session_cost(1.0)
        app._get_thread_state_values = AsyncMock(side_effect=RuntimeError("no server"))

        await app._sync_session_cost_from_checkpoint()

        assert app._displayed_cost_usd == pytest.approx(1.0)
