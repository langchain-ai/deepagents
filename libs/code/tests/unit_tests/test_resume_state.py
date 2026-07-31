"""Tests for resume-state persistence and token display callbacks."""

from types import SimpleNamespace
from typing import Any, cast, get_type_hints

import pytest
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code._session_stats import SessionStats
from deepagents_code.app import DeepAgentsApp
from deepagents_code.resume_state import (
    ResumeState,
    ResumeStateMiddleware,
    _extract_context_tokens,
    coerce_goal_proposal_kind,
    coerce_goal_status,
)


def _runtime(context: dict[str, str | None] | None) -> SimpleNamespace:
    """Build a stand-in `Runtime` exposing only `.context`."""
    return SimpleNamespace(context=context)


class TestResumeState:
    def test_state_has_context_tokens_field(self):
        """ResumeState declares the `_context_tokens` channel."""
        assert "_context_tokens" in ResumeState.__annotations__

    def test_state_has_model_spec_field(self):
        """ResumeState declares the `_model_spec` channel."""
        assert "_model_spec" in ResumeState.__annotations__

    def test_state_has_model_params_field(self):
        """ResumeState declares the `_model_params` channel."""
        assert "_model_params" in ResumeState.__annotations__

    def test_sticky_rubric_field_is_private(self):
        """Persistent TUI rubrics must not leak through the public schema."""
        # `_sticky_rubric` is inherited from `GoalRubricChannels`, so resolve the
        # full (inherited) hints the way LangGraph does rather than reading
        # own-keys-only `__annotations__`. `get_type_hints` resolves the marker to
        # its real object (`PrivateStateAttr`), so assert membership of that
        # sentinel rather than matching the source text.
        hints = get_type_hints(ResumeState, include_extras=True)
        metadata = getattr(hints["_sticky_rubric"], "__metadata__", ())
        assert PrivateStateAttr in metadata

    def test_pending_goal_kind_is_private(self) -> None:
        """Proposal mode should persist without entering public graph I/O."""
        hints = get_type_hints(ResumeState, include_extras=True)
        metadata = getattr(hints["_pending_goal_kind"], "__metadata__", ())
        assert PrivateStateAttr in metadata

    def test_pending_goal_request_id_is_private(self) -> None:
        """Proposal correlation must persist without entering public graph I/O."""
        hints = get_type_hints(ResumeState, include_extras=True)
        metadata = getattr(hints["_pending_goal_request_id"], "__metadata__", ())
        assert PrivateStateAttr in metadata

    def test_middleware_exposes_state_schema(self):
        """ResumeStateMiddleware registers the correct state schema."""
        assert ResumeStateMiddleware.state_schema is ResumeState


class TestCoerceGoalStatus:
    """Tests for `coerce_goal_status`."""

    def test_returns_known_statuses(self) -> None:
        assert coerce_goal_status("active") == "active"
        assert coerce_goal_status("paused") == "paused"
        assert coerce_goal_status("blocked") == "blocked"
        assert coerce_goal_status("complete") == "complete"

    def test_unknown_string_coerces_to_none(self) -> None:
        assert coerce_goal_status("deleted") is None
        assert coerce_goal_status("") is None

    def test_non_string_coerces_to_none(self) -> None:
        assert coerce_goal_status(None) is None
        assert coerce_goal_status(123) is None
        assert coerce_goal_status(["active"]) is None


class TestCoerceGoalProposalKind:
    """Tests for persisted pending-review mode coercion."""

    def test_returns_known_kinds(self) -> None:
        assert coerce_goal_proposal_kind("create") == "create"
        assert coerce_goal_proposal_kind("amend") == "amend"

    def test_unknown_value_coerces_to_none(self) -> None:
        assert coerce_goal_proposal_kind("replace") is None
        assert coerce_goal_proposal_kind(None) is None


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

    def test_on_tokens_update_sets_cache_and_calls_display(self):
        """_on_tokens_update should set the local cache and update the status bar."""
        display_calls: list[int] = []

        class FakeApp:
            _context_tokens: int = 0
            _status_bar = None

            def _update_tokens(self, count: int) -> None:
                display_calls.append(count)

            def _on_tokens_update(self, count: int) -> None:
                self._context_tokens = count
                self._update_tokens(count)

        app = FakeApp()
        app._on_tokens_update(4200)

        assert app._context_tokens == 4200
        assert display_calls == [4200]

    def test_show_tokens_restores_cached_value(self):
        """_show_tokens should re-display the cached value."""
        display_calls: list[int] = []

        class FakeApp:
            _context_tokens: int = 1500

            def _update_tokens(self, count: int) -> None:
                display_calls.append(count)

            def _show_tokens(self) -> None:
                self._update_tokens(self._context_tokens)

        app = FakeApp()
        app._show_tokens()

        assert display_calls == [1500]

    def test_show_tokens_preserves_approximate_marker_without_fresh_usage(self):
        """Turns without usage metadata should not clear a stale-token marker."""
        display_calls: list[tuple[int, bool]] = []

        def update_tokens(count: int, *, approximate: bool = False) -> None:
            display_calls.append((count, approximate))

        app = SimpleNamespace(
            _context_tokens=1500,
            _tokens_approximate=True,
            _update_tokens=update_tokens,
        )

        DeepAgentsApp._show_tokens(app, approximate=False)  # ty: ignore

        assert app._tokens_approximate is True
        assert display_calls == [(1500, True)]

    def test_reset_clears_cache(self):
        """Resetting (e.g. /clear) should zero the cache and display."""
        display_calls: list[int] = []

        class FakeApp:
            _context_tokens: int = 3000

            def _update_tokens(self, count: int) -> None:
                display_calls.append(count)

        app = FakeApp()
        app._context_tokens = 0
        app._update_tokens(0)

        assert app._context_tokens == 0
        assert display_calls == [0]


class TestCostDisplayCallbacks:
    """Verify persisted thread cost is restored and accumulated in the TUI."""

    def test_payload_restores_valid_session_cost(self) -> None:
        payload = DeepAgentsApp._goal_rubric_payload_from_state(
            {"_session_cost_usd": 1.25},
            messages=[],
            context_tokens=0,
            model_spec="",
        )
        assert payload.session_cost_usd == pytest.approx(1.25)

    def test_payload_rejects_invalid_session_cost(self) -> None:
        for value in (-1, float("nan"), "1.25", True, 10**1000):
            payload = DeepAgentsApp._goal_rubric_payload_from_state(
                {"_session_cost_usd": value},
                messages=[],
                context_tokens=0,
                model_spec="",
            )
            assert payload.session_cost_usd == pytest.approx(0.0)

    def test_server_total_replaces_the_displayed_value(self) -> None:
        """The graph's absolute total is adopted outright, never added to."""
        app = DeepAgentsApp()
        app._session_cost_usd = 1.0

        app._set_session_cost(1.25)

        assert app._session_cost_usd == pytest.approx(1.25)
        assert app._displayed_cost_usd == pytest.approx(1.25)

    def test_streamed_estimate_shows_ahead_of_the_server_total(self) -> None:
        """Spend the graph has not reported yet still moves the display."""
        app = DeepAgentsApp()
        app._set_session_cost(1.0)

        app._add_provisional_cost(0.25)

        assert app._displayed_cost_usd == pytest.approx(1.25)
        # The server-owned figure is untouched by a client estimate.
        assert app._session_cost_usd == pytest.approx(1.0)

    def test_server_total_supersedes_provisional_estimates(self) -> None:
        """A provisional estimate cannot be counted twice once a total lands."""
        app = DeepAgentsApp()
        app._set_session_cost(1.0)
        app._add_provisional_cost(0.25)

        app._set_session_cost(1.25)

        assert app._displayed_cost_usd == pytest.approx(1.25)

    def test_zero_cost_does_not_change_running_total(self) -> None:
        app = DeepAgentsApp()
        app._session_cost_usd = 1.0

        app._add_provisional_cost(0.0)

        assert app._displayed_cost_usd == pytest.approx(1.0)

    def test_total_for_an_inactive_thread_is_discarded(self) -> None:
        """A total in flight during `/force-clear` must not land on the new thread."""
        app = DeepAgentsApp()
        app._lc_thread_id = "thread-1"
        app._set_session_cost(1.0, thread_id="thread-1")

        app._set_session_cost(99.0, thread_id="thread-0")

        assert app._displayed_cost_usd == pytest.approx(1.0)

    def test_total_for_the_active_thread_is_applied(self) -> None:
        app = DeepAgentsApp()
        app._lc_thread_id = "thread-1"

        app._set_session_cost(2.5, thread_id="thread-1")

        assert app._displayed_cost_usd == pytest.approx(2.5)

    def test_unattributed_total_is_applied(self) -> None:
        """A restored checkpoint read names no thread and must still apply."""
        app = DeepAgentsApp()
        app._lc_thread_id = "thread-1"

        app._set_session_cost(3.0)

        assert app._displayed_cost_usd == pytest.approx(3.0)

    def test_downward_reprice_lowers_the_provisional_display(self) -> None:
        """A re-priced request must not strand the estimate it superseded."""
        app = DeepAgentsApp()
        app._set_session_cost(1.0)
        app._add_provisional_cost(0.1545)

        app._add_provisional_cost(-0.1512)

        assert app._displayed_cost_usd == pytest.approx(1.0033)

    def test_provisional_total_never_goes_below_the_server_total(self) -> None:
        """Clamping the accumulator, not the increment, keeps retractions honest."""
        app = DeepAgentsApp()
        app._set_session_cost(1.0)
        app._add_provisional_cost(0.25)

        app._add_provisional_cost(-10.0)

        assert app._displayed_cost_usd == pytest.approx(1.0)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), "0.5", True, None])
    def test_malformed_provisional_delta_is_ignored(self, value: object) -> None:
        app = DeepAgentsApp()
        app._set_session_cost(1.0)

        app._add_provisional_cost(cast("float", value))

        assert app._displayed_cost_usd == pytest.approx(1.0)

    def test_streamed_pricing_health_is_remembered(self) -> None:
        """Pricing runs server-side, so only the event can report it broken."""
        app = DeepAgentsApp()
        app._lc_thread_id = "thread-1"

        app._set_session_cost(1.0, thread_id="thread-1", pricing_ok=False)

        assert app._pricing_is_broken() is True

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

    def test_state_without_a_cost_channel_leaves_the_display_alone(self) -> None:
        app = DeepAgentsApp()
        app._set_session_cost(1.0)

        app._sync_session_cost_from_state({"messages": []})

        assert app._displayed_cost_usd == pytest.approx(1.0)

    def test_cost_summary_guides_fresh_thread(self) -> None:
        app = DeepAgentsApp()

        summary = app._format_cost_summary()

        assert summary == "No model usage recorded for this thread yet."

    def test_cost_summary_explains_unreported_usage_after_completed_turn(self) -> None:
        app = DeepAgentsApp()
        app._thread_has_completed_turn = True

        summary = app._format_cost_summary()

        assert summary == (
            "We couldn't track the requests so far because the provider didn't "
            "report token usage. Requests from providers that report usage will "
            "appear here."
        )

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

    def test_cost_summary_guides_thread_with_unpriced_usage(self) -> None:
        stats = SessionStats()
        stats.record_request("unknown-model", 100, 10, provider="example")
        app = DeepAgentsApp()
        app._thread_stats = stats

        summary = app._format_cost_summary()

        assert summary == (
            "We couldn't calculate costs for the requests so far because pricing "
            "isn't available for the models used. Unpriced usage may still count "
            "toward subscription limits or incur charges."
        )

    def test_cost_summary_blames_the_pricing_install_when_it_failed_to_load(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken `genai-prices` makes every model unpriceable.

        Reporting that as missing catalog coverage points the user at their
        model choice instead of the one thing they can actually fix.
        """
        from deepagents_code import cost_tracking

        monkeypatch.setattr(cost_tracking, "_PRICING_UNAVAILABLE", True)
        stats = SessionStats()
        stats.record_request("gpt-5.5", 100, 10, provider="openai")
        app = DeepAgentsApp()
        app._thread_stats = stats

        summary = app._format_cost_summary()

        assert "pricing data failed to load" in summary
        assert "Reinstalling Deep Agents Code" in summary
        assert "pricing isn't available for the models used" not in summary

    def test_cost_summary_includes_total_and_type_model_breakdown(self) -> None:
        stats = SessionStats()
        stats.record_request(
            "gpt-5.5",
            1_000,
            100,
            provider="openai",
            cost_usd=0.32,
            kind="assistant",
        )
        stats.record_request(
            "gpt-5.5",
            200,
            20,
            provider="openai",
            cost_usd=0.10,
            kind="offload",
        )
        process_stats = SessionStats()
        process_stats.record_request(
            "claude-sonnet-4-6",
            1_000,
            100,
            provider="anthropic",
            cost_usd=2.0,
        )
        app = DeepAgentsApp()
        app._session_cost_usd = 0.42
        app._thread_stats = stats
        app._session_stats = process_stats

        summary = app._format_cost_summary()

        assert "Estimated thread cost: $0.42" in summary
        assert "By type since this thread was loaded:" in summary
        assert "Assistant: $0.32" in summary
        assert "Offload: $0.10" in summary
        assert "openai:gpt-5.5: $0.42" in summary
        assert "claude-sonnet-4-6" not in summary
        assert "detailed usage metadata was unavailable" not in summary

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

    def test_missing_current_details_are_not_labeled_as_restored(self) -> None:
        """Fresh-thread usage without details gets only the current warning."""
        app = DeepAgentsApp()
        app._set_session_cost(0.20)

        summary = app._format_cost_summary()

        assert "Estimated thread cost: $0.20" in summary
        assert "restored usage" not in summary.lower()
        assert "detailed usage metadata was unavailable" in summary

    def test_cost_summary_marks_restored_usage_outside_breakdown(self) -> None:
        stats = SessionStats()
        stats.record_request(
            "gpt-5.5",
            1_000,
            100,
            provider="openai",
            cost_usd=0.42,
        )
        app = DeepAgentsApp()
        app._reset_thread_usage(1.0)
        app._thread_stats = stats
        app._set_session_cost(1.42)

        summary = app._format_cost_summary()

        assert "Estimated thread cost: $1.42" in summary
        assert "openai:gpt-5.5: $0.42" in summary
        assert "Restored usage is included only in the total above." in summary

    def test_cost_summary_distinguishes_zero_priced_and_unknown_requests(self) -> None:
        stats = SessionStats()
        stats.record_request(
            "free-model",
            100,
            10,
            provider="example",
            cost_usd=0.0,
        )
        stats.record_request("unknown-model", 100, 10, provider="example")
        app = DeepAgentsApp()
        app._thread_stats = stats

        summary = app._format_cost_summary()

        assert "Estimated cost for priced requests: $0.00" in summary
        assert "1 of 2 recorded requests is included." in summary
        assert "example:unknown-model — 1 request" in summary
        assert "The full thread cost may be higher." in summary
        assert "example:free-model: $0.00" in summary
        assert "The recorded request was priced at $0.00." not in summary

    @pytest.mark.parametrize(
        ("request_count", "expected"),
        [
            (1, "The recorded request was priced at $0.00."),
            (2, "All 2 recorded requests were priced at $0.00."),
        ],
    )
    def test_cost_summary_explains_fully_priced_zero_cost(
        self,
        request_count: int,
        expected: str,
    ) -> None:
        stats = SessionStats()
        for _ in range(request_count):
            stats.record_request(
                "free-model",
                100,
                10,
                provider="example",
                cost_usd=0.0,
            )
        app = DeepAgentsApp()
        app._thread_stats = stats

        summary = app._format_cost_summary()

        assert "Estimated thread cost: $0.00" in summary
        assert "example:free-model: $0.00" in summary
        assert expected in summary
        assert "Your provider's bill may still differ" in summary

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
