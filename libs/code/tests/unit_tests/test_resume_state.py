"""Tests for resume-state persistence and token display callbacks."""

from types import SimpleNamespace
from typing import Any, get_type_hints

from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code.app import DeepAgentsApp
from deepagents_code.resume_state import (
    ResumeState,
    ResumeStateMiddleware,
    _extract_context_tokens,
    _extract_model_spec,
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

    def test_middleware_exposes_state_schema(self):
        """ResumeStateMiddleware registers the correct state schema."""
        assert ResumeStateMiddleware.state_schema is ResumeState


class TestCoerceGoalStatus:
    """Tests for `coerce_goal_status`."""

    def test_returns_known_statuses(self) -> None:
        assert coerce_goal_status("active") == "active"
        assert coerce_goal_status("blocked") == "blocked"
        assert coerce_goal_status("complete") == "complete"

    def test_unknown_string_coerces_to_none(self) -> None:
        assert coerce_goal_status("deleted") is None
        assert coerce_goal_status("") is None

    def test_non_string_coerces_to_none(self) -> None:
        assert coerce_goal_status(None) is None
        assert coerce_goal_status(123) is None
        assert coerce_goal_status(["active"]) is None


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


class TestExtractModelSpec:
    """Tests for `_extract_model_spec`."""

    def test_returns_effective_model_from_context(self) -> None:
        runtime = _runtime({"effective_model": "anthropic:claude-sonnet-4-5"})
        assert _extract_model_spec(runtime) == "anthropic:claude-sonnet-4-5"  # ty: ignore

    def test_returns_none_when_context_missing(self) -> None:
        assert _extract_model_spec(_runtime(None)) is None  # ty: ignore

    def test_returns_none_when_field_absent(self) -> None:
        assert _extract_model_spec(_runtime({"model": "x"})) is None  # ty: ignore

    def test_returns_none_for_blank_or_nonstring(self) -> None:
        assert _extract_model_spec(_runtime({"effective_model": ""})) is None  # ty: ignore
        assert _extract_model_spec(_runtime({"effective_model": None})) is None  # ty: ignore

    def test_returns_none_when_runtime_is_none(self) -> None:
        assert _extract_model_spec(None) is None  # ty: ignore


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

    async def test_writes_model_spec_from_context(self) -> None:
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
        runtime = _runtime({"effective_model": "openai:gpt-5.1"})
        result = middleware.after_model(state, runtime)  # ty: ignore
        assert result == {
            "_context_tokens": 1700,
            "_model_spec": "openai:gpt-5.1",
        }

    async def test_writes_model_spec_without_token_usage(self) -> None:
        """Model spec is recorded even when the AI message reports no usage."""
        middleware = ResumeStateMiddleware()
        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content="no usage info"),
            ],
        }
        runtime = _runtime({"effective_model": "openai:gpt-5.1"})
        result = middleware.after_model(state, runtime)  # ty: ignore
        assert result == {"_model_spec": "openai:gpt-5.1"}

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
