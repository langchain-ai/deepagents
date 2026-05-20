"""Tests for token state persistence and display callbacks."""

from types import SimpleNamespace
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code.app import DeepAgentsApp
from deepagents_code.token_state import (
    TokenStateMiddleware,
    TokenTrackingState,
    _extract_context_tokens,
)


class TestTokenTrackingState:
    def test_state_has_context_tokens_field(self):
        """TokenTrackingState declares the `_context_tokens` channel."""
        annotations = TokenTrackingState.__annotations__
        assert "_context_tokens" in annotations

    def test_middleware_exposes_state_schema(self):
        """TokenStateMiddleware registers the correct state schema."""
        assert TokenStateMiddleware.state_schema is TokenTrackingState


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
        middleware = TokenStateMiddleware()
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
        result = middleware.after_model(state, runtime=None)  # type: ignore[arg-type]
        assert result == {"_context_tokens": 1700}

    async def test_returns_none_when_no_ai_message(self) -> None:
        middleware = TokenStateMiddleware()
        state: dict[str, Any] = {"messages": [HumanMessage(content="hi")]}
        result = middleware.after_model(state, runtime=None)  # type: ignore[arg-type]
        assert result is None

    async def test_returns_none_when_last_ai_lacks_usage(self) -> None:
        middleware = TokenStateMiddleware()
        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content="hi"),
                AIMessage(content="no usage info"),
            ],
        }
        result = middleware.after_model(state, runtime=None)  # type: ignore[arg-type]
        assert result is None

    async def test_handles_empty_messages(self) -> None:
        middleware = TokenStateMiddleware()
        result = middleware.after_model({"messages": []}, runtime=None)  # type: ignore[arg-type]
        assert result is None

    async def test_skips_intervening_tool_messages(self) -> None:
        """Picks up the most recent AIMessage even when followed by tool turns."""
        from langchain_core.messages import ToolMessage

        middleware = TokenStateMiddleware()
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
        result = middleware.after_model(state, runtime=None)  # type: ignore[arg-type]
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

        DeepAgentsApp._show_tokens(app, approximate=False)  # type: ignore[arg-type]

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
