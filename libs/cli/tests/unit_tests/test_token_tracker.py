"""Tests for token state persistence and display callbacks."""

from types import SimpleNamespace

from deepagents_cli.app import DeepAgentsApp
from deepagents_cli.token_state import TokenStateMiddleware, TokenTrackingState


class TestTokenTrackingState:
    def test_state_has_context_tokens_field(self):
        """TokenTrackingState declares the `_context_tokens` channel."""
        annotations = TokenTrackingState.__annotations__
        assert "_context_tokens" in annotations

    def test_middleware_exposes_state_schema(self):
        """TokenStateMiddleware registers the correct state schema."""
        assert TokenStateMiddleware.state_schema is TokenTrackingState


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


class TestPersistContextTokens:
    """Tests for the `_persist_context_tokens` helper."""

    async def test_calls_aupdate_state_with_token_count(self):
        """Happy path: persists the count via `aupdate_state`."""
        from unittest.mock import AsyncMock

        from deepagents_cli.textual_adapter import _persist_context_tokens

        agent = AsyncMock()
        config = {"configurable": {"thread_id": "t-1"}}

        await _persist_context_tokens(agent, config, 4200)  # type: ignore[arg-type]

        agent.aupdate_state.assert_awaited_once_with(config, {"_context_tokens": 4200})

    async def test_suppresses_exceptions(self):
        """Failures should be swallowed (non-critical persistence)."""
        from unittest.mock import AsyncMock

        from deepagents_cli.textual_adapter import _persist_context_tokens

        agent = AsyncMock()
        agent.aupdate_state.side_effect = RuntimeError("checkpointer down")
        config = {"configurable": {"thread_id": "t-1"}}

        # Should not raise
        await _persist_context_tokens(agent, config, 1000)  # type: ignore[arg-type]
