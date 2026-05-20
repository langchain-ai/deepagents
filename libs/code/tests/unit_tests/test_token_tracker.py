"""Tests for token state persistence and display callbacks."""

from types import SimpleNamespace

from deepagents_code.app import DeepAgentsApp
from deepagents_code.token_state import TokenStateMiddleware, TokenTrackingState


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

    async def test_writes_to_local_cache(
        self, tmp_path, monkeypatch
    ):
        """Happy path: persists the count to the local cache."""
        from deepagents_code import context_tokens_cache
        from deepagents_code.textual_adapter import _persist_context_tokens

        monkeypatch.setattr(
            context_tokens_cache, "CONTEXT_TOKENS_DIR", tmp_path / "context_tokens"
        )
        config = {"configurable": {"thread_id": "t-1"}}

        await _persist_context_tokens(config, 4200)  # type: ignore[arg-type]

        assert context_tokens_cache.read_context_tokens("t-1") == 4200

    # OSError swallowing is now owned by `write_context_tokens` itself
    # (covered in `test_context_tokens_cache.py`); `_persist_context_tokens`
    # is just an async wrapper around it.
