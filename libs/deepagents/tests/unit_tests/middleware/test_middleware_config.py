"""Tests for MiddlewareConfig."""

from langchain.agents.middleware import TodoListMiddleware

from deepagents import MiddlewareConfig
from deepagents.middleware.config import resolve_middleware


class TestMiddlewareConfig:
    """Tests for MiddlewareConfig dataclass."""

    def test_default_config(self):
        """Test that default config has all middleware enabled."""
        config = MiddlewareConfig()
        assert config.todo_list is True
        assert config.filesystem is True
        assert config.subagents is True
        assert config.summarization is True
        assert config.prompt_caching is True
        assert config.patch_tool_calls is True

    def test_disable_single_middleware(self):
        """Test disabling a single middleware."""
        config = MiddlewareConfig(todo_list=False)
        assert config.todo_list is False
        assert config.filesystem is True  # Others remain enabled

    def test_disable_multiple_middleware(self):
        """Test disabling multiple middleware."""
        config = MiddlewareConfig(
            todo_list=False,
            summarization=False,
            prompt_caching=False,
        )
        assert config.todo_list is False
        assert config.summarization is False
        assert config.prompt_caching is False
        assert config.filesystem is True
        assert config.subagents is True
        assert config.patch_tool_calls is True

    def test_replace_with_custom_middleware(self):
        """Test replacing middleware with custom instance."""
        # Use TodoListMiddleware since it doesn't require a backend
        custom = TodoListMiddleware()
        config = MiddlewareConfig(todo_list=custom)
        assert config.todo_list is custom

    def test_disable_all_middleware(self):
        """Test disabling all middleware."""
        config = MiddlewareConfig(
            todo_list=False,
            filesystem=False,
            subagents=False,
            summarization=False,
            prompt_caching=False,
            patch_tool_calls=False,
        )
        assert config.todo_list is False
        assert config.filesystem is False
        assert config.subagents is False
        assert config.summarization is False
        assert config.prompt_caching is False
        assert config.patch_tool_calls is False


class TestResolveMiddleware:
    """Tests for the resolve_middleware helper function."""

    def test_resolve_true_returns_default(self):
        """Test that True returns the default middleware."""
        result = resolve_middleware(True, TodoListMiddleware)
        assert isinstance(result, TodoListMiddleware)

    def test_resolve_false_returns_none(self):
        """Test that False returns None."""
        result = resolve_middleware(False, TodoListMiddleware)
        assert result is None

    def test_resolve_custom_returns_custom(self):
        """Test that custom middleware is returned as-is."""
        custom = TodoListMiddleware()
        result = resolve_middleware(custom, lambda: TodoListMiddleware())
        assert result is custom

    def test_resolve_with_lambda_factory(self):
        """Test that lambda factory is called for True."""
        result = resolve_middleware(True, lambda: TodoListMiddleware())
        assert isinstance(result, TodoListMiddleware)
