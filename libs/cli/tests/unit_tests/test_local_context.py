"""Tests for local context middleware."""

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from deepagents_cli.local_context import LocalContextMiddleware, LocalContextState


def _make_backend(output: str = "", exit_code: int = 0) -> Mock:
    """Create a mock backend with execute() returning the given output."""
    backend = Mock()
    result = Mock()
    result.output = output
    result.exit_code = exit_code
    backend.execute.return_value = result
    return backend


# Sample script output fragments for testing
SAMPLE_CONTEXT = (
    "## Local Context\n\n"
    "**Current Directory**: `/home/user/project`\n\n"
    "**Git**: Current branch `main`, main branch available: `main`, `master`,"
    " 1 uncommitted change\n\n"
    "**Runtimes**: Python 3.12.4, Node 20.11.0\n"
)

SAMPLE_CONTEXT_NO_GIT = (
    "## Local Context\n\n"
    "**Current Directory**: `/home/user/project`\n\n"
    "**Runtimes**: Python 3.12.4\n"
)


class TestLocalContextMiddleware:
    """Test local context middleware functionality."""

    def test_before_agent_stores_context(self) -> None:
        """Test before_agent runs script and stores output in state."""
        backend = _make_backend(output=SAMPLE_CONTEXT)
        middleware = LocalContextMiddleware(backend=backend)
        state: LocalContextState = {"messages": []}
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)

        assert result is not None
        assert "local_context" in result
        assert "## Local Context" in result["local_context"]
        assert "Current Directory" in result["local_context"]
        backend.execute.assert_called_once()

    def test_before_agent_skips_when_already_set(self) -> None:
        """Test before_agent returns None when local_context already exists."""
        backend = _make_backend(output=SAMPLE_CONTEXT)
        middleware = LocalContextMiddleware(backend=backend)
        state: LocalContextState = {
            "messages": [],
            "local_context": "already set",
        }
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)

        assert result is None
        backend.execute.assert_not_called()

    def test_before_agent_handles_script_failure(self) -> None:
        """Test before_agent returns None when script exits non-zero."""
        backend = _make_backend(output="", exit_code=1)
        middleware = LocalContextMiddleware(backend=backend)
        state: LocalContextState = {"messages": []}
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)

        assert result is None

    def test_before_agent_handles_empty_output(self) -> None:
        """Test before_agent returns None when script produces no output."""
        backend = _make_backend(output="   \n  ", exit_code=0)
        middleware = LocalContextMiddleware(backend=backend)
        state: LocalContextState = {"messages": []}
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)

        assert result is None

    def test_before_agent_handles_execute_exception(self) -> None:
        """Test before_agent returns None when backend.execute() raises."""
        backend = Mock()
        backend.execute.side_effect = RuntimeError("connection failed")
        middleware = LocalContextMiddleware(backend=backend)
        state: LocalContextState = {"messages": []}
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)

        assert result is None

    def test_before_agent_git_context(self) -> None:
        """Test that git info is preserved from script output."""
        backend = _make_backend(output=SAMPLE_CONTEXT)
        middleware = LocalContextMiddleware(backend=backend)
        state: LocalContextState = {"messages": []}
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)

        assert result is not None
        ctx = result["local_context"]
        assert "**Git**: Current branch `main`" in ctx
        assert "main branch available:" in ctx
        assert "`main`" in ctx
        assert "`master`" in ctx
        assert "1 uncommitted change" in ctx

    def test_before_agent_no_git(self) -> None:
        """Test output without git info."""
        backend = _make_backend(output=SAMPLE_CONTEXT_NO_GIT)
        middleware = LocalContextMiddleware(backend=backend)
        state: LocalContextState = {"messages": []}
        runtime: Any = Mock()

        result = middleware.before_agent(state, runtime)

        assert result is not None
        ctx = result["local_context"]
        assert "Current Directory" in ctx
        assert "**Git**:" not in ctx

    def test_wrap_model_call_with_local_context(self) -> None:
        """Test that wrap_model_call appends local context to system prompt."""
        backend = _make_backend()
        middleware = LocalContextMiddleware(backend=backend)

        request = Mock()
        request.system_prompt = "Base system prompt"
        request.state = {"local_context": SAMPLE_CONTEXT}

        overridden_request = Mock()
        request.override.return_value = overridden_request

        handler = Mock(return_value="response")

        result = middleware.wrap_model_call(request, handler)

        request.override.assert_called_once()
        call_args = request.override.call_args[1]
        assert "system_prompt" in call_args
        assert "Base system prompt" in call_args["system_prompt"]
        assert "Current branch `main`" in call_args["system_prompt"]

        handler.assert_called_once_with(overridden_request)
        assert result == "response"

    def test_wrap_model_call_without_local_context(self) -> None:
        """Test that wrap_model_call passes through when no local context."""
        backend = _make_backend()
        middleware = LocalContextMiddleware(backend=backend)

        request = Mock()
        request.system_prompt = "Base system prompt"
        request.state = {}

        handler = Mock(return_value="response")

        result = middleware.wrap_model_call(request, handler)

        request.override.assert_not_called()
        handler.assert_called_once_with(request)
        assert result == "response"

    @pytest.mark.asyncio
    async def test_awrap_model_call_with_local_context(self) -> None:
        """Test that awrap_model_call appends local context to system prompt."""
        backend = _make_backend()
        middleware = LocalContextMiddleware(backend=backend)

        request = Mock()
        request.system_prompt = "Base system prompt"
        request.state = {"local_context": SAMPLE_CONTEXT}

        overridden_request = Mock()
        request.override.return_value = overridden_request

        handler = AsyncMock(return_value="async response")

        result = await middleware.awrap_model_call(request, handler)

        request.override.assert_called_once()
        call_args = request.override.call_args[1]
        assert "system_prompt" in call_args
        assert "Base system prompt" in call_args["system_prompt"]
        assert "Current branch `main`" in call_args["system_prompt"]

        handler.assert_called_once_with(overridden_request)
        assert result == "async response"

    @pytest.mark.asyncio
    async def test_awrap_model_call_without_local_context(self) -> None:
        """Test that awrap_model_call passes through when no local context."""
        backend = _make_backend()
        middleware = LocalContextMiddleware(backend=backend)

        request = Mock()
        request.system_prompt = "Base system prompt"
        request.state = {}

        handler = AsyncMock(return_value="async response")

        result = await middleware.awrap_model_call(request, handler)

        request.override.assert_not_called()
        handler.assert_called_once_with(request)
        assert result == "async response"
