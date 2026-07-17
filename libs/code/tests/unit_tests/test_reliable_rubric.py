"""Tests for rubric grader model retries."""

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from deepagents.middleware.rubric import GraderResponse, RubricState
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code.config import CLI_MAX_RETRIES_KEY
from deepagents_code.model_retry import CodeModelRetryMiddleware
from deepagents_code.reliable_rubric import ReliableRubricMiddleware


def _read_error() -> httpx.ReadError:
    return httpx.ReadError(
        "connection closed while reading",
        request=httpx.Request("POST", "https://grader.test"),
    )


def _state(*, retries: int | None = None) -> RubricState:
    state: dict[str, Any] = {
        "rubric": "tests pass",
        "messages": [
            HumanMessage(content="implement it"),
            AIMessage(content="implementation complete"),
        ],
    }
    if retries is not None:
        state["_model_params"] = {CLI_MAX_RETRIES_KEY: retries}
    return cast("RubricState", state)


def _satisfied_result() -> dict[str, Any]:
    return {
        "structured_response": GraderResponse(
            result="satisfied",
            explanation="all checks pass",
            criteria=[],
        )
    }


class TestGraderConstruction:
    def test_string_model_uses_dcode_retry_configuration(self) -> None:
        middleware = ReliableRubricMiddleware(
            model="openai:gpt-5.5",
            model_retry_override=0,
        )
        model = MagicMock()
        grader = MagicMock()

        with (
            patch(
                "deepagents_code.config.create_model",
                return_value=SimpleNamespace(model=model),
            ) as create_model,
            patch("langchain.agents.create_agent", return_value=grader) as create_agent,
        ):
            assert middleware._ensure_grader() is grader

        create_model.assert_called_once_with(
            "openai:gpt-5.5",
            extra_kwargs={CLI_MAX_RETRIES_KEY: 0},
        )
        kwargs = create_agent.call_args.kwargs
        assert kwargs["model"] is model
        retries = kwargs["middleware"]
        assert len(retries) == 1
        assert isinstance(retries[0], CodeModelRetryMiddleware)
        assert retries[0].max_retries == 0
        assert kwargs["context_schema"].__name__ == "CLIContextSchema"

    def test_string_model_honors_public_caller_fallback(self) -> None:
        middleware = ReliableRubricMiddleware(
            model="openai:gpt-5.5",
            model_retry_fallback=0,
        )
        model = MagicMock()
        grader = MagicMock()

        with (
            patch(
                "deepagents_code.config.create_model",
                return_value=SimpleNamespace(model=model),
            ) as create_model,
            patch("langchain.agents.create_agent", return_value=grader) as create_agent,
        ):
            middleware._ensure_grader()

        create_model.assert_called_once_with("openai:gpt-5.5", extra_kwargs=None)
        retry = create_agent.call_args.kwargs["middleware"][0]
        assert retry.max_retries == 0
        assert model._deepagents_model_retries == 0

    def test_concrete_model_is_reused(self) -> None:
        model = MagicMock()
        grader = MagicMock()
        middleware = ReliableRubricMiddleware(model=cast("Any", model))

        with (
            patch("deepagents_code.config.create_model") as create_model,
            patch("langchain.agents.create_agent", return_value=grader) as create_agent,
        ):
            assert middleware._ensure_grader() is grader

        create_model.assert_not_called()
        assert create_agent.call_args.kwargs["model"] is model

    def test_grader_agent_is_cached(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = MagicMock()
        middleware._grader = grader

        with patch("langchain.agents.create_agent") as create_agent:
            assert middleware._ensure_grader() is grader

        create_agent.assert_not_called()

    @pytest.mark.parametrize("value", [True, -1, 1.5])
    def test_invalid_retry_override_is_rejected(self, value: object) -> None:
        error = ValueError if value == -1 else TypeError
        with pytest.raises(error):
            ReliableRubricMiddleware(
                model="fake-model",
                model_retry_override=cast("Any", value),
            )


class TestRubricRetryContext:
    def test_checkpoint_retry_override_reaches_nested_grader(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = MagicMock()
        grader.invoke.return_value = _satisfied_result()
        middleware._grader = grader

        result = middleware._grade(_state(retries=2), 0)

        assert result.result == "satisfied"
        context = grader.invoke.call_args.kwargs["context"]
        assert context.model_params == {CLI_MAX_RETRIES_KEY: 2}

    def test_launch_override_wins_over_checkpoint_value(self) -> None:
        middleware = ReliableRubricMiddleware(
            model="fake-model",
            model_retry_override=0,
        )
        grader = MagicMock()
        grader.invoke.return_value = _satisfied_result()
        middleware._grader = grader

        middleware._grade(_state(retries=3), 0)

        context = grader.invoke.call_args.kwargs["context"]
        assert context.model_params == {CLI_MAX_RETRIES_KEY: 0}

    async def test_async_checkpoint_override_reaches_nested_grader(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = MagicMock()
        grader.ainvoke = AsyncMock(return_value=_satisfied_result())
        middleware._grader = grader

        result = await middleware._agrade(_state(retries=1), 0)

        assert result.result == "satisfied"
        context = grader.ainvoke.call_args.kwargs["context"]
        assert context.model_params == {CLI_MAX_RETRIES_KEY: 1}

    def test_retryable_error_is_not_replayed_above_model_budget(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = MagicMock()
        grader.invoke.side_effect = _read_error()
        middleware._grader = grader

        with pytest.raises(httpx.ReadError):
            middleware._grade(_state(), 0)

        grader.invoke.assert_called_once()

    def test_wrapped_retryable_error_is_not_replayed(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        wrapper = RuntimeError("grader graph failed")
        wrapper.__cause__ = _read_error()
        grader = MagicMock()
        grader.invoke.side_effect = wrapper
        middleware._grader = grader

        with pytest.raises(RuntimeError, match="grader graph failed"):
            middleware._grade(_state(), 0)

        grader.invoke.assert_called_once()
