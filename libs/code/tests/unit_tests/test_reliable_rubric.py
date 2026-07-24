"""Tests for reliable rubric grader retries and nested construction."""

from collections.abc import Callable, Iterator, Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from deepagents.graph import create_deep_agent
from deepagents.middleware.rubric import GraderResponse, RubricState
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import ApproveDecision
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from pydantic import Field

from deepagents_code._constants import SDK_DEFAULT_RUBRIC_MAX_ITERATIONS
from deepagents_code.config import CLI_MAX_RETRIES_KEY
from deepagents_code.model_retry import CodeModelRetryMiddleware
from deepagents_code.reliable_rubric import (
    ReliableRubricMiddleware,
    RubricGraderState,
    _is_transient_grader_transport_error,
    _without_internal_control_messages,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


class _FixedGenericFakeChatModel(GenericFakeChatModel):
    """Fake chat model whose structured-output tool binding returns itself."""

    messages: Iterator[AIMessage | str] = Field(exclude=True)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],  # noqa: ARG002
        *,
        tool_choice: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Return this deterministic model after tool binding."""
        return self


def _grader_call(
    *,
    result: str,
    explanation: str,
    criteria: list[dict[str, Any]] | None = None,
) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "GraderResponse",
                "args": {
                    "result": result,
                    "explanation": explanation,
                    "criteria": criteria or [],
                },
                "id": "grader-call",
                "type": "tool_call",
            }
        ],
    )


def _read_error() -> httpx.ReadError:
    return httpx.ReadError(
        "connection closed while reading",
        request=httpx.Request("POST", "https://grader.test"),
    )


def _typed_error(module: str, name: str, message: str = "boom") -> Exception:
    """Build an exception whose type mimics an external library's error class."""
    error_type = type(name, (Exception,), {"__module__": module})
    return error_type(message)


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


def _tool_satisfied_result() -> dict[str, Any]:
    return {
        **_satisfied_result(),
        "messages": [
            _grader_call(
                result="satisfied",
                explanation="all checks pass",
            )
        ],
    }


class TestTransientGraderTransportClassification:
    def test_read_error_is_transient(self) -> None:
        assert _is_transient_grader_transport_error(_read_error()) is True

    def test_remote_protocol_error_is_transient(self) -> None:
        assert (
            _is_transient_grader_transport_error(httpx.RemoteProtocolError("boom"))
            is True
        )

    def test_httpcore_read_error_is_transient(self) -> None:
        # httpcore errors cannot be caught via a stable isinstance, so the
        # classifier matches them by module/name; exercise that path directly.
        error = _typed_error("httpcore", "ReadError")

        assert _is_transient_grader_transport_error(error) is True

    def test_httpcore_remote_protocol_error_is_transient(self) -> None:
        error = _typed_error("httpcore._exceptions", "RemoteProtocolError")

        assert _is_transient_grader_transport_error(error) is True

    def test_read_error_in_exception_group_is_transient(self) -> None:
        group = ExceptionGroup(
            "grading failed", [ValueError("unrelated"), _read_error()]
        )

        assert _is_transient_grader_transport_error(group) is True

    def test_read_error_in_context_chain_is_transient(self) -> None:
        wrapper = RuntimeError("grader request failed")
        wrapper.__context__ = _read_error()

        assert _is_transient_grader_transport_error(wrapper) is True

    def test_transfer_encoding_error_in_cause_chain_is_transient(self) -> None:
        error_type = type(
            "TransferEncodingError",
            (Exception,),
            {"__module__": "aiohttp.http_exceptions"},
        )
        cause = error_type("Not enough data to satisfy transfer length header")
        wrapper = RuntimeError("grader request failed")
        wrapper.__cause__ = cause

        assert _is_transient_grader_transport_error(wrapper) is True

    def test_unrelated_exception_is_not_transient(self) -> None:
        assert _is_transient_grader_transport_error(RuntimeError("bug")) is False


class TestReliableRubricMiddleware:
    def test_displayed_max_iterations_default_matches_sdk(self) -> None:
        """Drift guard for the TUI-display duplicate of the SDK default.

        The constant must equal the `RubricMiddleware` default that the app
        actually instantiates.
        """
        middleware = ReliableRubricMiddleware(model="fake-model")

        assert middleware.max_iterations == SDK_DEFAULT_RUBRIC_MAX_ITERATIONS

    def test_filters_goal_controls_before_sdk_grading(self) -> None:
        visible = HumanMessage(content="user request")
        state_notice = HumanMessage(
            content="goal state",
            additional_kwargs={"lc_source": "goal_state"},
        )
        continuation = HumanMessage(
            content="goal continuation",
            additional_kwargs={"lc_source": "goal_control"},
        )
        summary = HumanMessage(
            content="conversation summary",
            additional_kwargs={"lc_source": "summarization"},
        )
        state = cast(
            "RubricState",
            {
                "rubric": "tests pass",
                "messages": [visible, state_notice, continuation, summary],
            },
        )

        filtered = _without_internal_control_messages(state)

        assert filtered["messages"] == [visible, summary]
        assert state["messages"] == [visible, state_notice, continuation, summary]

    async def test_retries_only_grading_without_mutating_agent_transcript(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        error = httpx.ReadError(
            "connection closed while reading",
            request=httpx.Request("POST", "https://grader.test"),
        )
        grader = AsyncMock()
        grader.ainvoke.side_effect = [error, _satisfied_result()]
        middleware._grader = grader
        state = _state()
        state["_current_grading_run_id"] = "run-123"
        messages_before = list(state["messages"])

        context = {"approval_mode": "manual"}
        result = await middleware._agrade(state, 2, context=context)

        assert result.result == "satisfied"
        assert grader.ainvoke.await_count == 2
        assert all(
            call.kwargs["context"] is context for call in grader.ainvoke.await_args_list
        )
        operation_ids = {
            call.args[0]["rubric_grading_operation_id"]
            for call in grader.ainvoke.await_args_list
        }
        assert operation_ids == {"run-123:2"}
        assert state["messages"] == messages_before

    async def test_does_not_retry_unrelated_exception(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = AsyncMock()
        grader.ainvoke.side_effect = RuntimeError("programming error")
        middleware._grader = grader

        with pytest.raises(RuntimeError, match="programming error"):
            await middleware._agrade(_state(), 0)

        grader.ainvoke.assert_awaited_once()

    def test_sync_grade_retries_transient_transport_failure(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = MagicMock()
        grader.invoke.side_effect = [_read_error(), _satisfied_result()]
        middleware._grader = grader

        result = middleware._grade(_state(), 0)

        assert result.result == "satisfied"
        assert grader.invoke.call_count == 2

    def test_sync_grade_preserves_trace_metadata_and_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        middleware = ReliableRubricMiddleware(model="anthropic:claude-sonnet-4-6")
        grader = MagicMock()
        grader.invoke.return_value = _tool_satisfied_result()
        middleware._grader = grader
        monkeypatch.setattr(
            middleware,
            "_resolved_model",
            SimpleNamespace(
                model_name="claude-sonnet-4-6",
                profile={"structured_output": True},
            ),
        )
        recorded: list[dict[str, str]] = []
        monkeypatch.setattr(
            middleware,
            "_record_grader_trace_metadata",
            recorded.append,
        )
        monkeypatch.setattr(
            "deepagents.middleware.rubric.ensure_config",
            lambda: {"metadata": {"tenant_id": "tenant-123"}},
        )
        context = {"approval_mode": "manual"}

        result = middleware._grade_once(_state(), 0, context=context)

        assert result.result == "satisfied"
        assert grader.invoke.call_args.kwargs == {
            "config": {
                "metadata": {
                    "tenant_id": "tenant-123",
                    "rubric_grader_configured_model": ("anthropic:claude-sonnet-4-6"),
                    "rubric_grader_effective_strategy": "ProviderStrategy",
                }
            },
            "context": context,
        }
        assert recorded[0]["rubric_grader_effective_strategy"] == "ProviderStrategy"
        assert recorded[-1]["rubric_grader_effective_strategy"] == "ToolStrategy"

    async def test_async_grade_preserves_trace_metadata_and_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        middleware = ReliableRubricMiddleware(model="anthropic:claude-sonnet-4-6")
        grader = AsyncMock()
        grader.ainvoke.return_value = _tool_satisfied_result()
        middleware._grader = grader
        monkeypatch.setattr(
            middleware,
            "_resolved_model",
            SimpleNamespace(
                model_name="claude-sonnet-4-6",
                profile={"structured_output": True},
            ),
        )
        recorded: list[dict[str, str]] = []
        monkeypatch.setattr(
            middleware,
            "_record_grader_trace_metadata",
            recorded.append,
        )
        monkeypatch.setattr(
            "deepagents.middleware.rubric.ensure_config",
            lambda: {"metadata": {"experiment_id": "experiment-123"}},
        )
        context = {"approval_mode": "manual"}

        result = await middleware._agrade_once(_state(), 0, context=context)

        assert result.result == "satisfied"
        assert grader.ainvoke.await_args.kwargs == {
            "config": {
                "metadata": {
                    "experiment_id": "experiment-123",
                    "rubric_grader_configured_model": ("anthropic:claude-sonnet-4-6"),
                    "rubric_grader_effective_strategy": "ProviderStrategy",
                }
            },
            "context": context,
        }
        assert recorded[0]["rubric_grader_effective_strategy"] == "ProviderStrategy"
        assert recorded[-1]["rubric_grader_effective_strategy"] == "ToolStrategy"

    async def test_second_transient_failure_propagates_async(self) -> None:
        # The retry is bounded to one attempt: a second transient failure must
        # surface so the base middleware can report it as a grader_error.
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = AsyncMock()
        grader.ainvoke.side_effect = [_read_error(), _read_error()]
        middleware._grader = grader

        with pytest.raises(httpx.ReadError):
            await middleware._agrade(_state(), 0)

        assert grader.ainvoke.await_count == 2

    def test_second_transient_failure_propagates_sync(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = MagicMock()
        grader.invoke.side_effect = [_read_error(), _read_error()]
        middleware._grader = grader

        with pytest.raises(httpx.ReadError):
            middleware._grade(_state(), 0)

        assert grader.invoke.call_count == 2

    def test_builds_context_aware_nested_grader(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        seen: dict[str, Any] = {}
        grader = SimpleNamespace()

        def fake_create_agent(**kwargs: Any) -> SimpleNamespace:
            seen.update(kwargs)
            return grader

        class GraderContext:
            pass

        resolved_model = SimpleNamespace(
            model_name="claude-sonnet-4-6",
            profile={"structured_output": True},
        )
        nested_middleware = AgentMiddleware()
        monkeypatch.setattr("langchain.agents.create_agent", fake_create_agent)
        middleware = ReliableRubricMiddleware(
            model="fake-model",
            grader_middleware=[nested_middleware],
            grader_context_schema=GraderContext,
            model_retry_fallback=0,
        )

        with patch(
            "deepagents_code.config.create_model",
            return_value=SimpleNamespace(model=resolved_model),
        ) as create_model:
            assert middleware._ensure_grader() is grader

        create_model.assert_called_once()
        assert isinstance(seen["middleware"][0], CodeModelRetryMiddleware)
        assert seen["middleware"][0].max_retries == 0
        assert seen["middleware"][1:] == [nested_middleware]
        assert seen["context_schema"] is GraderContext
        assert seen["state_schema"] is RubricGraderState
        assert middleware._resolved_model is resolved_model
        assert (
            middleware._grader_trace_metadata()["rubric_grader_effective_strategy"]
            == "ProviderStrategy"
        )

    async def test_nested_grader_interrupt_propagates_with_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grade = AsyncMock(side_effect=GraphInterrupt(()))
        monkeypatch.setattr(middleware, "_agrade", grade)
        context = {"approval_mode": "manual"}
        runtime = cast(
            "Runtime[Any]",
            SimpleNamespace(stream_writer=lambda _event: None, context=context),
        )

        with pytest.raises(GraphInterrupt):
            await middleware.aafter_agent(_state(), runtime)

        assert grade.await_args is not None
        assert grade.await_args.kwargs["context"] is context

    @pytest.mark.filterwarnings(
        r"ignore:The middleware `RubricMiddleware` is in beta\..*"
    )
    def test_nested_grader_tool_approval_resumes_through_parent_graph(self) -> None:
        observed: list[str] = []

        @tool
        def inspect_external(resource_id: str) -> str:
            """Inspect an external resource without modifying it."""
            observed.append(resource_id)
            return "resource is updated"

        main_model = _FixedGenericFakeChatModel(
            messages=iter([AIMessage(content="external update complete")])
        )
        grader_model = _FixedGenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "inspect_external",
                                "args": {"resource_id": "page-123"},
                                "id": "inspect-call",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    _grader_call(
                        result="satisfied",
                        explanation="external state verified",
                        criteria=[{"name": "resource updated", "passed": True}],
                    ),
                ]
            )
        )
        rubric = ReliableRubricMiddleware(
            model=grader_model,
            tools=[inspect_external],
            grader_middleware=[HumanInTheLoopMiddleware({"inspect_external": True})],
        )
        agent = create_deep_agent(
            model=main_model,
            middleware=[rubric],
            checkpointer=InMemorySaver(),
        )
        config: RunnableConfig = {
            "configurable": {"thread_id": "rubric-grader-tool-hitl"}
        }

        first = agent.invoke(
            {
                "messages": [HumanMessage(content="update the external resource")],
                "rubric": "- resource updated",
            },
            config=config,
        )
        interrupt = first["__interrupt__"][0]
        agent.invoke(
            Command(
                resume={interrupt.id: {"decisions": [ApproveDecision(type="approve")]}}
            ),
            config=config,
        )

        assert observed == ["page-123"]
        state = agent.get_state(config).values
        assert state["_rubric_status"] == "satisfied"
        assert state["_rubric_evaluations"][-1]["criteria"] == [
            {"name": "resource updated", "passed": True}
        ]


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
        assert len(retries) >= 1
        assert isinstance(retries[0], CodeModelRetryMiddleware)
        assert retries[0].max_retries == 0
        assert kwargs["state_schema"] is RubricGraderState

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
        assert context["model_params"] == {CLI_MAX_RETRIES_KEY: 2}

    def test_launch_override_wins_over_checkpoint_value(self) -> None:
        middleware = ReliableRubricMiddleware(
            model="fake-model",
            model_retry_override=0,
        )
        grader = MagicMock()
        grader.invoke.return_value = _satisfied_result()
        middleware._grader = grader

        middleware._grade(_state(retries=3), 0, context={"approval_mode": "manual"})

        context = grader.invoke.call_args.kwargs["context"]
        assert context["approval_mode"] == "manual"
        assert context["model_params"] == {CLI_MAX_RETRIES_KEY: 0}

    def test_runtime_context_retry_override_is_preserved(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = MagicMock()
        grader.invoke.return_value = _satisfied_result()
        middleware._grader = grader
        runtime_context = {
            "approval_mode": "manual",
            "model_params": {CLI_MAX_RETRIES_KEY: 4},
        }

        middleware._grade(_state(), 0, context=runtime_context)

        assert grader.invoke.call_args.kwargs["context"] is runtime_context

    async def test_async_checkpoint_override_reaches_nested_grader(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = MagicMock()
        grader.ainvoke = AsyncMock(return_value=_satisfied_result())
        middleware._grader = grader

        result = await middleware._agrade(_state(retries=1), 0)

        assert result.result == "satisfied"
        context = grader.ainvoke.call_args.kwargs["context"]
        assert context["model_params"] == {CLI_MAX_RETRIES_KEY: 1}
