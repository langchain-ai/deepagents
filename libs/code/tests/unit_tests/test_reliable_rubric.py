"""Tests for CLI-specific rubric grader behavior."""

import json
from collections.abc import Callable, Iterator, Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from deepagents.graph import create_deep_agent
from deepagents.middleware.rubric import GraderResponse, RubricState
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import ApproveDecision
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command
from pydantic import Field

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code._constants import SDK_DEFAULT_RUBRIC_MAX_ITERATIONS
from deepagents_code.reliable_rubric import (
    ReliableRubricMiddleware,
    ReliableRubricState,
    RubricGraderState,
    _without_internal_control_messages,
)
from deepagents_code.resume_state import INHERIT_RUBRIC_MODEL

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


class _RetryingGraderModel(BaseChatModel):
    """Stream a partial verdict, fail once, then return structured output."""

    attempts: ClassVar[int] = 0

    @property
    def _llm_type(self) -> str:
        return "retrying-grader"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],  # noqa: ARG002
        *,
        tool_choice: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Runnable[LanguageModelInput, AIMessage]:
        """Return this deterministic model after structured-output binding."""
        return self

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=_grader_call(
                        result="satisfied",
                        explanation="verified after retry",
                        criteria=[{"name": "tests pass", "passed": True}],
                    )
                )
            ]
        )

    def _stream(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Iterator[ChatGenerationChunk]:
        type(self).attempts += 1
        if type(self).attempts == 1:
            yield ChatGenerationChunk(message=AIMessageChunk(content="partial"))
            msg = "grader connection dropped"
            raise httpx.ReadError(msg)
        args = json.dumps(
            {
                "result": "satisfied",
                "explanation": "verified after retry",
                "criteria": [{"name": "tests pass", "passed": True}],
            }
        )
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": "GraderResponse",
                        "args": args,
                        "id": "grader-call",
                        "index": 0,
                        "type": "tool_call_chunk",
                    }
                ],
                chunk_position="last",
            )
        )


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


def _state() -> RubricState:
    return cast(
        "RubricState",
        {
            "rubric": "tests pass",
            "messages": [
                HumanMessage(content="implement it"),
                AIMessage(content="implementation complete"),
            ],
        },
    )


def _satisfied_result() -> dict[str, Any]:
    """A usable verdict: at least one per-criterion result, so no coverage retry."""
    return {
        "structured_response": GraderResponse(
            result="satisfied",
            explanation="all checks pass",
            criteria=[{"name": "tests pass", "passed": True}],
        )
    }


def _frozen_criteria_state() -> RubricState:
    """State whose frozen criteria list the grader is expected to cover exactly."""
    state = _state()
    state["_rubric_criteria"] = ["compiles", "tests pass"]
    return state


def _under_reported_result() -> dict[str, Any]:
    """A `satisfied` verdict backed by fewer criteria than the rubric froze."""
    return {
        "structured_response": GraderResponse(
            result="satisfied",
            explanation="looks fine",
            criteria=[{"name": "compiles", "passed": True}],
        )
    }


def _fully_reported_result() -> dict[str, Any]:
    return {
        "structured_response": GraderResponse(
            result="satisfied",
            explanation="all checks pass",
            criteria=[
                {"name": "compiles", "passed": True},
                {"name": "tests pass", "passed": True},
            ],
        )
    }


def _grader_payload(call: Any) -> str:  # noqa: ANN401
    """Return the prompt text of the grader input passed to a recorded call."""
    return str(call.args[0]["messages"][0].content)


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

    @pytest.mark.parametrize(
        ("selection", "inherit_main", "expected_model", "expected_params"),
        [
            (None, True, "openai:gpt-5.5", {"temperature": 0.2}),
            (42, True, "openai:gpt-5.5", {"temperature": 0.2}),
            ("  ", True, "openai:gpt-5.5", {"temperature": 0.2}),
            ("anthropic:claude-sonnet-4-6", True, "anthropic:claude-sonnet-4-6", {}),
            (INHERIT_RUBRIC_MODEL, False, "openai:gpt-5.5", {"temperature": 0.2}),
        ],
    )
    def test_selects_request_local_grader_context(
        self,
        selection: object | None,
        inherit_main: bool,
        expected_model: str,
        expected_params: dict[str, Any],
    ) -> None:
        middleware = ReliableRubricMiddleware(
            model="startup:model", inherit_main_model=inherit_main
        )
        state = cast(
            "Any",
            {
                "_model_spec": "openai:gpt-5.5",
                "_model_params": {"temperature": 0.2},
            },
        )
        if selection is not None:
            state["_rubric_model_spec"] = selection
        parent = CLIContextSchema(
            model="openai:gpt-5.5",
            model_params={"max_tokens": 1},
            profile_overrides={"context_window": 1000},
        )

        selected = middleware._grader_context(state, parent)

        assert selected.model == expected_model
        assert selected.model_params == expected_params
        assert selected.profile_overrides == {"context_window": 1000}
        assert parent.model == "openai:gpt-5.5"
        assert parent.model_params == {"max_tokens": 1}

    def test_inherit_clears_stale_params_after_main_model_fallback(self) -> None:
        """A failed runtime switch records its fallback without new params."""
        middleware = ReliableRubricMiddleware(
            model="startup:model", inherit_main_model=True
        )
        state = cast(
            "Any",
            {
                "_model_spec": "fallback:model",
                "_model_params": {"temperature": 0.2},
            },
        )
        parent = CLIContextSchema(
            model="rejected:model",
            model_params={"max_tokens": 1},
        )

        selected = middleware._grader_context(state, parent)

        assert selected.model == "fallback:model"
        assert selected.model_params == {}

    def test_startup_dedicated_model_ignores_main_context(self) -> None:
        middleware = ReliableRubricMiddleware(
            model="anthropic:claude-sonnet-4-6", inherit_main_model=False
        )
        parent = CLIContextSchema(
            model="openai:gpt-5.5",
            model_params={"temperature": 0.2},
        )

        selected = middleware._grader_context(cast("Any", {}), parent)

        assert selected.model is None
        assert selected.model_params == {}
        assert parent.model == "openai:gpt-5.5"

    def test_state_schema_exposes_private_model_channels(self) -> None:
        """The schema override is what makes the state channels readable.

        Without it `_grader_context` reads `None` for every channel and
        silently grades with the construction-time model.
        """
        from typing import get_type_hints

        from langchain.agents.middleware.types import PrivateStateAttr

        assert ReliableRubricMiddleware.state_schema is ReliableRubricState

        hints = get_type_hints(ReliableRubricState, include_extras=True)
        for channel in ("_model_spec", "_model_params", "_rubric_model_spec"):
            assert channel in hints, channel
            metadata = getattr(hints[channel], "__metadata__", ())
            assert PrivateStateAttr in metadata, channel

    def test_grader_context_copies_parent_mutable_containers(self) -> None:
        """Concurrent grader calls must not share containers with the parent."""
        middleware = ReliableRubricMiddleware(
            model="startup:model", inherit_main_model=True
        )
        parent = CLIContextSchema(
            model="openai:gpt-5.5",
            model_params={"temperature": 0.2},
            profile_overrides={"context_window": 1000},
            hooks_server_events=["PreToolUse"],
        )

        selected = middleware._grader_context(cast("Any", {}), parent)

        assert selected.model_params is not parent.model_params
        assert selected.profile_overrides is not parent.profile_overrides
        assert selected.hooks_server_events is not parent.hooks_server_events
        assert selected.profile_overrides == {"context_window": 1000}
        assert selected.hooks_server_events == ["PreToolUse"]

    def test_grader_context_carries_every_field_from_a_dict_payload(self) -> None:
        """RemoteGraph delivers the context as JSON; no field may be dropped."""
        payload = {
            "model": "openai:gpt-5.5",
            "model_params": {"temperature": 0.2},
            "profile_overrides": {"context_window": 1000},
            "model_context_limit": 4096,
            "classifier_model": "openai:gpt-5.1",
            "approval_mode": "yolo",
            "auto_approve": True,
            "approval_mode_key": "key-1",
            "thread_id": "t-1",
            "turn_id": "turn-1",
            "hooks_snapshot_id": "snap-1",
            "hooks_server_events": ["PreToolUse"],
            "prompt_id": "prompt-1",
        }
        middleware = ReliableRubricMiddleware(
            model="startup:model", inherit_main_model=True
        )

        selected = middleware._grader_context(cast("Any", {}), payload)

        # `model`/`model_params` are the grader's to choose; every other field
        # describes the session and must survive the copy verbatim.
        assert selected.model_context_limit == 4096
        assert selected.classifier_model == "openai:gpt-5.1"
        assert selected.approval_mode == "yolo"
        assert selected.auto_approve is True
        assert selected.approval_mode_key == "key-1"
        assert selected.thread_id == "t-1"
        assert selected.turn_id == "turn-1"
        assert selected.hooks_snapshot_id == "snap-1"
        assert selected.hooks_server_events == ["PreToolUse"]
        assert selected.prompt_id == "prompt-1"
        assert selected.profile_overrides == {"context_window": 1000}

    def test_inherit_keeps_parent_model_and_params_together(self) -> None:
        """A thread's first grading pass has no `_model_spec` checkpointed yet."""
        middleware = ReliableRubricMiddleware(
            model="startup:model", inherit_main_model=True
        )
        parent = CLIContextSchema(
            model="openai:gpt-5.5",
            model_params={"temperature": 0.2},
        )

        selected = middleware._grader_context(cast("Any", {}), parent)

        assert selected.model == "openai:gpt-5.5"
        assert selected.model_params == {"temperature": 0.2}

    @pytest.mark.parametrize(
        ("checkpoint_model", "checkpoint_params", "expected_model", "expected_params"),
        [
            (42, {"temperature": 0.2}, "parent:model", {"max_tokens": 1}),
            ("openai:gpt-5.5", "bad-params", "openai:gpt-5.5", {}),
        ],
    )
    def test_inherit_tolerates_malformed_checkpoint_model_metadata(
        self,
        checkpoint_model: object,
        checkpoint_params: object,
        expected_model: str,
        expected_params: dict[str, Any],
    ) -> None:
        """Malformed inherited metadata must not prevent rubric grading."""
        middleware = ReliableRubricMiddleware(
            model="startup:model", inherit_main_model=True
        )
        state = cast(
            "Any",
            {
                "_model_spec": checkpoint_model,
                "_model_params": checkpoint_params,
            },
        )
        parent = CLIContextSchema(
            model="parent:model",
            model_params={"max_tokens": 1},
        )

        selected = middleware._grader_context(state, parent)

        assert selected.model == expected_model
        assert selected.model_params == expected_params

    def test_unrecognized_context_warns_instead_of_degrading_silently(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Defaults drop `approval_mode`, so a wiring bug must not be silent."""
        middleware = ReliableRubricMiddleware(
            model="startup:model", inherit_main_model=True
        )

        with caplog.at_level("WARNING", logger="deepagents_code.reliable_rubric"):
            selected = middleware._grader_context(cast("Any", {}), object())

        assert selected.approval_mode == "manual"
        assert "Unrecognized grader context type" in caplog.text

    async def test_grading_does_not_mutate_agent_transcript(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = AsyncMock()
        grader.ainvoke.return_value = _satisfied_result()
        middleware._grader = grader
        state = _state()
        state["_current_grading_run_id"] = "run-123"
        messages_before = list(state["messages"])

        context = {"approval_mode": "yolo"}
        result = await middleware._agrade(state, 2, context=context)

        assert result.result == "satisfied"
        grader.ainvoke.assert_awaited_once()
        assert all(
            call.kwargs["context"].approval_mode == context["approval_mode"]
            for call in grader.ainvoke.await_args_list
        )
        operation_ids = {
            call.args[0]["rubric_grading_operation_id"]
            for call in grader.ainvoke.await_args_list
        }
        assert operation_ids == {"run-123:2"}
        assert state["messages"] == messages_before

    @pytest.mark.filterwarnings(
        r"ignore:The middleware `RubricMiddleware` is in beta\..*"
    )
    async def test_midstream_failure_retries_and_parses_verdict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hidden partial grader response retries only its failed model node."""
        from deepagents_code.model_retry import CodeModelRetryMiddleware

        monkeypatch.setattr(
            "deepagents_code.model_retry._retry_delay_seconds", lambda *_: 0
        )
        main_model = _FixedGenericFakeChatModel(
            messages=iter([AIMessage(content="implementation complete")])
        )
        _RetryingGraderModel.attempts = 0
        grader_model = _RetryingGraderModel()
        rubric = ReliableRubricMiddleware(
            model=grader_model,
            grader_middleware=[
                CodeModelRetryMiddleware(
                    max_retries=1,
                    stream_output_is_visible=False,
                )
            ],
        )
        agent = create_deep_agent(model=main_model, middleware=[rubric])

        result: dict[str, Any] = {}
        async for namespace, mode, data in agent.astream(
            {
                "messages": [HumanMessage(content="implement it")],
                "rubric": "- tests pass",
            },
            stream_mode=["messages", "values"],
            subgraphs=True,
        ):
            if not namespace and mode == "values" and isinstance(data, dict):
                result = data

        assert _RetryingGraderModel.attempts == 2
        assert result["_rubric_status"] == "satisfied"
        assert result["_rubric_evaluations"][-1]["criteria"] == [
            {"name": "tests pass", "passed": True}
        ]

    async def test_does_not_apply_legacy_retry_async(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = AsyncMock()
        grader.ainvoke.side_effect = TimeoutError("provider timed out")
        middleware._grader = grader

        with pytest.raises(TimeoutError, match="provider timed out"):
            await middleware._agrade(_state(), 0)

        grader.ainvoke.assert_awaited_once()

    def test_does_not_apply_legacy_retry_sync(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = MagicMock()
        grader.invoke.side_effect = TimeoutError("provider timed out")
        middleware._grader = grader

        with pytest.raises(TimeoutError, match="provider timed out"):
            middleware._grade(_state(), 0)

        grader.invoke.assert_called_once()

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
        context = {"approval_mode": "yolo"}
        state = cast("ReliableRubricState", _state())
        state["_rubric_model_spec"] = "openai:gpt-5.5"

        result = middleware._invoke_grader(state, 0, context=context)

        assert result.result == "satisfied"
        assert grader.invoke.call_args.kwargs["config"] == {
            "metadata": {
                "tenant_id": "tenant-123",
                "rubric_grader_configured_model": "openai:gpt-5.5",
                "rubric_grader_effective_strategy": "unknown",
            }
        }
        assert grader.invoke.call_args.kwargs["context"].approval_mode == "yolo"
        assert recorded[0]["rubric_grader_configured_model"] == "openai:gpt-5.5"
        assert recorded[0]["rubric_grader_effective_strategy"] == "unknown"
        assert recorded[-1]["rubric_grader_configured_model"] == "openai:gpt-5.5"
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
        context = {"approval_mode": "yolo"}

        result = await middleware._ainvoke_grader(_state(), 0, context=context)

        assert result.result == "satisfied"
        assert grader.ainvoke.await_args.kwargs["config"] == {
            "metadata": {
                "experiment_id": "experiment-123",
                "rubric_grader_configured_model": ("anthropic:claude-sonnet-4-6"),
                "rubric_grader_effective_strategy": "ProviderStrategy",
            }
        }
        assert grader.ainvoke.await_args.kwargs["context"].approval_mode == "yolo"
        assert recorded[0]["rubric_grader_effective_strategy"] == "ProviderStrategy"
        assert recorded[-1]["rubric_grader_effective_strategy"] == "ToolStrategy"

    async def test_grader_error_reports_runtime_selected_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        middleware = ReliableRubricMiddleware(model="startup:model")
        grader = AsyncMock()
        grader.ainvoke.side_effect = TimeoutError("provider timed out")
        middleware._grader = grader
        recorded: list[dict[str, str]] = []
        monkeypatch.setattr(
            middleware,
            "_record_grader_trace_metadata",
            recorded.append,
        )
        state = cast("ReliableRubricState", _state())
        state["_rubric_model_spec"] = "openai:gpt-5.5"
        runtime = cast(
            "Runtime[Any]",
            SimpleNamespace(stream_writer=lambda _event: None, context={}),
        )

        update = await middleware.aafter_agent(state, runtime)

        assert update is not None
        evaluation = update["_rubric_evaluations"][-1]
        assert evaluation["result"] == "grader_error"
        assert "configured_model='openai:gpt-5.5'" in evaluation["explanation"]
        assert "startup:model" not in evaluation["explanation"]
        assert recorded
        assert all(
            item["rubric_grader_configured_model"] == "openai:gpt-5.5"
            for item in recorded
        )

    def test_inherits_sdk_coverage_retry_sync(self) -> None:
        # The SDK's coverage retry still fires when the grader under-reports its
        # criteria; this is separate from model transport retries.
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = MagicMock()
        grader.invoke.side_effect = [
            _under_reported_result(),
            _fully_reported_result(),
        ]
        middleware._grader = grader

        result = middleware._grade(_frozen_criteria_state(), 0)

        assert result.result == "satisfied"
        assert grader.invoke.call_count == 2
        assert "1 of the 2 criteria" in _grader_payload(grader.invoke.call_args_list[1])

    async def test_inherits_sdk_coverage_retry_async(self) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grader = AsyncMock()
        grader.ainvoke.side_effect = [
            _under_reported_result(),
            _fully_reported_result(),
        ]
        middleware._grader = grader

        result = await middleware._agrade(_frozen_criteria_state(), 0)

        assert result.result == "satisfied"
        assert grader.ainvoke.await_count == 2
        assert "1 of the 2 criteria" in _grader_payload(
            grader.ainvoke.await_args_list[1]
        )

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
        monkeypatch.setattr(
            "deepagents._models.resolve_model",
            lambda _model: resolved_model,
        )
        middleware = ReliableRubricMiddleware(
            model="fake-model",
            grader_middleware=[nested_middleware],
            grader_context_schema=GraderContext,
        )

        assert middleware._ensure_grader() is grader
        assert seen["middleware"] == [nested_middleware]
        assert seen["context_schema"] is GraderContext
        assert seen["state_schema"] is RubricGraderState
        assert middleware._resolved_model is resolved_model
        assert (
            middleware._grader_trace_metadata()["rubric_grader_effective_strategy"]
            == "ProviderStrategy"
        )

    def test_runtime_model_bypasses_an_unavailable_startup_grader(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A thread override must run before resolving its unused startup model."""
        bootstrap = _FixedGenericFakeChatModel(messages=iter([]))
        grader = MagicMock()
        grader.invoke.return_value = _tool_satisfied_result()
        resolved: list[object] = []

        def resolve_model(model: object) -> object:
            resolved.append(model)
            if model == "unavailable:startup":
                msg = "startup provider unavailable"
                raise RuntimeError(msg)
            return model

        monkeypatch.setattr("deepagents._models.resolve_model", resolve_model)
        monkeypatch.setattr("langchain.agents.create_agent", lambda **_kwargs: grader)
        middleware = ReliableRubricMiddleware(
            model="unavailable:startup",
            runtime_bootstrap_model=bootstrap,
        )
        state = cast("ReliableRubricState", _state())
        state["_rubric_model_spec"] = "openai:gpt-5.5"

        result = middleware._invoke_grader(state, 0)

        assert result.result == "satisfied"
        assert resolved == [bootstrap]
        assert middleware._grader is None
        assert grader.invoke.call_args.kwargs["context"].model == "openai:gpt-5.5"
        with pytest.raises(RuntimeError, match="startup provider unavailable"):
            middleware._invoke_grader(_state(), 0)
        assert resolved == [bootstrap, "unavailable:startup"]

    def test_runtime_model_bypasses_with_a_string_bootstrap(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A string main-model spec bootstraps the runtime grader as well.

        `create_cli_agent` leaves the main model as a spec when startup
        cannot resolve it, so the bypass must not require an instance.
        """
        grader = MagicMock()
        grader.invoke.return_value = _tool_satisfied_result()
        resolved: list[object] = []

        def resolve_model(model: object) -> object:
            resolved.append(model)
            if model == "unavailable:startup":
                msg = "startup provider unavailable"
                raise RuntimeError(msg)
            return model

        monkeypatch.setattr("deepagents._models.resolve_model", resolve_model)
        monkeypatch.setattr("langchain.agents.create_agent", lambda **_kwargs: grader)
        middleware = ReliableRubricMiddleware(
            model="unavailable:startup",
            runtime_bootstrap_model="working:model",
        )
        state = cast("ReliableRubricState", _state())
        state["_rubric_model_spec"] = "openai:gpt-5.5"

        result = middleware._invoke_grader(state, 0)

        assert result.result == "satisfied"
        assert resolved == ["working:model"]
        assert middleware._grader is None
        assert grader.invoke.call_args.kwargs["context"].model == "openai:gpt-5.5"

    async def test_nested_grader_interrupt_propagates_with_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        middleware = ReliableRubricMiddleware(model="fake-model")
        grade = AsyncMock(side_effect=GraphInterrupt(()))
        monkeypatch.setattr(middleware, "_agrade", grade)
        context = {"approval_mode": "yolo"}
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
