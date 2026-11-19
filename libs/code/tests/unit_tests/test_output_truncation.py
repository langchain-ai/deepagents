"""Truncated model replies are classified and surfaced, never silently dropped."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.exceptions import ModelRefusalError
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.pregel.remote import RemoteException, RemoteGraph
from langgraph_api.serde import json_dumpb, json_loads
from langgraph_sdk.client import LangGraphClient

from deepagents_code._glm_5p2_profile import _GlmTerminalStallRecovery
from deepagents_code.agent import _format_task_error
from deepagents_code.app import DeepAgentsApp
from deepagents_code.client.remote_client import RemoteAgent, format_agent_exception
from deepagents_code.model_retry import CodeModelRetryMiddleware
from deepagents_code.output_truncation import (
    EmptyTruncatedReplyMiddleware,
    EmptyTruncatedReplyRecovery,
    ModelOutputTruncatedError,
    OutputTruncation,
    SubagentTruncatedReplyMiddleware,
    output_truncation,
)
from deepagents_code.tui.widgets.messages import ErrorMessage

if TYPE_CHECKING:
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.messages import BaseMessage

_THREAD_ID = "01966f3a-0000-7000-8000-000000000002"

# The shape observed from `openai:gpt-6.1-sol` on the Responses API.
_RESPONSES_INCOMPLETE = {
    "id": "resp_123",
    "status": "incomplete",
    "incomplete_details": {"reason": "max_output_tokens"},
    "model_name": "gpt-6.1-sol",
    "model_provider": "openai",
}


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        pytest.param(
            _RESPONSES_INCOMPLETE,
            OutputTruncation("max_output_tokens", "max_output_tokens"),
            id="openai-responses-max-output-tokens",
        ),
        pytest.param(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "content_filter"},
            },
            OutputTruncation("content_filter", "content_filter"),
            id="openai-responses-content-filter",
        ),
        pytest.param(
            {"status": "completed", "model_provider": "openai"},
            None,
            id="openai-responses-completed",
        ),
        pytest.param(
            {"finish_reason": "length"},
            OutputTruncation("max_output_tokens", "length"),
            id="chat-completions-length",
        ),
        pytest.param(
            {"finish_reason": "content_filter"},
            OutputTruncation("content_filter", "content_filter"),
            id="chat-completions-content-filter",
        ),
        pytest.param({"finish_reason": "stop"}, None, id="chat-completions-stop"),
        pytest.param(
            {"finish_reason": "tool_calls"}, None, id="chat-completions-tool-calls"
        ),
        pytest.param(
            {"stop_reason": "max_tokens"},
            OutputTruncation("max_output_tokens", "max_tokens"),
            id="anthropic-max-tokens",
        ),
        pytest.param(
            {"stop_reason": "refusal"},
            OutputTruncation("content_filter", "refusal"),
            id="anthropic-refusal",
        ),
        pytest.param(
            {"stop_reason": "model_context_window_exceeded"},
            OutputTruncation(
                "context_window_exceeded", "model_context_window_exceeded"
            ),
            id="anthropic-context-window",
        ),
        pytest.param({"stop_reason": "end_turn"}, None, id="anthropic-end-turn"),
        pytest.param(
            {"finish_reason": "MAX_TOKENS"},
            OutputTruncation("max_output_tokens", "MAX_TOKENS"),
            id="google-max-tokens",
        ),
        pytest.param(
            {"finish_reason": "SAFETY"},
            OutputTruncation("content_filter", "SAFETY"),
            id="google-safety",
        ),
        pytest.param({"finish_reason": "STOP"}, None, id="google-stop"),
        pytest.param(
            {"stopReason": "max_tokens"},
            OutputTruncation("max_output_tokens", "max_tokens"),
            id="bedrock-max-tokens",
        ),
        pytest.param(
            {"stopReason": "guardrail_intervened"},
            OutputTruncation("content_filter", "guardrail_intervened"),
            id="bedrock-guardrail",
        ),
        pytest.param({"stopReason": "end_turn"}, None, id="bedrock-end-turn"),
        pytest.param({}, None, id="no-metadata"),
    ],
)
def test_classifies_provider_stop_reasons(
    metadata: dict[str, Any], expected: OutputTruncation | None
) -> None:
    message = AIMessage(content="", response_metadata=metadata)
    assert output_truncation(message) == expected


def _response(
    content: str | list[str | dict[str, Any]] = "",
    *,
    metadata: dict[str, Any] | None = None,
    with_tool_call: bool = False,
) -> ModelResponse[Any]:
    tool_calls = (
        [{"id": "call-1", "name": "ls", "args": {}, "type": "tool_call"}]
        if with_tool_call
        else []
    )
    message = AIMessage(
        content=content,
        tool_calls=tool_calls,
        response_metadata=_RESPONSES_INCOMPLETE if metadata is None else metadata,
    )
    return ModelResponse(result=[message])


def _request() -> ModelRequest:
    return cast("ModelRequest", SimpleNamespace(model="openai:gpt-6.1-sol", state={}))


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "content",
    [
        pytest.param([], id="empty-blocks"),
        pytest.param("  \n", id="blank-text"),
        pytest.param(
            [{"type": "reasoning", "summary": [], "id": "rs_1"}], id="reasoning-only"
        ),
    ],
)
async def test_empty_truncated_reply_raises(
    content: str | list[str | dict[str, Any]], *, asynchronous: bool
) -> None:
    middleware = EmptyTruncatedReplyMiddleware()
    response = _response(content)

    async def call() -> None:
        if asynchronous:
            await middleware.awrap_model_call(
                _request(), AsyncMock(return_value=response)
            )
        else:
            middleware.wrap_model_call(_request(), MagicMock(return_value=response))

    with pytest.raises(ModelOutputTruncatedError) as caught:
        await call()

    assert caught.value.truncation.reason == "max_output_tokens"
    assert caught.value.model == "gpt-6.1-sol"
    assert str(caught.value).startswith(
        "Model 'gpt-6.1-sol' stopped before replying: it reached its output token "
        "limit (max_output_tokens)."
    )


@pytest.mark.parametrize(
    ("content", "expected_hint"),
    [
        ("", "Retry, or raise the output token limit."),
        (
            [{"type": "reasoning", "reasoning": "Thinking"}],
            "Retry, or lower the reasoning effort.",
        ),
        (
            [{"type": "reasoning", "summary": [], "id": "rs_1"}],
            "Retry, or lower the reasoning effort.",
        ),
    ],
)
def test_output_limit_hint_matches_reply(
    content: str | list[str | dict[str, Any]], expected_hint: str
) -> None:
    with pytest.raises(ModelOutputTruncatedError) as caught:
        EmptyTruncatedReplyMiddleware().wrap_model_call(
            _request(), lambda _: _response(content)
        )

    assert str(caught.value).endswith(expected_hint)
    assert str(copy.deepcopy(caught.value)) == str(caught.value)


def test_context_window_hint_suggests_compact() -> None:
    with pytest.raises(ModelOutputTruncatedError, match="Run /compact"):
        EmptyTruncatedReplyMiddleware().wrap_model_call(
            _request(),
            lambda _: _response(
                metadata={"stop_reason": "model_context_window_exceeded"}
            ),
        )


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_invalid_truncated_tool_call_reports_interrupted_call(
    *, asynchronous: bool
) -> None:
    message = AIMessage(
        content="",
        invalid_tool_calls=[
            {"name": "ls", "args": '{"path":', "id": "call-1", "error": "Invalid JSON"}
        ],
        response_metadata={"finish_reason": "length"},
    )
    response = ModelResponse(result=[message])
    middleware = EmptyTruncatedReplyMiddleware()

    async def call() -> None:
        if asynchronous:
            await middleware.awrap_model_call(
                _request(), AsyncMock(return_value=response)
            )
        else:
            middleware.wrap_model_call(_request(), lambda _: response)

    with pytest.raises(ModelOutputTruncatedError) as caught:
        await call()

    text = str(caught.value)
    assert (
        "stopped while writing a tool call: it reached its output token limit (length)"
        in text
    )
    assert "before replying" not in text
    assert str(copy.deepcopy(caught.value)) == text


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(_response("Partial answer"), id="partial-text"),
        pytest.param(_response(with_tool_call=True), id="partial-tool-call"),
        pytest.param(_response(metadata={"finish_reason": "stop"}), id="normal-empty"),
        pytest.param(
            _response("Done.", metadata={"finish_reason": "stop"}), id="normal-reply"
        ),
    ],
)
def test_reply_with_output_or_normal_stop_passes_through(
    response: ModelResponse[Any],
) -> None:
    middleware = EmptyTruncatedReplyMiddleware()
    assert middleware.wrap_model_call(_request(), lambda _: response) is response


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.filterwarnings("ignore:The feature `forked subagents` is in beta")
@pytest.mark.parametrize("mode", ["fork", "isolated"])
@pytest.mark.parametrize("content", ["", "Partial findings"])
async def test_subagent_truncation_reaches_parent_without_aborting(
    mode: Literal["fork", "isolated"], content: str, *, asynchronous: bool
) -> None:
    from deepagents import create_deep_agent
    from langchain_core.messages import ToolMessage

    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "task-1",
                            "name": "task",
                            "args": {
                                "subagent_type": "general-purpose",
                                "description": "Do the task",
                            },
                        }
                    ],
                ),
                AIMessage(content=content, response_metadata=_RESPONSES_INCOMPLETE),
                AIMessage(content="Parent finished"),
            ]
        )
    )
    with patch.object(GenericFakeChatModel, "bind_tools", return_value=model):
        graph = create_deep_agent(
            model,
            middleware=[
                EmptyTruncatedReplyMiddleware(),
                ToolErrorMiddleware(_format_task_error, tools=["task"]),
            ],
            subagents=[
                {
                    "name": "general-purpose",
                    "description": "General-purpose subagent",
                    "system_prompt": "Complete the task",
                    "mode": mode,
                    "middleware": [
                        SubagentTruncatedReplyMiddleware(),
                        CodeModelRetryMiddleware(max_retries=5),
                    ],
                }
            ],
        )
        inputs = {"messages": [HumanMessage(content="Delegate the task")]}
        result = await graph.ainvoke(inputs) if asynchronous else graph.invoke(inputs)

    task_result = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    if content:
        assert task_result.content == (
            "Partial findings\n\nSubagent reply was cut off: "
            "it reached its output token limit (max_output_tokens)."
        )
        assert task_result.status == "success"
    else:
        assert "stopped before replying" in task_result.text
        assert "output token limit" in task_result.text
        assert "Retry, or raise the output token limit." in task_result.text
        assert task_result.text.startswith("Subagent 'general-purpose' failed:")
        assert task_result.status == "error"
    assert result["messages"][-1].text == "Parent finished"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("recovered", [False, True])
@pytest.mark.filterwarnings("ignore:The feature `forked subagents` is in beta")
async def test_forked_subagent_gets_glm_recovery_before_truncation_guard(
    *, asynchronous: bool, recovered: bool
) -> None:
    from deepagents import create_deep_agent
    from langchain_core.messages import ToolMessage

    stalled = AIMessage(content="", response_metadata={"finish_reason": "length"})
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "task-1",
                            "name": "task",
                            "args": {
                                "subagent_type": "general-purpose",
                                "description": "Do the task",
                            },
                        }
                    ],
                ),
                stalled,
                AIMessage(content="Recovered findings") if recovered else stalled,
                AIMessage(content="Parent finished"),
            ]
        )
    )
    with (
        patch.object(GenericFakeChatModel, "bind_tools", return_value=model),
        patch(
            "deepagents_code._glm_5p2_profile._is_fireworks_glm_5p2_model",
            return_value=True,
        ),
    ):
        graph = create_deep_agent(
            model,
            middleware=[
                EmptyTruncatedReplyMiddleware(),
                _GlmTerminalStallRecovery(),
                CodeModelRetryMiddleware(max_retries=0),
            ],
            subagents=[
                {
                    "name": "general-purpose",
                    "description": "General-purpose subagent",
                    "system_prompt": "Complete the task",
                    "mode": "fork",
                    "middleware": [
                        SubagentTruncatedReplyMiddleware(),
                        _GlmTerminalStallRecovery(),
                        CodeModelRetryMiddleware(max_retries=0),
                    ],
                }
            ],
        )
        inputs = {"messages": [HumanMessage(content="Delegate the task")]}
        result = await graph.ainvoke(inputs) if asynchronous else graph.invoke(inputs)

    task_result = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    if recovered:
        assert task_result.content == "Recovered findings"
        assert task_result.status == "success"
    else:
        assert "stopped before replying" in task_result.text
        assert task_result.status == "error"
    assert result["messages"][-1].text == "Parent finished"


def test_zero_usage_does_not_blame_output_limit() -> None:
    """A zero-usage reply did not run out of output budget."""
    message = AIMessage(
        content=[],
        response_metadata=_RESPONSES_INCOMPLETE,
        usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    )
    middleware = EmptyTruncatedReplyMiddleware()

    with pytest.raises(ModelOutputTruncatedError) as caught:
        middleware.wrap_model_call(
            _request(),
            lambda _: ModelResponse(result=[message]),
        )

    text = str(caught.value)
    assert "likely failed upstream rather than hitting the limit" in text
    assert "reasoning effort" not in text
    assert copy.deepcopy(caught.value).no_usage is True


def test_error_message_survives_server_serialization() -> None:
    error = ModelOutputTruncatedError(
        OutputTruncation("content_filter", "SAFETY"), "gemini-3-pro"
    )
    payload = json_loads(json_dumpb(error))
    assert payload == {"error": "ModelOutputTruncatedError", "message": str(error)}
    assert "the provider blocked the reply (SAFETY)" in payload["message"]
    clone = copy.deepcopy(error)
    assert (clone.truncation, clone.model, str(clone)) == (
        error.truncation,
        error.model,
        str(error),
    )


class _TruncatingModel(BaseChatModel):
    attempts: int = 0

    @property
    def _llm_type(self) -> str:
        return "fake-truncating"

    def _generate(
        self,
        messages: list[BaseMessage],  # noqa: ARG002
        stop: list[str] | None = None,  # noqa: ARG002
        run_manager: CallbackManagerForLLMRun | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> ChatResult:
        self.attempts += 1
        message = AIMessage(content=[], response_metadata=_RESPONSES_INCOMPLETE)
        return ChatResult(generations=[ChatGeneration(message=message)])


async def test_empty_truncated_reply_reaches_tui_as_error() -> None:
    """The run fails, is not retried, saves nothing, and the TUI shows why."""
    model = _TruncatingModel()
    checkpointer = InMemorySaver()
    # Same relative order as `create_cli_agent`: the guard wraps the retry.
    graph = create_agent(
        model,
        middleware=[
            EmptyTruncatedReplyMiddleware(),
            CodeModelRetryMiddleware(max_retries=5),
        ],
        checkpointer=checkpointer,
    )
    config: Any = {"configurable": {"thread_id": _THREAD_ID}}
    frames: list[bytes] = []

    async def respond(_request: httpx.Request) -> httpx.Response:
        try:
            async for namespace, mode, data in graph.astream(
                {"messages": [HumanMessage(content="hi")]},
                config,
                stream_mode=["messages", "updates", "custom"],
                subgraphs=True,
            ):
                event = "|".join((mode, *namespace))
                frames.append(
                    b"event: "
                    + event.encode()
                    + b"\ndata: "
                    + json_dumpb(data)
                    + b"\n\n"
                )
        except ModelOutputTruncatedError as exc:
            frames.append(b"event: error\ndata: " + json_dumpb(exc) + b"\n\n")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b"".join(frames),
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), base_url="http://test"
    ) as client:
        remote = RemoteAgent("http://test", local_ownership=False)
        remote._graph = RemoteGraph("agent", client=LangGraphClient(client))
        remote._workspaces[_THREAD_ID] = {"workspace_id": "test-workspace"}
        with pytest.raises(RemoteException) as caught:
            async for _ in remote.astream(
                {"messages": []},
                config=config,
                stream_mode=["messages", "updates", "custom"],
                subgraphs=True,
            ):
                pass

    assert model.attempts == 1
    state = await graph.aget_state(config)
    assert [type(m).__name__ for m in state.values["messages"]] == ["HumanMessage"]

    expected = (
        "ModelOutputTruncatedError: Model 'gpt-6.1-sol' stopped before replying: "
        "it reached its output token limit (max_output_tokens)."
    )
    assert format_agent_exception(caught.value).startswith(expected)
    app = DeepAgentsApp(agent=MagicMock())
    async with app.run_test() as pilot:
        await pilot.pause()
        app._ui_adapter = MagicMock()
        with patch(
            "deepagents_code.tui.textual_adapter.execute_task_textual",
            new_callable=AsyncMock,
            side_effect=caught.value,
        ):
            await app._run_agent_task("hi")
            await pilot.pause()
        errors = app.query(ErrorMessage)
        assert any(expected in str(widget._content) for widget in errors)


def _glm_request() -> ModelRequest:
    model = MagicMock(spec=BaseChatModel)
    model.model_name = "accounts/fireworks/models/glm-5p2"
    model._get_ls_params.return_value = {"ls_provider": "fireworks"}
    return ModelRequest(
        model=cast("BaseChatModel", model),
        messages=[HumanMessage(content="run")],
        tools=[],
        system_prompt="base prompt",
        state={"messages": []},
        runtime=cast("Any", SimpleNamespace(context={"model": None})),
    )


def _glm_stack(
    responses: list[ModelResponse[Any]],
) -> tuple[ModelResponse[Any], int]:
    """Run the guard outside GLM recovery, as `create_cli_agent` orders them."""
    guard = EmptyTruncatedReplyMiddleware()
    recovery = _GlmTerminalStallRecovery()
    remaining = list(responses)

    def model_call(_request: ModelRequest) -> ModelResponse[Any]:
        return remaining.pop(0)

    response = guard.wrap_model_call(
        _glm_request(), lambda request: recovery.wrap_model_call(request, model_call)
    )
    return response, len(responses) - len(remaining)


def test_glm_recovery_runs_before_the_guard() -> None:
    stalled = _response(metadata={"finish_reason": "length"})
    recovered = _response(metadata={"finish_reason": "tool_calls"}, with_tool_call=True)

    response, calls = _glm_stack([stalled, recovered])

    assert calls == 2
    assert response is recovered


def test_guard_raises_when_glm_recovery_still_stalls() -> None:
    stalled = _response(metadata={"finish_reason": "length"})

    with pytest.raises(ModelOutputTruncatedError, match=r"\(length\)"):
        _glm_stack([stalled, stalled])


_ANTHROPIC_REFUSAL = {"model_provider": "anthropic", "stop_reason": "refusal"}


async def _guard_refusal(
    error: ModelRefusalError, *, asynchronous: bool
) -> ModelResponse[Any]:
    middleware = EmptyTruncatedReplyMiddleware()
    if asynchronous:
        return await middleware.awrap_model_call(
            _request(), AsyncMock(side_effect=error)
        )
    return middleware.wrap_model_call(_request(), MagicMock(side_effect=error))


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_raised_refusal_with_empty_reply_is_reported_as_cut_off(
    *, asynchronous: bool
) -> None:
    refused = AIMessage(content=[], response_metadata=_ANTHROPIC_REFUSAL)
    error = ModelRefusalError(raw_message=refused)

    with pytest.raises(ModelOutputTruncatedError) as caught:
        await _guard_refusal(error, asynchronous=asynchronous)

    assert caught.value.truncation.reason == "content_filter"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_raised_refusal_with_partial_reply_passes_through(
    *, asynchronous: bool
) -> None:
    refused = AIMessage(
        content="Partial findings", response_metadata=_ANTHROPIC_REFUSAL
    )
    error = ModelRefusalError(raw_message=refused)

    response = await _guard_refusal(error, asynchronous=asynchronous)

    assert response.result == [refused]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_raised_refusal_without_message_propagates(*, asynchronous: bool) -> None:
    error = ModelRefusalError()

    with pytest.raises(ModelRefusalError):
        await _guard_refusal(error, asynchronous=asynchronous)


# The shape reported in #7761: Anthropic reasoning only, stopped at max_tokens.
_ANTHROPIC_MAX_TOKENS = {
    "model_provider": "anthropic",
    "model_name": "claude-haiku-5-5",
    "stop_reason": "max_tokens",
}


def _headless_request(
    model: str = "anthropic:claude-haiku-5-5",
) -> ModelRequest:
    return ModelRequest(
        model=cast("BaseChatModel", model),
        messages=[HumanMessage(content="run")],
        tools=[],
        system_prompt="base prompt",
        state={"messages": []},
        runtime=cast("Any", SimpleNamespace(context={"model": None})),
    )


async def _headless_stack(
    responses: list[ModelResponse[Any]],
    *,
    request: ModelRequest | None = None,
    asynchronous: bool = False,
) -> tuple[ModelResponse[Any], list[ModelRequest]]:
    """Run the guard outside the recovery, as headless `create_cli_agent` does.

    Returns:
        The final response and every request the model received.
    """
    guard = EmptyTruncatedReplyMiddleware()
    recovery = EmptyTruncatedReplyRecovery()
    remaining = list(responses)
    seen: list[ModelRequest] = []

    def model_call(request: ModelRequest) -> ModelResponse[Any]:
        seen.append(request)
        return remaining.pop(0)

    request = _headless_request() if request is None else request
    if asynchronous:
        amodel_call = AsyncMock(side_effect=model_call)

        async def ahandler(inner: ModelRequest) -> ModelResponse[Any]:
            return await recovery.awrap_model_call(inner, amodel_call)

        return await guard.awrap_model_call(request, ahandler), seen
    response = guard.wrap_model_call(
        request, lambda inner: recovery.wrap_model_call(inner, model_call)
    )
    return response, seen


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "stalled",
    [
        pytest.param(
            _response(
                [{"type": "thinking", "thinking": "Planning...", "signature": "s"}],
                metadata=_ANTHROPIC_MAX_TOKENS,
            ),
            id="anthropic-reasoning-only",
        ),
        pytest.param(_response([]), id="openai-responses-empty"),
        pytest.param(_response(metadata={"finish_reason": "length"}), id="length"),
    ],
)
async def test_headless_recovery_retries_empty_truncated_reply_once(
    stalled: ModelResponse[Any], *, asynchronous: bool
) -> None:
    recovered = _response(metadata={"stop_reason": "tool_use"}, with_tool_call=True)

    response, seen = await _headless_stack(
        [stalled, recovered], asynchronous=asynchronous
    )

    assert response is recovered
    assert len(seen) == 2
    assert seen[0].system_prompt == "base prompt"
    retry_prompt = seen[1].system_prompt or ""
    assert retry_prompt.startswith("base prompt\n\n<output_limit_recovery>")
    assert seen[1].messages == seen[0].messages


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_guard_raises_when_headless_recovery_still_empty(
    *, asynchronous: bool
) -> None:
    stalled = _response([], metadata=_ANTHROPIC_MAX_TOKENS)

    with pytest.raises(ModelOutputTruncatedError, match=r"\(max_tokens\)"):
        await _headless_stack([stalled, stalled], asynchronous=asynchronous)


@pytest.mark.parametrize(
    "response",
    [
        pytest.param(_response("Partial answer"), id="partial-text"),
        pytest.param(_response(with_tool_call=True), id="partial-tool-call"),
        pytest.param(_response(metadata={"finish_reason": "stop"}), id="normal-empty"),
        pytest.param(
            _response(
                metadata={"model_provider": "anthropic", "stop_reason": "refusal"}
            ),
            id="refusal",
        ),
        pytest.param(_response(metadata={"finish_reason": "SAFETY"}), id="filtered"),
        pytest.param(
            _response(
                metadata={
                    "model_provider": "anthropic",
                    "stop_reason": "model_context_window_exceeded",
                }
            ),
            id="context-window",
        ),
        pytest.param(
            ModelResponse(
                result=[
                    AIMessage(
                        content="",
                        invalid_tool_calls=[
                            {
                                "name": "ls",
                                "args": '{"path":',
                                "id": "call-1",
                                "error": "Invalid JSON",
                            }
                        ],
                        response_metadata={"finish_reason": "length"},
                    )
                ]
            ),
            id="interrupted-tool-call",
        ),
        pytest.param(
            ModelResponse(
                result=[
                    AIMessage(
                        content="",
                        response_metadata={"finish_reason": "length"},
                        usage_metadata={
                            "input_tokens": 0,
                            "output_tokens": 0,
                            "total_tokens": 0,
                        },
                    )
                ]
            ),
            id="zero-usage",
        ),
    ],
)
def test_headless_recovery_leaves_other_replies_alone(
    response: ModelResponse[Any],
) -> None:
    recovery = EmptyTruncatedReplyRecovery()
    calls = MagicMock(return_value=response)

    assert recovery.wrap_model_call(_headless_request(), calls) is response
    calls.assert_called_once()


def test_headless_recovery_defers_to_glm_recovery() -> None:
    """The measured GLM stall keeps its own recovery: one retry, not two."""
    guard = EmptyTruncatedReplyMiddleware()
    recovery = EmptyTruncatedReplyRecovery()
    glm = _GlmTerminalStallRecovery()
    stalled = _response(metadata={"finish_reason": "length"})
    remaining = [stalled, stalled, stalled]

    def model_call(_request: ModelRequest) -> ModelResponse[Any]:
        return remaining.pop(0)

    with pytest.raises(ModelOutputTruncatedError):
        guard.wrap_model_call(
            _glm_request(),
            lambda request: recovery.wrap_model_call(
                request, lambda inner: glm.wrap_model_call(inner, model_call)
            ),
        )

    assert len(remaining) == 1
