"""Context recovery must fit requests, preserve tool pairs, and stop making no progress."""

from collections.abc import Callable
from typing import cast

import pytest
from langchain.agents.middleware.types import AgentState, ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.exceptions import ContextOverflowError
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage

from deepagents.middleware.summarization import SummarizationMiddleware
from tests.unit_tests.middleware.test_summarization_middleware import MockBackend, make_mock_model, make_mock_runtime


class ProviderOverflowError(Exception):
    """An OpenAI-compatible server's unclassified bad request."""

    status_code = 400


def _count(messages: list[AnyMessage], *, tools: list | None = None) -> int:
    """Deterministic units, including system text and tool schemas."""
    return sum(len(str(message.content)) for message in messages) + len(str(tools or ""))


def _request(*, tail_size: int = 20000, limit: int = 100000) -> ModelRequest:
    model = make_mock_model("summary")
    model.profile = {"max_input_tokens": limit}
    messages = [
        HumanMessage(content="old context " * 1000, id="old"),
        AIMessage(content="old answer", id="answer"),
        HumanMessage(content="run the tool", id="user"),
        AIMessage(content="", tool_calls=[{"id": "call", "name": "search", "args": {}}], id="ai"),
        ToolMessage(content="x" * tail_size, tool_call_id="call", id="result"),
    ]
    return ModelRequest(model=model, messages=messages, state=cast("AgentState", {"messages": messages}), runtime=make_mock_runtime())


def _middleware(request: ModelRequest, backend: MockBackend, *, trigger: int = 1, keep: int = 3) -> SummarizationMiddleware:
    return SummarizationMiddleware(model=request.model, backend=backend, trigger=("messages", trigger), keep=("messages", keep), token_counter=_count)


async def _invoke(
    middleware: SummarizationMiddleware,
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
    *,
    asynchronous: bool,
) -> ModelResponse | ExtendedModelResponse:
    if asynchronous:

        async def async_handler(current: ModelRequest) -> ModelResponse:
            return handler(current)

        return await middleware.awrap_model_call(request, async_handler)
    return middleware.wrap_model_call(request, handler)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("trigger", [1, 100])
async def test_oversized_tail_fits_before_send(*, asynchronous: bool, trigger: int) -> None:
    request = _request(limit=10000)
    backend = MockBackend()
    sent: list[ModelRequest] = []

    def handler(current: ModelRequest) -> ModelResponse:
        sent.append(current)
        assert _count(current.messages) <= 9500
        return ModelResponse(result=[AIMessage(content="done")])

    result = await _invoke(_middleware(request, backend, trigger=trigger), request, handler, asynchronous=asynchronous)
    assert len(sent) == 1
    assert isinstance(result, ExtendedModelResponse)
    replacements = result.command.update["messages"]
    assert replacements[-1].id == "result"
    assert replacements[-1].tool_call_id == sent[0].messages[-2].tool_calls[0]["id"]
    assert any(content == "x" * 20000 for _, content in backend.write_calls)
    assert request.messages[-1].content == "x" * 20000


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("profile", [None, {}, {"max_input_tokens": True}])
@pytest.mark.parametrize("trigger", [1, 100])
@pytest.mark.parametrize("overflow", [False, True])
async def test_unknown_agent_limit_does_not_use_summarizer_budget(
    *, asynchronous: bool, profile: dict[str, int] | None, trigger: int, overflow: bool
) -> None:
    request = _request().override(system_message=SystemMessage(content="s" * 10000))
    request.model.profile = profile
    summarizer = make_mock_model("summary")
    summarizer.profile = {"max_input_tokens": 10000}
    middleware = SummarizationMiddleware(
        model=summarizer, backend=MockBackend(), trigger=("messages", trigger), keep=("messages", 3), token_counter=_count
    )
    sent: list[int] = []

    def handler(current: ModelRequest) -> ModelResponse:
        assert current.system_message == request.system_message
        sent.append(_count(current.messages))
        if overflow and len(sent) == 1:
            raise ContextOverflowError
        return ModelResponse(result=[AIMessage(content="done")])

    result = await _invoke(middleware, request, handler, asynchronous=asynchronous)
    response = result.model_response if isinstance(result, ExtendedModelResponse) else result
    assert response.result[0].content == "done"
    assert len(sent) == (2 if overflow else 1)
    if overflow:
        assert sent[1] < sent[0]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("trigger", [1, 100])
async def test_provider_overflow_recovers_once(*, asynchronous: bool, trigger: int) -> None:
    request = _request()
    sent: list[int] = []

    def handler(current: ModelRequest) -> ModelResponse:
        sent.append(_count(current.messages))
        if len(sent) == 1:
            msg = "maximum context length exceeded"
            raise ProviderOverflowError(msg)
        return ModelResponse(result=[AIMessage(content="done")])

    result = await _invoke(_middleware(request, MockBackend(), trigger=trigger), request, handler, asynchronous=asynchronous)
    assert isinstance(result, ExtendedModelResponse)
    assert len(sent) == 2
    assert sent[1] < sent[0]
    assert result.command.update["messages"][-1].id == "result"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("trigger", [1, 100])
async def test_repeated_overflow_stops_after_smaller_retry(*, asynchronous: bool, trigger: int) -> None:
    request = _request()
    sent: list[int] = []

    def handler(current: ModelRequest) -> ModelResponse:
        sent.append(_count(current.messages))
        raise ContextOverflowError

    with pytest.raises(ContextOverflowError):
        await _invoke(_middleware(request, MockBackend(), trigger=trigger), request, handler, asynchronous=asynchronous)
    assert len(sent) == 2
    assert sent[1] < sent[0]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", [False, True])
async def test_no_cutoff_never_resends_unchanged_input(*, asynchronous: bool, failure: bool) -> None:
    request = _request(tail_size=20000 if failure else 1)
    backend = MockBackend(should_fail=failure)
    calls = 0

    def handler(_current: ModelRequest) -> ModelResponse:
        nonlocal calls
        calls += 1
        raise ContextOverflowError

    with pytest.raises(ContextOverflowError):
        await _invoke(_middleware(request, backend, trigger=100, keep=100), request, handler, asynchronous=asynchronous)
    assert calls == 1
    assert request.messages[-1].content == "x" * (20000 if failure else 1)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("overhead", ["system", "tools", "output", "model_output"])
async def test_irreducible_overhead_never_reaches_server(*, asynchronous: bool, overhead: str) -> None:
    request = _request(limit=10000)
    if overhead == "system":
        request = request.override(system_message=SystemMessage(content="s" * 10000))
    elif overhead == "tools":
        request = request.override(tools=[{"name": "tool", "description": "s" * 10000}])
    elif overhead == "output":
        request = request.override(model_settings={"max_completion_tokens": 10000})
    else:
        request.model.max_tokens = 10000

    def handler(_current: ModelRequest) -> ModelResponse:
        pytest.fail("A known oversized request must not reach the server")

    with pytest.raises(ContextOverflowError, match="above the input budget"):
        await _invoke(_middleware(request, MockBackend()), request, handler, asynchronous=asynchronous)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_unrelated_bad_request_is_not_retried(*, asynchronous: bool) -> None:
    request = _request()
    backend = MockBackend()
    error = ProviderOverflowError("invalid tool schema")

    def handler(_current: ModelRequest) -> ModelResponse:
        raise error

    with pytest.raises(ProviderOverflowError) as caught:
        await _invoke(_middleware(request, backend, trigger=100), request, handler, asynchronous=asynchronous)
    assert caught.value is error
    assert backend.write_calls == []


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_no_cutoff_recovers_and_persists_tool_result(*, asynchronous: bool) -> None:
    request = _request()
    backend = MockBackend()
    sent: list[ModelRequest] = []

    def handler(current: ModelRequest) -> ModelResponse:
        sent.append(current)
        if len(sent) == 1:
            raise ContextOverflowError
        return ModelResponse(result=[AIMessage(content="done")])

    result = await _invoke(_middleware(request, backend, trigger=100, keep=100), request, handler, asynchronous=asynchronous)
    assert len(sent) == 2
    assert isinstance(result, ExtendedModelResponse)
    assert "_summarization_event" not in result.command.update
    assert result.command.update["messages"][-1].id == "result"
    assert sent[-1].messages[-2].tool_calls == request.messages[-2].tool_calls
    assert _count(sent[-1].messages) < _count(sent[0].messages)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_exhausted_provider_error_is_terminal(*, asynchronous: bool) -> None:
    request = _request()
    error = ProviderOverflowError("maximum context length exceeded")
    calls = 0

    def handler(_current: ModelRequest) -> ModelResponse:
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(ContextOverflowError, match="after recovery") as caught:
        await _invoke(_middleware(request, MockBackend()), request, handler, asynchronous=asynchronous)
    assert calls == 2
    assert caught.value.__cause__ is error
    assert not hasattr(caught.value, "status_code")


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_output_reservation_clips_an_otherwise_fitting_summary(*, asynchronous: bool) -> None:
    request = _request(tail_size=6000, limit=30000).override(model_settings={"max_tokens": 24000})
    sent: list[ModelRequest] = []

    def handler(current: ModelRequest) -> ModelResponse:
        sent.append(current)
        assert _count(current.messages) <= 4500
        return ModelResponse(result=[AIMessage(content="done")])

    await _invoke(_middleware(request, MockBackend()), request, handler, asynchronous=asynchronous)
    assert len(sent) == 1
    assert len(sent[0].messages[-1].content) < 6000
