"""Tests for repeated tool-call guard middleware."""

from collections.abc import Callable

import pytest
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool

from deepagents.middleware.filesystem import FilesystemMiddleware
from deepagents.middleware.repeated_tool_call import RepeatedToolCallMiddleware


def _runtime(tool_call_id: str = "call-current") -> ToolRuntime:
    return ToolRuntime(
        state={},
        context=None,
        tool_call_id=tool_call_id,
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def _read_file_tool() -> BaseTool:
    middleware = FilesystemMiddleware()
    return next(tool for tool in middleware.tools if tool.name == "read_file")


def _ai_tool_call(call_id: str, name: str, args: dict[str, object]) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": args}])


def _tool_result(call_id: str, name: str = "read_file", content: str = "ok") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=call_id, name=name, status="success")


def _request(
    *,
    call_id: str,
    name: str,
    args: dict[str, object],
    messages: list[object],
    tool: object | None = None,
) -> ToolCallRequest:
    return ToolCallRequest(
        runtime=_runtime(call_id),
        tool_call={"id": call_id, "name": name, "args": args},
        state={"messages": messages},
        tool=tool,
    )


def _handler(calls: list[ToolCallRequest]) -> Callable[[ToolCallRequest], ToolMessage]:
    def handle(request: ToolCallRequest) -> ToolMessage:
        calls.append(request)
        return ToolMessage(
            content="handler called",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="success",
        )

    return handle


def test_blocks_repeated_read_file_after_schema_normalization() -> None:
    read_file = _read_file_tool()
    messages = [
        HumanMessage(content="read directionality"),
        _ai_tool_call("call-1", "read_file", {"file_path": "/directionality.md", "offset": 100}),
        _tool_result("call-1"),
        _ai_tool_call("call-2", "read_file", {"file_path": "/directionality.md", "offset": "100", "limit": "100"}),
        _tool_result("call-2"),
        _ai_tool_call("call-3", "read_file", {"file_path": "/directionality.md", "offset": 100}),
    ]
    request = _request(
        call_id="call-3",
        name="read_file",
        args={"file_path": "/directionality.md", "offset": 100},
        messages=messages,
        tool=read_file,
    )
    calls: list[ToolCallRequest] = []

    result = RepeatedToolCallMiddleware(max_repeats=2).wrap_tool_call(request, _handler(calls))

    assert calls == []
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert result.tool_call_id == "call-3"
    assert result.name == "read_file"
    assert result.content == (
        "Repeated tool call blocked: `read_file` was called more than "
        "2 consecutive times with the same normalized arguments. "
        "The repeated call was not executed. Continue the task using the "
        "existing context or another approach."
    )


def test_allows_same_tool_with_different_normalized_args() -> None:
    read_file = _read_file_tool()
    messages = [
        HumanMessage(content="read directionality"),
        _ai_tool_call("call-1", "read_file", {"file_path": "/directionality.md", "offset": 100}),
        _tool_result("call-1"),
        _ai_tool_call("call-2", "read_file", {"file_path": "/directionality.md", "offset": 100}),
        _tool_result("call-2"),
        _ai_tool_call("call-3", "read_file", {"file_path": "/directionality.md", "offset": 200}),
    ]
    request = _request(
        call_id="call-3",
        name="read_file",
        args={"file_path": "/directionality.md", "offset": 200},
        messages=messages,
        tool=read_file,
    )
    calls: list[ToolCallRequest] = []

    result = RepeatedToolCallMiddleware(max_repeats=2).wrap_tool_call(request, _handler(calls))

    assert len(calls) == 1
    assert isinstance(result, ToolMessage)
    assert result.status == "success"


def test_human_message_resets_consecutive_repeat_count() -> None:
    read_file = _read_file_tool()
    messages = [
        HumanMessage(content="first request"),
        _ai_tool_call("call-1", "read_file", {"file_path": "/directionality.md", "offset": 100}),
        _tool_result("call-1"),
        _ai_tool_call("call-2", "read_file", {"file_path": "/directionality.md", "offset": 100}),
        _tool_result("call-2"),
        HumanMessage(content="read that section again"),
        _ai_tool_call("call-3", "read_file", {"file_path": "/directionality.md", "offset": 100}),
    ]
    request = _request(
        call_id="call-3",
        name="read_file",
        args={"file_path": "/directionality.md", "offset": 100},
        messages=messages,
        tool=read_file,
    )
    calls: list[ToolCallRequest] = []

    result = RepeatedToolCallMiddleware(max_repeats=2).wrap_tool_call(request, _handler(calls))

    assert len(calls) == 1
    assert isinstance(result, ToolMessage)
    assert result.status == "success"


def test_falls_back_to_stable_json_for_tools_without_schema() -> None:
    messages = [
        HumanMessage(content="search"),
        _ai_tool_call("call-1", "web_search", {"query": "LangGraph", "limit": 5}),
        _tool_result("call-1", name="web_search"),
        _ai_tool_call("call-2", "web_search", {"limit": 5, "query": "LangGraph"}),
    ]
    request = _request(
        call_id="call-2",
        name="web_search",
        args={"limit": 5, "query": "LangGraph"},
        messages=messages,
    )
    calls: list[ToolCallRequest] = []

    result = RepeatedToolCallMiddleware(max_repeats=1).wrap_tool_call(request, _handler(calls))

    assert calls == []
    assert isinstance(result, ToolMessage)
    assert result.status == "error"


def test_blocks_repeated_tool_calls_in_same_ai_message() -> None:
    read_file = _read_file_tool()
    messages = [
        HumanMessage(content="read directionality"),
        AIMessage(
            content="",
            tool_calls=[
                {"id": "call-1", "name": "read_file", "args": {"file_path": "/directionality.md", "offset": 100}},
                {"id": "call-2", "name": "read_file", "args": {"file_path": "/directionality.md", "offset": 100}},
            ],
        ),
    ]
    request = _request(
        call_id="call-2",
        name="read_file",
        args={"file_path": "/directionality.md", "offset": 100},
        messages=messages,
        tool=read_file,
    )
    calls: list[ToolCallRequest] = []

    result = RepeatedToolCallMiddleware(max_repeats=1).wrap_tool_call(request, _handler(calls))

    assert calls == []
    assert isinstance(result, ToolMessage)
    assert result.status == "error"


def test_rejects_invalid_max_repeats() -> None:
    with pytest.raises(ValueError, match="max_repeats must be >= 1"):
        RepeatedToolCallMiddleware(max_repeats=0)


async def test_async_wrap_tool_call_blocks_repeated_call() -> None:
    read_file = _read_file_tool()
    messages = [
        HumanMessage(content="read directionality"),
        _ai_tool_call("call-1", "read_file", {"file_path": "/directionality.md", "offset": 100}),
        _tool_result("call-1"),
        _ai_tool_call("call-2", "read_file", {"file_path": "/directionality.md", "offset": 100}),
    ]
    request = _request(
        call_id="call-2",
        name="read_file",
        args={"file_path": "/directionality.md", "offset": 100},
        messages=messages,
        tool=read_file,
    )
    calls: list[ToolCallRequest] = []

    async def handle(req: ToolCallRequest) -> ToolMessage:
        calls.append(req)
        return ToolMessage(content="handler called", tool_call_id=req.tool_call["id"], name=req.tool_call["name"])

    result = await RepeatedToolCallMiddleware(max_repeats=1).awrap_tool_call(request, handle)

    assert calls == []
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
