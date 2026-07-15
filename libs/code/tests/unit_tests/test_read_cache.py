"""Unit tests for the per-session read_file deduplication middleware."""

from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code.read_cache_middleware import ReadFileCacheMiddleware

_BODY = "line1\nline2\nline3"


def _request(
    thread_id: str = "t1",
    file_path: str = "/app.py",
    offset: int = 0,
    limit: int = 100,
) -> ToolCallRequest:
    runtime = SimpleNamespace(context=CLIContextSchema(thread_id=thread_id))
    return ToolCallRequest(
        tool_call={
            "name": "read_file",
            "args": {"file_path": file_path, "offset": offset, "limit": limit},
            "id": f"call-{offset}-{limit}",
            "type": "tool_call",
        },
        tool=None,
        state={},
        runtime=runtime,  # ty: ignore[invalid-argument-type]
    )


def _read_file_result(request: ToolCallRequest, body: str) -> ToolMessage:
    return ToolMessage(
        content=body,
        name="read_file",
        tool_call_id=request.tool_call["id"],
    )


def test_unchanged_file_returns_marker_on_second_read() -> None:
    """Same unchanged file yields the full body once, then the unchanged marker."""
    mw = ReadFileCacheMiddleware()

    def handler(req: ToolCallRequest) -> ToolMessage:
        return _read_file_result(req, _BODY)

    first = mw.wrap_tool_call(_request(), handler)
    assert isinstance(first, ToolMessage)
    assert first.content == _BODY

    second = mw.wrap_tool_call(_request(), handler)
    assert isinstance(second, ToolMessage)
    assert second.content != _BODY
    assert "unchanged since it was read earlier" in second.content
    assert "3 lines" in second.content


def test_changed_file_returns_full_body() -> None:
    """A changed file returns the fresh full body, not the unchanged marker."""
    mw = ReadFileCacheMiddleware()
    bodies = iter([_BODY, "line1\nCHANGED\nline3"])

    def handler(req: ToolCallRequest) -> ToolMessage:
        return _read_file_result(req, next(bodies))

    mw.wrap_tool_call(_request(), handler)
    second = mw.wrap_tool_call(_request(), handler)
    assert isinstance(second, ToolMessage)
    assert second.content == "line1\nCHANGED\nline3"


def test_distinct_windows_are_cached_separately() -> None:
    """A not-previously-served offset/limit window returns the full body."""
    mw = ReadFileCacheMiddleware()

    def handler(req: ToolCallRequest) -> ToolMessage:
        return _read_file_result(req, _BODY)

    mw.wrap_tool_call(_request(offset=0, limit=100), handler)
    other = mw.wrap_tool_call(_request(offset=100, limit=100), handler)
    assert isinstance(other, ToolMessage)
    assert other.content == _BODY


async def test_async_unchanged_file_returns_marker() -> None:
    """Async path deduplicates a repeated unchanged read the same way."""
    mw = ReadFileCacheMiddleware()

    async def handler(req: ToolCallRequest) -> ToolMessage:  # noqa: RUF029
        return _read_file_result(req, _BODY)

    first = await mw.awrap_tool_call(_request(), handler)
    assert isinstance(first, ToolMessage)
    assert first.content == _BODY

    second = await mw.awrap_tool_call(_request(), handler)
    assert isinstance(second, ToolMessage)
    assert "unchanged since it was read earlier" in second.content
