"""Tests for ``ToolErrorHandlingMiddleware``."""

from __future__ import annotations

import pytest
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage

from deepagents.middleware.tool_error_handling import ToolErrorHandlingMiddleware


def _runtime(tool_call_id: str = "tc1") -> ToolRuntime:
    return ToolRuntime(
        state={},
        context=None,
        tool_call_id=tool_call_id,
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def _request(name: str = "broken_tool", tool_call_id: str = "tc1") -> ToolCallRequest:
    return ToolCallRequest(
        runtime=_runtime(tool_call_id),
        tool_call={"id": tool_call_id, "name": name, "args": {}},
        state={},
        tool=None,
    )


class TestToolErrorHandlingMiddleware:
    def test_passes_through_successful_result(self) -> None:
        mw = ToolErrorHandlingMiddleware()
        expected = ToolMessage(content="ok", tool_call_id="tc1")
        result = mw.wrap_tool_call(_request(), lambda _req: expected)
        assert result is expected

    def test_catches_exception_returns_error_tool_message(self) -> None:
        mw = ToolErrorHandlingMiddleware()

        def handler(_req: ToolCallRequest) -> ToolMessage:
            msg = "boom"
            raise ValueError(msg)

        result = mw.wrap_tool_call(_request(), handler)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.tool_call_id == "tc1"
        assert result.name == "broken_tool"
        assert "ValueError" in result.content
        assert "boom" in result.content

    def test_does_not_catch_base_exception_by_default(self) -> None:
        mw = ToolErrorHandlingMiddleware()

        def handler(_req: ToolCallRequest) -> ToolMessage:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            mw.wrap_tool_call(_request(), handler)

    def test_restricts_to_configured_exception_types(self) -> None:
        mw = ToolErrorHandlingMiddleware(catch=TypeError)

        def handler(_req: ToolCallRequest) -> ToolMessage:
            msg = "not caught"
            raise ValueError(msg)

        with pytest.raises(ValueError, match="not caught"):
            mw.wrap_tool_call(_request(), handler)

    def test_custom_formatter_is_used(self) -> None:
        mw = ToolErrorHandlingMiddleware(format_error=lambda e, req: f"{req.tool_call['name']}!{e}")

        def handler(_req: ToolCallRequest) -> ToolMessage:
            msg = "x"
            raise RuntimeError(msg)

        result = mw.wrap_tool_call(_request(name="my_tool"), handler)
        assert isinstance(result, ToolMessage)
        assert result.content == "my_tool!x"

    async def test_async_catches_exception(self) -> None:
        mw = ToolErrorHandlingMiddleware()

        async def handler(_req: ToolCallRequest) -> ToolMessage:
            msg = "async boom"
            raise ValueError(msg)

        result = await mw.awrap_tool_call(_request(), handler)
        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert "async boom" in result.content

    async def test_async_passes_through_success(self) -> None:
        mw = ToolErrorHandlingMiddleware()
        expected = ToolMessage(content="ok", tool_call_id="tc1")

        async def handler(_req: ToolCallRequest) -> ToolMessage:
            return expected

        result = await mw.awrap_tool_call(_request(), handler)
        assert result is expected
