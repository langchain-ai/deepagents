"""Unit tests for _ToolAliasingMiddleware."""

from __future__ import annotations

from typing import Any, ClassVar

from langchain.agents.middleware.types import ExtendedModelResponse, ModelResponse
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool

from deepagents.middleware._tool_aliasing import (
    _rename_tool,
    _rewrite_ai_message_tool_calls,
    _rewrite_response_tool_names,
    _ToolAliasingMiddleware,
)


class TestRenameTool:
    """Tests for _rename_tool helper."""

    def test_rename_dict_tool(self) -> None:
        tool: dict[str, Any] = {"name": "execute", "description": "Run a command"}
        result = _rename_tool(tool, "shell_command")
        assert result["name"] == "shell_command"
        assert result["description"] == "Run a command"
        assert tool["name"] == "execute"

    def test_rename_basetool(self) -> None:
        def sample(text: str) -> str:
            return text

        tool = StructuredTool.from_function(
            func=sample, name="execute", description="Run a command"
        )
        result = _rename_tool(tool, "shell_command")
        assert result.name == "shell_command"
        assert result.description == "Run a command"
        assert tool.name == "execute"

    def test_rename_plain_callable_is_noop(self) -> None:
        def my_func() -> None:
            pass

        result = _rename_tool(my_func, "new_name")
        assert result is my_func


class TestRewriteAIMessageToolCalls:
    """Tests for _rewrite_ai_message_tool_calls."""

    def test_rewrites_matching_tool_calls(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "shell_command", "args": {"command": "ls"}, "id": "1", "type": "tool_call"},
                {"name": "read_file", "args": {"path": "/a"}, "id": "2", "type": "tool_call"},
            ],
        )
        aliases = {"shell_command": "execute"}
        result = _rewrite_ai_message_tool_calls(msg, aliases)
        assert result.tool_calls[0]["name"] == "execute"
        assert result.tool_calls[1]["name"] == "read_file"
        assert msg.tool_calls[0]["name"] == "shell_command"

    def test_no_match_returns_original(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "read_file", "args": {"path": "/a"}, "id": "1", "type": "tool_call"},
            ],
        )
        aliases = {"shell_command": "execute"}
        result = _rewrite_ai_message_tool_calls(msg, aliases)
        assert result is msg

    def test_empty_tool_calls_returns_original(self) -> None:
        msg = AIMessage(content="hello")
        aliases = {"shell_command": "execute"}
        result = _rewrite_ai_message_tool_calls(msg, aliases)
        assert result is msg

    def test_rewrites_invalid_tool_calls(self) -> None:
        msg = AIMessage(
            content="",
            invalid_tool_calls=[
                {"name": "shell_command", "args": "bad", "id": "1", "error": "parse error"},
            ],
        )
        aliases = {"shell_command": "execute"}
        result = _rewrite_ai_message_tool_calls(msg, aliases)
        assert result.invalid_tool_calls[0]["name"] == "execute"


class TestRewriteResponseToolNames:
    """Tests for _rewrite_response_tool_names with various response types."""

    def test_ai_message(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "list_dir", "args": {"path": "/"}, "id": "1", "type": "tool_call"},
            ],
        )
        result = _rewrite_response_tool_names(msg, {"list_dir": "ls"})
        assert isinstance(result, AIMessage)
        assert result.tool_calls[0]["name"] == "ls"

    def test_model_response(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "list_dir", "args": {"path": "/"}, "id": "1", "type": "tool_call"},
                {"name": "read_file", "args": {"path": "/a"}, "id": "2", "type": "tool_call"},
            ],
        )
        resp = ModelResponse(result=[msg])
        result = _rewrite_response_tool_names(resp, {"list_dir": "ls"})
        assert isinstance(result, ModelResponse)
        rewritten_msg = result.result[0]
        assert isinstance(rewritten_msg, AIMessage)
        assert rewritten_msg.tool_calls[0]["name"] == "ls"
        assert rewritten_msg.tool_calls[1]["name"] == "read_file"

    def test_model_response_no_match(self) -> None:
        msg = AIMessage(content="hello")
        resp = ModelResponse(result=[msg])
        result = _rewrite_response_tool_names(resp, {"x": "y"})
        assert result is resp

    def test_extended_model_response(self) -> None:
        msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "shell_command", "args": {"cmd": "ls"}, "id": "1", "type": "tool_call"},
            ],
        )
        inner = ModelResponse(result=[msg])
        resp = ExtendedModelResponse(model_response=inner)
        result = _rewrite_response_tool_names(resp, {"shell_command": "execute"})
        assert isinstance(result, ExtendedModelResponse)
        rewritten_msg = result.model_response.result[0]
        assert isinstance(rewritten_msg, AIMessage)
        assert rewritten_msg.tool_calls[0]["name"] == "execute"

    def test_unknown_type_returned_unchanged(self) -> None:
        obj = {"not": "a message"}
        result = _rewrite_response_tool_names(obj, {"x": "y"})
        assert result is obj


class TestToolAliasingMiddleware:
    """Tests for the full middleware round-trip."""

    def test_roundtrip_sync_bare_ai_message(self) -> None:
        """Handler returns a bare AIMessage (legacy path)."""
        mw = _ToolAliasingMiddleware(aliases={"execute": "shell_command", "ls": "list_dir"})

        tool_dict: dict[str, Any] = {"name": "execute", "description": "Run cmd"}
        ls_tool: dict[str, Any] = {"name": "ls", "description": "List files"}
        other: dict[str, Any] = {"name": "read_file", "description": "Read a file"}

        model_response = AIMessage(
            content="",
            tool_calls=[
                {"name": "shell_command", "args": {"command": "git status"}, "id": "1", "type": "tool_call"},
                {"name": "list_dir", "args": {"path": "/"}, "id": "2", "type": "tool_call"},
                {"name": "read_file", "args": {"path": "/a.py"}, "id": "3", "type": "tool_call"},
            ],
        )

        captured_request: dict[str, Any] = {}

        def handler(request: object) -> AIMessage:
            captured_request["tools"] = request.tools  # type: ignore[attr-defined]
            return model_response

        class FakeRequest:
            tools: ClassVar[list[dict[str, Any]]] = [tool_dict, ls_tool, other]

            def override(self, **kwargs: Any) -> FakeRequest:
                new = FakeRequest()
                new.tools = kwargs.get("tools", self.tools)
                return new

        response = mw.wrap_model_call(FakeRequest(), handler)  # type: ignore[arg-type]

        tool_names_sent = [t["name"] for t in captured_request["tools"]]
        assert tool_names_sent == ["shell_command", "list_dir", "read_file"]

        assert isinstance(response, AIMessage)
        assert response.tool_calls[0]["name"] == "execute"
        assert response.tool_calls[1]["name"] == "ls"
        assert response.tool_calls[2]["name"] == "read_file"

    def test_roundtrip_sync_model_response(self) -> None:
        """Handler returns a ModelResponse (realistic path)."""
        mw = _ToolAliasingMiddleware(aliases={"execute": "shell_command", "ls": "list_dir"})

        tool_dict: dict[str, Any] = {"name": "execute", "description": "Run cmd"}
        ls_tool: dict[str, Any] = {"name": "ls", "description": "List files"}

        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "shell_command", "args": {"command": "git status"}, "id": "1", "type": "tool_call"},
                {"name": "list_dir", "args": {"path": "/"}, "id": "2", "type": "tool_call"},
            ],
        )

        captured_request: dict[str, Any] = {}

        def handler(request: object) -> ModelResponse[Any]:
            captured_request["tools"] = request.tools  # type: ignore[attr-defined]
            return ModelResponse(result=[ai_msg])

        class FakeRequest:
            tools: ClassVar[list[dict[str, Any]]] = [tool_dict, ls_tool]

            def override(self, **kwargs: Any) -> FakeRequest:
                new = FakeRequest()
                new.tools = kwargs.get("tools", self.tools)
                return new

        response = mw.wrap_model_call(FakeRequest(), handler)  # type: ignore[arg-type]

        tool_names_sent = [t["name"] for t in captured_request["tools"]]
        assert tool_names_sent == ["shell_command", "list_dir"]

        assert isinstance(response, ModelResponse)
        rewritten = response.result[0]
        assert isinstance(rewritten, AIMessage)
        assert rewritten.tool_calls[0]["name"] == "execute"
        assert rewritten.tool_calls[1]["name"] == "ls"

    def test_empty_aliases_is_noop(self) -> None:
        mw = _ToolAliasingMiddleware(aliases={})

        tool: dict[str, Any] = {"name": "execute", "description": "Run cmd"}
        model_response = AIMessage(content="ok")

        class FakeRequest:
            tools: ClassVar[list[dict[str, Any]]] = [tool]

            def override(self, **kwargs: Any) -> FakeRequest:
                new = FakeRequest()
                new.tools = kwargs.get("tools", self.tools)
                return new

        response = mw.wrap_model_call(FakeRequest(), lambda _r: model_response)  # type: ignore[arg-type]

        assert response is model_response

    async def test_roundtrip_async_bare_ai_message(self) -> None:
        mw = _ToolAliasingMiddleware(aliases={"execute": "shell_command"})

        tool: dict[str, Any] = {"name": "execute", "description": "Run cmd"}
        model_response = AIMessage(
            content="",
            tool_calls=[
                {"name": "shell_command", "args": {"command": "ls"}, "id": "1", "type": "tool_call"},
            ],
        )

        captured: dict[str, Any] = {}

        async def handler(request: object) -> AIMessage:
            captured["tools"] = request.tools  # type: ignore[attr-defined]
            return model_response

        class FakeRequest:
            tools: ClassVar[list[dict[str, Any]]] = [tool]

            def override(self, **kwargs: Any) -> FakeRequest:
                new = FakeRequest()
                new.tools = kwargs.get("tools", self.tools)
                return new

        response = await mw.awrap_model_call(FakeRequest(), handler)  # type: ignore[arg-type]

        assert captured["tools"][0]["name"] == "shell_command"
        assert isinstance(response, AIMessage)
        assert response.tool_calls[0]["name"] == "execute"

    async def test_roundtrip_async_model_response(self) -> None:
        """Async handler returns ModelResponse (realistic path)."""
        mw = _ToolAliasingMiddleware(aliases={"ls": "list_dir"})

        tool: dict[str, Any] = {"name": "ls", "description": "List files"}
        ai_msg = AIMessage(
            content="",
            tool_calls=[
                {"name": "list_dir", "args": {"path": "/"}, "id": "1", "type": "tool_call"},
            ],
        )

        captured: dict[str, Any] = {}

        async def handler(request: object) -> ModelResponse[Any]:
            captured["tools"] = request.tools  # type: ignore[attr-defined]
            return ModelResponse(result=[ai_msg])

        class FakeRequest:
            tools: ClassVar[list[dict[str, Any]]] = [tool]

            def override(self, **kwargs: Any) -> FakeRequest:
                new = FakeRequest()
                new.tools = kwargs.get("tools", self.tools)
                return new

        response = await mw.awrap_model_call(FakeRequest(), handler)  # type: ignore[arg-type]

        assert captured["tools"][0]["name"] == "list_dir"
        assert isinstance(response, ModelResponse)
        rewritten = response.result[0]
        assert isinstance(rewritten, AIMessage)
        assert rewritten.tool_calls[0]["name"] == "ls"
