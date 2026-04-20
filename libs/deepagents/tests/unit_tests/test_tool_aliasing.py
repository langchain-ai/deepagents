"""Unit tests for _ToolAliasingMiddleware."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pytest
from langchain.agents.middleware.types import ExtendedModelResponse, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from deepagents.middleware._tool_aliasing import (
    _rename_tool,
    _rewrite_ai_message_tool_calls,
    _rewrite_response_tool_names,
    _rewrite_tool_message_name,
    _ToolAliasingMiddleware,
    _validate_aliases,
)


@dataclass
class FakeRequest:
    """Minimal stand-in for ``ModelRequest`` in middleware tests.

    Matches just enough of the real interface (``tools``, ``messages``,
    ``override``) for the aliasing middleware to exercise its full code path.
    """

    tools: list[Any] = field(default_factory=list)
    messages: list[BaseMessage] = field(default_factory=list)

    def override(self, **kwargs: Any) -> FakeRequest:
        return FakeRequest(
            tools=kwargs.get("tools", self.tools),
            messages=kwargs.get("messages", self.messages),
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

        tool = StructuredTool.from_function(func=sample, name="execute", description="Run a command")
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

        def handler(request: FakeRequest) -> AIMessage:
            captured_request["tools"] = request.tools
            return model_response

        request = FakeRequest(tools=[tool_dict, ls_tool, other])
        response = mw.wrap_model_call(request, handler)  # type: ignore[arg-type]

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

        def handler(request: FakeRequest) -> ModelResponse[Any]:
            captured_request["tools"] = request.tools
            return ModelResponse(result=[ai_msg])

        request = FakeRequest(tools=[tool_dict, ls_tool])
        response = mw.wrap_model_call(request, handler)  # type: ignore[arg-type]

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

        request = FakeRequest(tools=[tool])
        response = mw.wrap_model_call(request, lambda _r: model_response)  # type: ignore[arg-type]

        assert response is model_response

    def test_no_rename_preserves_request_identity(self) -> None:
        """If nothing matches the alias map, pass the original request through.

        This lets downstream middleware / cache keys that hold onto request
        object identity continue to observe the same reference.
        """
        mw = _ToolAliasingMiddleware(aliases={"execute": "shell_command"})

        request = FakeRequest(
            tools=[{"name": "read_file", "description": "Read"}],
            messages=[HumanMessage(content="hi")],
        )

        captured: dict[str, Any] = {}

        def handler(received: FakeRequest) -> AIMessage:
            captured["request"] = received
            return AIMessage(content="ok")

        mw.wrap_model_call(request, handler)  # type: ignore[arg-type]
        assert captured["request"] is request

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

        async def handler(request: FakeRequest) -> AIMessage:
            captured["tools"] = request.tools
            return model_response

        request = FakeRequest(tools=[tool])
        response = await mw.awrap_model_call(request, handler)  # type: ignore[arg-type]

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

        async def handler(request: FakeRequest) -> ModelResponse[Any]:
            captured["tools"] = request.tools
            return ModelResponse(result=[ai_msg])

        request = FakeRequest(tools=[tool])
        response = await mw.awrap_model_call(request, handler)  # type: ignore[arg-type]

        assert captured["tools"][0]["name"] == "list_dir"
        assert isinstance(response, ModelResponse)
        rewritten = response.result[0]
        assert isinstance(rewritten, AIMessage)
        assert rewritten.tool_calls[0]["name"] == "ls"


class TestValidateAliases:
    """Tests for alias-map validation."""

    def test_empty_passes(self) -> None:
        _validate_aliases({})

    def test_simple_valid_mapping_passes(self) -> None:
        _validate_aliases({"execute": "shell_command", "list_dir": "ls"})

    def test_duplicate_values_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be unique"):
            _validate_aliases({"a": "x", "b": "x"})

    def test_key_value_collision_rejected(self) -> None:
        """`b` appears as both a canonical name and as an alias for `a`."""
        with pytest.raises(ValueError, match="must be disjoint"):
            _validate_aliases({"a": "b", "b": "c"})

    def test_constructor_rejects_invalid_aliases(self) -> None:
        with pytest.raises(ValueError, match="must be unique"):
            _ToolAliasingMiddleware(aliases={"a": "x", "b": "x"})


class TestRewriteToolMessageName:
    """Tests for _rewrite_tool_message_name."""

    def test_rewrites_matching_name(self) -> None:
        msg = ToolMessage(content="ok", tool_call_id="1", name="execute")
        result = _rewrite_tool_message_name(msg, {"execute": "shell_command"})
        assert result.name == "shell_command"
        assert result.tool_call_id == "1"
        assert result.content == "ok"
        assert msg.name == "execute"  # original untouched

    def test_no_match_returns_original(self) -> None:
        msg = ToolMessage(content="ok", tool_call_id="1", name="read_file")
        result = _rewrite_tool_message_name(msg, {"execute": "shell_command"})
        assert result is msg

    def test_missing_name_returns_original(self) -> None:
        msg = ToolMessage(content="ok", tool_call_id="1")
        result = _rewrite_tool_message_name(msg, {"execute": "shell_command"})
        assert result is msg


class TestHistoryRewriteOutbound:
    """Outbound canonical→alias rewriting of request.messages.

    This is the main-bug fix: without this, turn N+1 ships aliased tools
    to the model but canonical names in the past-turn tool calls, so the
    model sees two names for the same tool.
    """

    def test_history_tool_calls_rewritten_on_outbound(self) -> None:
        """Canonical names in AIMessage.tool_calls become aliases for the model."""
        mw = _ToolAliasingMiddleware(aliases={"execute": "shell_command"})

        past_ai = AIMessage(
            content="",
            tool_calls=[
                {"name": "execute", "args": {"command": "ls"}, "id": "1", "type": "tool_call"},
            ],
        )
        past_tool = ToolMessage(content="file.py", tool_call_id="1", name="execute")

        request = FakeRequest(
            tools=[{"name": "execute", "description": "Run cmd"}],
            messages=[HumanMessage(content="run ls"), past_ai, past_tool],
        )

        captured: dict[str, Any] = {}

        def handler(req: FakeRequest) -> AIMessage:
            captured["messages"] = req.messages
            captured["tools"] = req.tools
            return AIMessage(content="done")

        mw.wrap_model_call(request, handler)  # type: ignore[arg-type]

        # Model saw aliased tool catalog
        assert captured["tools"][0]["name"] == "shell_command"
        # Model saw aliased past tool calls
        sent_ai = captured["messages"][1]
        assert isinstance(sent_ai, AIMessage)
        assert sent_ai.tool_calls[0]["name"] == "shell_command"
        # Model saw aliased ToolMessage.name
        sent_tool = captured["messages"][2]
        assert isinstance(sent_tool, ToolMessage)
        assert sent_tool.name == "shell_command"

        # Caller's original messages are untouched (defensive copy)
        assert past_ai.tool_calls[0]["name"] == "execute"
        assert past_tool.name == "execute"

    def test_history_with_invalid_tool_calls(self) -> None:
        """Invalid (malformed-args) past tool calls are also renamed."""
        mw = _ToolAliasingMiddleware(aliases={"execute": "shell_command"})

        past_ai = AIMessage(
            content="",
            invalid_tool_calls=[
                {"name": "execute", "args": "bad-json", "id": "1", "error": "parse"},
            ],
        )
        request = FakeRequest(tools=[], messages=[past_ai])

        captured: dict[str, Any] = {}

        def handler(req: FakeRequest) -> AIMessage:
            captured["messages"] = req.messages
            return AIMessage(content="ok")

        mw.wrap_model_call(request, handler)  # type: ignore[arg-type]

        sent = captured["messages"][0]
        assert isinstance(sent, AIMessage)
        assert sent.invalid_tool_calls[0]["name"] == "shell_command"

    def test_no_history_rename_preserves_other_messages(self) -> None:
        """HumanMessages and unrelated AIMessages pass through untouched."""
        mw = _ToolAliasingMiddleware(aliases={"execute": "shell_command"})

        human = HumanMessage(content="hi")
        bare_ai = AIMessage(content="hello")
        request = FakeRequest(tools=[], messages=[human, bare_ai])

        captured: dict[str, Any] = {}

        def handler(req: FakeRequest) -> AIMessage:
            captured["messages"] = req.messages
            return AIMessage(content="ok")

        mw.wrap_model_call(request, handler)  # type: ignore[arg-type]

        # Object identity preserved for unchanged messages
        assert captured["messages"][0] is human
        assert captured["messages"][1] is bare_ai

    async def test_history_rewrite_async(self) -> None:
        """Async path rewrites history identically."""
        mw = _ToolAliasingMiddleware(aliases={"list_dir": "ls"})

        past_ai = AIMessage(
            content="",
            tool_calls=[
                {"name": "list_dir", "args": {"path": "/"}, "id": "1", "type": "tool_call"},
            ],
        )
        request = FakeRequest(tools=[], messages=[past_ai])

        captured: dict[str, Any] = {}

        async def handler(req: FakeRequest) -> AIMessage:
            captured["messages"] = req.messages
            return AIMessage(content="ok")

        await mw.awrap_model_call(request, handler)  # type: ignore[arg-type]

        sent = captured["messages"][0]
        assert isinstance(sent, AIMessage)
        assert sent.tool_calls[0]["name"] == "ls"


class TestRenameToolWarning:
    """_rename_tool warns when a tool cannot be renamed."""

    def test_warns_on_unrenameable_tool(self, caplog: pytest.LogCaptureFixture) -> None:
        """Plain object with no model_copy and not a dict triggers the warning."""

        class WeirdTool:
            name = "execute"

        weird = WeirdTool()
        with caplog.at_level(logging.WARNING, logger="deepagents.middleware._tool_aliasing"):
            result = _rename_tool(weird, "shell_command")

        assert result is weird  # unchanged
        assert any("cannot be renamed" in record.message for record in caplog.records), (
            f"Expected warning not logged. Got: {[r.message for r in caplog.records]}"
        )

    def test_no_warning_on_dict_tool(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger="deepagents.middleware._tool_aliasing"):
            _rename_tool({"name": "execute"}, "shell_command")
        assert not any("cannot be renamed" in r.message for r in caplog.records)

    def test_no_warning_on_basetool(self, caplog: pytest.LogCaptureFixture) -> None:
        def sample(text: str) -> str:
            return text

        tool = StructuredTool.from_function(func=sample, name="execute", description="Run cmd")
        with caplog.at_level(logging.WARNING, logger="deepagents.middleware._tool_aliasing"):
            _rename_tool(tool, "shell_command")
        assert not any("cannot be renamed" in r.message for r in caplog.records)
