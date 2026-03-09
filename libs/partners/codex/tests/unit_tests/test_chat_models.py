from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deepagents_codex.chat_models import (
    ChatCodexOAuth,
    _content_to_text,
    _convert_messages,
)


class TestContentToText:
    def test_string_passthrough(self) -> None:
        assert _content_to_text("hello") == "hello"

    def test_list_of_text_blocks(self) -> None:
        content = [
            {"type": "text", "text": "line 1"},
            {"type": "text", "text": "line 2"},
        ]
        assert _content_to_text(content) == "line 1\nline 2"

    def test_list_of_strings(self) -> None:
        assert _content_to_text(["a", "b"]) == "a\nb"

    def test_mixed_list(self) -> None:
        content = ["plain", {"type": "text", "text": "block"}]
        assert _content_to_text(content) == "plain\nblock"

    def test_fallback_to_str(self) -> None:
        assert _content_to_text(42) == "42"


class TestConvertMessages:
    def test_system_becomes_instructions(self) -> None:
        msgs = [SystemMessage(content="You are helpful")]
        instructions, items = _convert_messages(msgs)
        assert instructions == "You are helpful"
        assert items == []

    def test_human_message(self) -> None:
        msgs = [HumanMessage(content="Hi")]
        instructions, items = _convert_messages(msgs)
        assert instructions == ""
        assert len(items) == 1
        assert items[0]["type"] == "message"
        assert items[0]["role"] == "user"
        assert items[0]["content"][0]["text"] == "Hi"

    def test_ai_message_uses_input_text(self) -> None:
        """P1-1: prior assistant turns must use input_text, not output_text."""
        msgs = [AIMessage(content="Hello!")]
        _, items = _convert_messages(msgs)
        assert len(items) == 1
        assert items[0]["type"] == "message"
        assert items[0]["role"] == "assistant"
        assert items[0]["content"][0]["type"] == "input_text"

    def test_ai_with_tool_calls(self) -> None:
        msgs = [
            AIMessage(
                content="",
                tool_calls=[{"id": "call_1", "name": "search", "args": {"q": "test"}}],
            )
        ]
        _, items = _convert_messages(msgs)
        assert len(items) == 1
        assert items[0]["type"] == "function_call"
        assert items[0]["name"] == "search"
        assert items[0]["call_id"] == "call_1"

    def test_tool_message(self) -> None:
        msgs = [ToolMessage(content="result", tool_call_id="call_1")]
        _, items = _convert_messages(msgs)
        assert len(items) == 1
        assert items[0]["type"] == "function_call_output"
        assert items[0]["call_id"] == "call_1"
        assert items[0]["output"] == "result"

    def test_multiple_system_messages_joined(self) -> None:
        msgs = [
            SystemMessage(content="Rule 1"),
            SystemMessage(content="Rule 2"),
        ]
        instructions, _ = _convert_messages(msgs)
        assert "Rule 1" in instructions
        assert "Rule 2" in instructions

    def test_list_content_not_stringified(self) -> None:
        """P1-2: list-based content must not become Python repr."""
        content = [{"type": "text", "text": "structured"}]
        msgs = [HumanMessage(content=content)]
        _, items = _convert_messages(msgs)
        assert items[0]["content"][0]["text"] == "structured"


class TestChatCodexOAuth:
    def test_llm_type(self) -> None:
        model = ChatCodexOAuth()
        assert model._llm_type == "codex-oauth"

    def test_identifying_params(self) -> None:
        model = ChatCodexOAuth(model="gpt-5.1-codex-mini")
        assert model._identifying_params == {"model": "gpt-5.1-codex-mini"}

    def test_default_model(self) -> None:
        model = ChatCodexOAuth()
        assert model.model == "gpt-5.4"

    @patch.object(ChatCodexOAuth, "_get_client")
    def test_generate_collects_events(self, mock_get_client) -> None:
        mock_client = MagicMock()
        mock_client.create_response.return_value = iter(
            [
                {"type": "response.output_text.delta", "delta": "Hi!"},
                {
                    "type": "response.completed",
                    "response": {"model": "gpt-5.1-codex"},
                },
            ]
        )
        mock_get_client.return_value = mock_client

        model = ChatCodexOAuth(streaming=False)
        result = model._generate([HumanMessage(content="Hello")])
        assert result.generations[0].message.content == "Hi!"

    def test_bind_tools_converts_to_responses_format(self) -> None:
        """P1-4: bind_tools must produce Responses API format (flat, not nested)."""
        model = ChatCodexOAuth()
        tool_def = {
            "type": "function",
            "function": {
                "name": "test_tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        bound = model.bind_tools([tool_def])
        assert bound._bound_tools is not None
        assert len(bound._bound_tools) == 1
        # Should be flat Responses format, not nested Chat Completions format
        tool = bound._bound_tools[0]
        assert tool["name"] == "test_tool"
        assert "function" not in tool
        assert tool["type"] == "function"

    def test_bind_tools_already_flat(self) -> None:
        """Tools already in Responses format should pass through unchanged."""
        model = ChatCodexOAuth()
        tool_def = {
            "type": "function",
            "name": "flat_tool",
            "parameters": {"type": "object", "properties": {}},
        }
        bound = model.bind_tools([tool_def])
        assert bound._bound_tools[0]["name"] == "flat_tool"
