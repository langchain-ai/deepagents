"""Tests for CodexClient internals (no network)."""

from deepagents_codex.client import CodexClient


class TestConvertTool:
    def test_nested_chat_completions_format(self) -> None:
        """Nested ``function`` key is flattened to top-level."""
        tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
        result = CodexClient._convert_tool(tool)
        assert result["type"] == "function"
        assert result["name"] == "get_weather"
        assert result["description"] == "Get weather"
        assert "function" not in result
        assert result["parameters"]["properties"]["city"]["type"] == "string"

    def test_already_flat_responses_format(self) -> None:
        """Tools already in Responses format pass through unchanged."""
        tool = {
            "type": "function",
            "name": "search",
            "parameters": {"type": "object", "properties": {}},
        }
        result = CodexClient._convert_tool(tool)
        assert result is tool

    def test_nested_format_without_description(self) -> None:
        """Nested format without optional fields still converts."""
        tool = {
            "type": "function",
            "function": {
                "name": "do_thing",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        result = CodexClient._convert_tool(tool)
        assert result["name"] == "do_thing"
        assert "description" not in result


class TestBuildPayload:
    def test_basic_payload(self) -> None:
        client = CodexClient.__new__(CodexClient)
        payload = client._build_payload(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                }
            ],
            "gpt-5.3-codex",
            instructions="Be helpful",
        )
        assert payload["model"] == "gpt-5.3-codex"
        assert payload["instructions"] == "Be helpful"
        assert payload["stream"] is True
        assert payload["store"] is False
        assert len(payload["input"]) == 1
        assert "tools" not in payload

    def test_payload_with_tools(self) -> None:
        client = CodexClient.__new__(CodexClient)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        payload = client._build_payload(
            [],
            "gpt-5.3-codex",
            tools=tools,
        )
        assert "tools" in payload
        assert payload["tool_choice"] == "auto"
        # Tools should be converted to flat format
        assert payload["tools"][0]["name"] == "search"
        assert "function" not in payload["tools"][0]

    def test_payload_no_tools_when_none(self) -> None:
        client = CodexClient.__new__(CodexClient)
        payload = client._build_payload([], "gpt-5.3-codex", tools=None)
        assert "tools" not in payload
        assert "tool_choice" not in payload

    def test_payload_no_tools_when_empty(self) -> None:
        client = CodexClient.__new__(CodexClient)
        payload = client._build_payload([], "gpt-5.3-codex", tools=[])
        assert "tools" not in payload
        assert "tool_choice" not in payload

    def test_payload_strips_id_from_input_items(self) -> None:
        """Input item IDs are stripped per official Codex CLI behavior."""
        client = CodexClient.__new__(CodexClient)
        items = [
            {
                "id": "msg_123",
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            },
        ]
        payload = client._build_payload(items, "gpt-5.3-codex")
        # ID should be stripped from payload
        assert "id" not in payload["input"][0]
        # Original should not be mutated
        assert items[0]["id"] == "msg_123"
