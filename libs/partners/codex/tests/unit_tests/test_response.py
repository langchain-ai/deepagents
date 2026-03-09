from deepagents_codex.response import parse_chat_response, parse_stream_chunk


class TestParseChatResponse:
    def test_basic_response(self) -> None:
        data = {
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Hello!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = parse_chat_response(data)
        assert len(result.generations) == 1
        msg = result.generations[0].message
        assert msg.content == "Hello!"

    def test_tool_calls(self) -> None:
        data = {
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "SF"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
        result = parse_chat_response(data)
        msg = result.generations[0].message
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_weather"

    def test_empty_choices(self) -> None:
        data = {"model": "gpt-4o", "choices": []}
        result = parse_chat_response(data)
        assert len(result.generations) == 0


class TestParseStreamChunk:
    def test_content_chunk(self) -> None:
        data = {
            "model": "gpt-4o",
            "choices": [
                {"delta": {"content": "Hello"}, "finish_reason": None}
            ],
        }
        chunk = parse_stream_chunk(data)
        assert chunk.message.content == "Hello"

    def test_empty_choices(self) -> None:
        data = {"model": "gpt-4o", "choices": []}
        chunk = parse_stream_chunk(data)
        assert chunk.message.content == ""

    def test_tool_call_chunk(self) -> None:
        data = {
            "model": "gpt-4o",
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_123",
                                "function": {"name": "search", "arguments": ""},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        }
        chunk = parse_stream_chunk(data)
        assert len(chunk.message.tool_call_chunks) == 1
        assert chunk.message.tool_call_chunks[0]["name"] == "search"
