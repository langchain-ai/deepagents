from deepagents_codex.response import (
    collect_response_events,
    parse_stream_event,
)


class TestCollectResponseEvents:
    def test_text_response(self) -> None:
        events = [
            {"type": "response.output_text.delta", "delta": "Hello"},
            {"type": "response.output_text.delta", "delta": " world!"},
            {
                "type": "response.completed",
                "response": {
                    "model": "gpt-5.1-codex",
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            },
        ]
        result = collect_response_events(events)
        assert len(result.generations) == 1
        assert result.generations[0].message.content == "Hello world!"

    def test_tool_call_response(self) -> None:
        events = [
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "get_weather",
                    "arguments": '{"city": "SF"}',
                },
            },
            {
                "type": "response.completed",
                "response": {"model": "gpt-5.1-codex"},
            },
        ]
        result = collect_response_events(events)
        msg = result.generations[0].message
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0]["name"] == "get_weather"
        assert msg.tool_calls[0]["args"] == {"city": "SF"}

    def test_empty_events(self) -> None:
        result = collect_response_events([])
        assert len(result.generations) == 1
        assert result.generations[0].message.content == ""


class TestParseStreamEvent:
    def test_text_delta(self) -> None:
        event = {"type": "response.output_text.delta", "delta": "Hello"}
        chunk = parse_stream_event(event)
        assert chunk is not None
        assert chunk.message.content == "Hello"

    def test_function_call_done_uses_tool_call_chunks(self) -> None:
        """P1-3: streaming tool calls must use tool_call_chunks, not tool_calls."""
        event = {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": "call_123",
                "name": "search",
                "arguments": "{}",
            },
        }
        chunk = parse_stream_event(event)
        assert chunk is not None
        assert len(chunk.message.tool_call_chunks) == 1
        assert chunk.message.tool_call_chunks[0]["name"] == "search"

    def test_completed_event(self) -> None:
        event = {
            "type": "response.completed",
            "response": {
                "model": "gpt-5.1-codex",
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            },
        }
        chunk = parse_stream_event(event)
        assert chunk is not None
        assert chunk.message.response_metadata["model"] == "gpt-5.1-codex"

    def test_irrelevant_event_returns_none(self) -> None:
        event = {"type": "response.created", "response": {}}
        chunk = parse_stream_event(event)
        assert chunk is None

    def test_non_function_output_item_returns_none(self) -> None:
        event = {
            "type": "response.output_item.done",
            "item": {"type": "message", "content": "hi"},
        }
        chunk = parse_stream_event(event)
        assert chunk is None
