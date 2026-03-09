import pytest

from deepagents_codex.sse import aparse_sse_stream, parse_sse_stream


class TestParseSSEStream:
    def test_basic_events(self) -> None:
        lines = [
            'data: {"id": "1", "choices": []}',
            'data: {"id": "2", "choices": []}',
            "data: [DONE]",
        ]
        events = list(parse_sse_stream(iter(lines)))
        assert len(events) == 2
        assert events[0]["id"] == "1"
        assert events[1]["id"] == "2"

    def test_done_sentinel(self) -> None:
        lines = [
            'data: {"id": "1"}',
            "data: [DONE]",
            'data: {"id": "should_not_appear"}',
        ]
        events = list(parse_sse_stream(iter(lines)))
        assert len(events) == 1

    def test_empty_lines_ignored(self) -> None:
        lines = [
            "",
            'data: {"id": "1"}',
            "",
            "",
            "data: [DONE]",
        ]
        events = list(parse_sse_stream(iter(lines)))
        assert len(events) == 1

    def test_non_data_lines_ignored(self) -> None:
        lines = [
            "event: message",
            'data: {"id": "1"}',
            ": comment",
            "data: [DONE]",
        ]
        events = list(parse_sse_stream(iter(lines)))
        assert len(events) == 1

    def test_malformed_json(self) -> None:
        lines = [
            "data: not-json",
            'data: {"id": "1"}',
            "data: [DONE]",
        ]
        events = list(parse_sse_stream(iter(lines)))
        assert len(events) == 1
        assert events[0]["id"] == "1"

    def test_partial_lines_with_whitespace(self) -> None:
        lines = [
            '  data: {"id": "1"}  ',
            "data: [DONE]",
        ]
        events = list(parse_sse_stream(iter(lines)))
        assert len(events) == 1


class TestAsyncParseSSEStream:
    @pytest.mark.asyncio
    async def test_basic_events(self) -> None:
        async def lines():
            for line in ['data: {"id": "1"}', "data: [DONE]"]:
                yield line

        events = [e async for e in aparse_sse_stream(lines())]
        assert len(events) == 1
        assert events[0]["id"] == "1"
