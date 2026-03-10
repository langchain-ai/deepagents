"""Tests for _convert_stream_chunk in deepagents_cli.remote_client."""

from typing import Any, NamedTuple

import pytest

from deepagents_cli.remote_client import _convert_stream_chunk


class StreamPart(NamedTuple):
    event: str
    data: Any


def test_error_event_raises() -> None:
    """Error event with a dict containing 'message' raises RuntimeError."""
    chunk = StreamPart(event="error", data={"message": "something broke"})
    with pytest.raises(RuntimeError, match="something broke"):
        _convert_stream_chunk(chunk, modes=["updates"])


def test_error_event_raises_non_dict() -> None:
    """Error event with a plain string raises RuntimeError."""
    chunk = StreamPart(event="error", data="plain error text")
    with pytest.raises(RuntimeError, match="plain error text"):
        _convert_stream_chunk(chunk, modes=["updates"])


def test_metadata_event_ignored() -> None:
    """Metadata events should be silently ignored (empty list)."""
    chunk = StreamPart(event="metadata", data={"run_id": "abc"})
    result = _convert_stream_chunk(chunk, modes=["updates"])
    assert result == []


def test_end_event_ignored() -> None:
    """End events should be silently ignored (empty list)."""
    chunk = StreamPart(event="end", data=None)
    result = _convert_stream_chunk(chunk, modes=["updates"])
    assert result == []


def test_updates_event() -> None:
    """Updates event with dict data returns a single 3-tuple."""
    data = {"agent": {"messages": [{"role": "ai", "content": "hi"}]}}
    chunk = StreamPart(event="updates", data=data)
    result = _convert_stream_chunk(chunk, modes=["updates"])
    assert result == [((), "updates", data)]
