"""Tests for _convert_stream_chunk and _to_uuid in deepagents_cli.remote_client."""

import uuid
from typing import Any, NamedTuple

import pytest

from deepagents_cli.remote_client import _convert_stream_chunk, _to_uuid


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


class TestToUuid:
    def test_short_id_becomes_valid_uuid(self) -> None:
        result = _to_uuid("461bc7c2")
        uuid.UUID(result)
        assert result == "461bc7c2-0000-0000-0000-000000000000"

    def test_full_uuid_passthrough(self) -> None:
        full = str(uuid.uuid4())
        assert _to_uuid(full) == full

    def test_deterministic(self) -> None:
        assert _to_uuid("abcd1234") == _to_uuid("abcd1234")
