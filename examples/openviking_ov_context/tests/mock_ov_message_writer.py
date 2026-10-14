"""Test-only writer double for OV middleware hook tests and live smoke tests."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from openviking.integrations.langchain.ov_context.session_writer import WriteResult


class MockOVMessageWriter:
    """Return the unwritten message slice without making an OV write request."""

    def __init__(self, session_id: str, write_index: int = 0) -> None:
        self.session_id = session_id
        self.write_index = write_index
        self.calls: list[tuple[str, int, tuple[Any, ...]]] = []

    async def write(
        self,
        *,
        session_id: str,
        raw_messages: Sequence[Any],
        write_index: int,
        run_id: str,
    ) -> WriteResult:
        if session_id != self.session_id or not run_id:
            raise ValueError("writer session_id and run_id must match the active run")
        if write_index != self.write_index:
            raise ValueError("write_index does not match the mock writer cursor")

        delta = tuple(raw_messages)
        end = write_index + len(delta)
        self.calls.append((session_id, write_index, delta))
        self.write_index = end
        return WriteResult(
            model_messages=list(delta),
            write_index=end,
            written_count=len(delta),
        )
