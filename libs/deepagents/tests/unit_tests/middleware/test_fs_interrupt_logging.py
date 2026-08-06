"""Regression tests for `_fs_interrupt` predicate visibility on path-validation failure.

The silent ``except ValueError: return False`` in the exact and bulk
interrupt predicates is intentional (fail-open by design), but a
breadcrumb for operators is required when an LLM-emitted path trips
``validate_path`` (e.g. contains ``..`` or a Windows-style absolute path).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest

from deepagents.middleware._fs_interrupt import (
    _make_bulk_when_predicate,
    _make_exact_when_predicate,
)
from deepagents.middleware.filesystem import FilesystemPermission

if TYPE_CHECKING:
    import pytest


def _runtime() -> ToolRuntime:
    return ToolRuntime(state={}, context=None, tool_call_id="", store=None, stream_writer=lambda _: None, config={})


def _request(path_arg: str, path_value: str) -> ToolCallRequest:
    return ToolCallRequest(
        runtime=_runtime(),
        tool_call={"id": "call-1", "name": "read_file", "args": {path_arg: path_value}},
        state={},
        tool=None,
    )


class TestFsInterruptLogging:
    def test_exact_predicate_logs_on_invalid_path(self, caplog: pytest.LogCaptureFixture) -> None:
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="interrupt")]
        predicate = _make_exact_when_predicate(rules, "read", "file_path")

        with caplog.at_level(logging.DEBUG, logger="deepagents.middleware._fs_interrupt"):
            assert predicate(_request("file_path", "../etc/passwd")) is False

        assert any("../etc/passwd" in rec.message and "read_file" in rec.message for rec in caplog.records), (
            f"expected a DEBUG log with raw_path and tool name; got: {[r.message for r in caplog.records]}"
        )

    def test_bulk_predicate_logs_on_invalid_path(self, caplog: pytest.LogCaptureFixture) -> None:
        rules = [FilesystemPermission(operations=["read"], paths=["/secrets/**"], mode="interrupt")]
        predicate = _make_bulk_when_predicate(rules, "read", "path", None)

        with caplog.at_level(logging.DEBUG, logger="deepagents.middleware._fs_interrupt"):
            assert predicate(_request("path", "../secrets")) is False

        assert any("../secrets" in rec.message and "read_file" in rec.message for rec in caplog.records), (
            f"expected a DEBUG log with raw_path and tool name; got: {[r.message for r in caplog.records]}"
        )
