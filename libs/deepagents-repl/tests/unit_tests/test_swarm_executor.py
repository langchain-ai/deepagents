"""Tests for ``execute_swarm`` — table-oriented dispatch."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from deepagents_repl._swarm.executor import SwarmExecutionOptions, execute_swarm


def _mock_subagent(response: Any = None, delay: float = 0.0) -> Runnable:
    if response is None:
        response = {"messages": [AIMessage("done")]}
    mock = AsyncMock(spec=Runnable)

    async def _ainvoke(state: dict, config: Any = None) -> Any:
        if delay > 0:
            await asyncio.sleep(delay)
        if isinstance(response, Exception):
            raise response
        return response

    mock.ainvoke.side_effect = _ainvoke
    return mock


class _TableStore:
    """In-memory table store simulating the pending-writes + read-through buffer."""

    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self.contents: dict[str, str] = dict(initial or {})
        self.write_calls: list[tuple[str, str]] = []

    def write(self, path: str, content: str) -> None:
        self.contents[path] = content
        self.write_calls.append((path, content))

    async def read(self, path: str) -> str:
        return self.contents[path]


def _table_options(
    store: _TableStore,
    *,
    instruction: str = "Process {id}.",
    subagent_graphs: dict[str, Runnable] | None = None,
    **overrides: Any,
) -> SwarmExecutionOptions:
    return SwarmExecutionOptions(
        file="/t.jsonl",
        instruction=instruction,
        subagent_graphs=subagent_graphs or {"general-purpose": _mock_subagent()},
        read=store.read,
        write=store.write,
        **overrides,
    )


def _seed_table(rows: list[dict[str, Any]]) -> _TableStore:
    content = "\n".join(json.dumps(r) for r in rows) + "\n"
    return _TableStore({"/t.jsonl": content})


class TestHappyPath:
    async def test_dispatches_all_rows(self) -> None:
        store = _seed_table([{"id": "a"}, {"id": "b"}])
        subagent = _mock_subagent({"messages": [AIMessage("ok")]})
        summary = await execute_swarm(
            _table_options(store, subagent_graphs={"general-purpose": subagent})
        )
        assert summary.total == 2
        assert summary.completed == 2
        assert summary.skipped == 0
        assert subagent.ainvoke.call_count == 2

    async def test_streams_results_into_rows(self) -> None:
        store = _seed_table([{"id": "a"}, {"id": "b"}])
        await execute_swarm(
            _table_options(
                store,
                subagent_graphs={
                    "general-purpose": _mock_subagent(
                        {"messages": [AIMessage("answer")]}
                    )
                },
            )
        )
        final_rows = [json.loads(line) for line in store.contents["/t.jsonl"].strip().split("\n")]
        assert all(r.get("result") == "answer" for r in final_rows)

    async def test_custom_column_name(self) -> None:
        store = _seed_table([{"id": "a"}])
        await execute_swarm(
            _table_options(
                store,
                column="analysis",
                subagent_graphs={
                    "general-purpose": _mock_subagent({"messages": [AIMessage("x")]})
                },
            )
        )
        rows = [json.loads(line) for line in store.contents["/t.jsonl"].strip().split("\n")]
        assert rows[0]["analysis"] == "x"
        assert "result" not in rows[0]

    async def test_interpolates_row_values(self) -> None:
        subagent = _mock_subagent()
        store = _seed_table([{"id": "a", "file": "/data/one.txt"}])
        await execute_swarm(
            _table_options(
                store,
                instruction="Process {file}.",
                subagent_graphs={"general-purpose": subagent},
            )
        )
        call_state = subagent.ainvoke.call_args.args[0]
        assert call_state["messages"][0].content == "Process /data/one.txt."


class TestFilter:
    async def test_skips_non_matching_rows(self) -> None:
        store = _seed_table([
            {"id": "a", "status": "pending"},
            {"id": "b", "status": "done"},
            {"id": "c", "status": "pending"},
        ])
        subagent = _mock_subagent()
        summary = await execute_swarm(
            _table_options(
                store,
                subagent_graphs={"general-purpose": subagent},
                filter={"column": "status", "equals": "pending"},
            )
        )
        assert summary.total == 2
        assert summary.skipped == 1
        assert subagent.ainvoke.call_count == 2

    async def test_skipped_rows_pass_through_unchanged(self) -> None:
        store = _seed_table([
            {"id": "a", "status": "done", "result": "existing"},
            {"id": "b", "status": "pending"},
        ])
        await execute_swarm(
            _table_options(
                store,
                subagent_graphs={
                    "general-purpose": _mock_subagent({"messages": [AIMessage("new")]})
                },
                filter={"column": "status", "equals": "pending"},
            )
        )
        rows = [json.loads(line) for line in store.contents["/t.jsonl"].strip().split("\n")]
        assert rows[0] == {"id": "a", "status": "done", "result": "existing"}
        assert rows[1]["result"] == "new"


class TestInterpolationFailures:
    async def test_missing_column_marks_row_failed(self) -> None:
        store = _seed_table([{"id": "a"}, {"id": "b", "name": "Bob"}])
        summary = await execute_swarm(
            _table_options(
                store,
                instruction="Hi {name}",
                subagent_graphs={
                    "general-purpose": _mock_subagent({"messages": [AIMessage("ok")]})
                },
            )
        )
        # Row "a" fails interpolation → counted as failed, not dispatched.
        assert summary.completed == 1
        assert summary.failed == 1
        assert any(f["id"] == "a" and "Interpolation" in f["error"] for f in summary.failed_tasks)


class TestResponseSchema:
    async def test_parses_result_as_json_when_schema_provided(self) -> None:
        json_result = '{"label": "bug", "confidence": 0.9}'
        store = _seed_table([{"id": "a"}])
        await execute_swarm(
            _table_options(
                store,
                subagent_graphs={
                    "general-purpose": _mock_subagent(
                        {"messages": [AIMessage(json_result)]}
                    )
                },
                response_schema={
                    "type": "object",
                    "properties": {"label": {"type": "string"}},
                },
            )
        )
        rows = [json.loads(line) for line in store.contents["/t.jsonl"].strip().split("\n")]
        # Structured results are spread onto the row — each property becomes
        # a top-level column instead of a nested object under `result`.
        assert rows[0] == {"id": "a", "label": "bug", "confidence": 0.9}

    async def test_factory_called_with_schema(self) -> None:
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        captured: list[Any] = []
        default = _mock_subagent()
        variant = _mock_subagent({"messages": [AIMessage('{"x":"y"}')]})

        def factory(fmt: Any) -> Any:
            captured.append(fmt)
            return variant

        store = _seed_table([{"id": "a"}, {"id": "b"}])
        summary = await execute_swarm(
            _table_options(
                store,
                subagent_graphs={"general-purpose": default},
                subagent_factories={"general-purpose": factory},
                response_schema=schema,
            )
        )
        # Factory called exactly once (both tasks share the same schema).
        assert len(captured) == 1
        assert captured[0] == schema
        assert summary.completed == 2
        # Variant, not default, was invoked.
        assert variant.ainvoke.call_count == 2
        assert default.ainvoke.call_count == 0

    async def test_rejects_non_object_schema(self) -> None:
        store = _seed_table([{"id": "a"}])
        with pytest.raises(ValueError, match='must have type "object"'):
            await execute_swarm(
                _table_options(
                    store,
                    response_schema={"type": "array", "items": {"type": "string"}},
                )
            )

    async def test_rejects_missing_properties(self) -> None:
        store = _seed_table([{"id": "a"}])
        with pytest.raises(ValueError, match='must define "properties"'):
            await execute_swarm(
                _table_options(
                    store,
                    response_schema={"type": "object"},
                )
            )


class TestErrors:
    async def test_subagent_errors_become_failed_rows(self) -> None:
        store = _seed_table([{"id": "a"}])
        summary = await execute_swarm(
            _table_options(
                store,
                subagent_graphs={"general-purpose": _mock_subagent(RuntimeError("boom"))},
            )
        )
        assert summary.failed == 1
        assert summary.failed_tasks[0] == {"id": "a", "error": "boom"}

    async def test_unknown_subagent_type_raises(self) -> None:
        store = _seed_table([{"id": "a"}])
        with pytest.raises(ValueError, match="Unknown subagent type"):
            await execute_swarm(
                _table_options(store, subagent_type="nonexistent")
            )


class TestCancellation:
    async def test_pre_set_event_short_circuits(self) -> None:
        event = asyncio.Event()
        event.set()
        store = _seed_table([{"id": "a"}, {"id": "b"}])
        summary = await execute_swarm(_table_options(store, cancel_event=event))
        assert summary.failed == 2
        assert all(r.error == "Aborted" for r in summary.results)
