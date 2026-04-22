"""Tests for batched subagent dispatch."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable

from deepagents_repl._swarm.batching import (
    _compose_batch_instruction,
    _enforce_batch_count,
    _group_into_batches,
    _is_pre_wrapped_batch_schema,
    _unpack_batch_result,
    _wrap_batch_schema,
)
from deepagents_repl._swarm.types import SwarmTaskSpec

# ---------------------------------------------------------------------------
# Schema wrapping
# ---------------------------------------------------------------------------


class TestWrapBatchSchema:
    def test_wraps_simple_per_item_schema(self) -> None:
        item = {
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
        }
        wrapped = _wrap_batch_schema(item, 3)
        assert wrapped["type"] == "object"
        assert wrapped["required"] == ["results"]
        results = wrapped["properties"]["results"]
        assert results["type"] == "array"
        assert results["minItems"] == 3
        assert results["maxItems"] == 3
        item_shape = results["items"]
        # `id` is prepended; user's properties are preserved.
        assert item_shape["properties"]["id"] == {"type": "string"}
        assert item_shape["properties"]["label"] == {"type": "string"}
        assert item_shape["required"] == ["id", "label"]

    def test_pre_wrapped_schema_only_gets_count_constraints(self) -> None:
        """Orchestrator-authored wrapper is preserved; only minItems/maxItems
        are stamped on so other fields (descriptions, extra props) survive."""
        pre = {
            "type": "object",
            "properties": {
                "results": {
                    "description": "User-authored description",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "score": {"type": "number"},
                        },
                        "required": ["id", "score"],
                    },
                },
            },
            "required": ["results"],
        }
        wrapped = _wrap_batch_schema(pre, 5)
        results = wrapped["properties"]["results"]
        assert results["description"] == "User-authored description"
        assert results["minItems"] == 5
        assert results["maxItems"] == 5
        # Untouched item shape.
        assert "score" in results["items"]["properties"]


def test_is_pre_wrapped_batch_schema_detection() -> None:
    pre = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {"type": "object", "properties": {"id": {"type": "string"}}},
            }
        },
    }
    assert _is_pre_wrapped_batch_schema(pre) is True
    # Missing id — not pre-wrapped.
    no_id = {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {"type": "object", "properties": {"x": {"type": "string"}}},
            }
        },
    }
    assert _is_pre_wrapped_batch_schema(no_id) is False
    # Plain per-item schema.
    plain = {"type": "object", "properties": {"label": {"type": "string"}}}
    assert _is_pre_wrapped_batch_schema(plain) is False


def test_enforce_batch_count_is_noop_when_no_results_property() -> None:
    """If the schema doesn't have a results property, the function returns
    the schema unchanged rather than fabricating one."""
    assert _enforce_batch_count({"type": "object"}, 3) == {"type": "object"}


# ---------------------------------------------------------------------------
# Chunking + prompt composition
# ---------------------------------------------------------------------------


def test_group_into_batches_uneven_last_chunk() -> None:
    assert _group_into_batches([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert _group_into_batches([], 3) == []
    assert _group_into_batches([1, 2, 3], 5) == [[1, 2, 3]]


def test_compose_batch_instruction_lists_each_task_with_id() -> None:
    batch = [
        SwarmTaskSpec(id="a", description="do A"),
        SwarmTaskSpec(id="b", description="do B"),
    ]
    out = _compose_batch_instruction(batch, context=None)
    assert "Process 2 items" in out
    assert "[a] do A" in out
    assert "[b] do B" in out


def test_compose_batch_instruction_prepends_context() -> None:
    batch = [SwarmTaskSpec(id="a", description="do A")]
    out = _compose_batch_instruction(batch, context="Domain rules apply.")
    assert out.startswith("Domain rules apply.\n\n---\n\n")


# ---------------------------------------------------------------------------
# Result unpacking
# ---------------------------------------------------------------------------


def test_unpack_batch_result_matches_by_id() -> None:
    batch = [SwarmTaskSpec(id="a", description="x"), SwarmTaskSpec(id="b", description="y")]
    raw = [{"id": "b", "label": "neg"}, {"id": "a", "label": "pos"}]
    results = _unpack_batch_result(batch, raw)
    assert results[0].id == "a"
    assert json.loads(results[0].result or "") == {"label": "pos"}
    assert results[1].id == "b"
    assert json.loads(results[1].result or "") == {"label": "neg"}


def test_unpack_batch_result_marks_missing_id_failed() -> None:
    batch = [SwarmTaskSpec(id="a", description="x"), SwarmTaskSpec(id="b", description="y")]
    raw = [{"id": "a", "label": "pos"}]  # b missing
    results = _unpack_batch_result(batch, raw)
    assert results[0].status == "completed"
    assert results[1].status == "failed"
    assert "b" in (results[1].error or "")


# ---------------------------------------------------------------------------
# End-to-end via execute_swarm
# ---------------------------------------------------------------------------


from deepagents_repl._swarm.executor import (  # noqa: E402
    SwarmExecutionOptions,
    execute_swarm,
)
from deepagents_repl._swarm.parse import parse_table_jsonl, serialize_table_jsonl  # noqa: E402


class _TableStore:
    """Minimal in-eval read/write callbacks backed by a dict."""

    def __init__(self) -> None:
        self.contents: dict[str, str] = {}

    def write(self, path: str, content: str) -> None:
        self.contents[path] = content

    async def read(self, path: str) -> str:
        return self.contents.get(path, "")

    def seed(self, path: str, rows: list[dict[str, Any]]) -> None:
        self.contents[path] = serialize_table_jsonl(rows)


def _options(store: _TableStore, **overrides: Any) -> SwarmExecutionOptions:
    base: dict[str, Any] = {
        "file": "/t.jsonl",
        "instruction": "do {id}",
        "subagent_graphs": {},
        "read": store.read,
        "write": store.write,
    }
    base.update(overrides)
    return SwarmExecutionOptions(**base)


def _batch_subagent(returns: dict[str, Any]) -> Runnable:
    """Mock subagent that returns ``{messages: [AIMessage(json.dumps(returns))]}``."""
    mock = AsyncMock(spec=Runnable)

    async def _ainvoke(_state: dict, _config: Any = None) -> dict:
        return {"messages": [AIMessage(json.dumps(returns))]}

    mock.ainvoke.side_effect = _ainvoke
    return mock


async def test_execute_swarm_batches_when_batch_size_set() -> None:
    """Two batches of 2 should result in exactly 2 subagent calls (not 4)."""
    store = _TableStore()
    store.seed(
        "/t.jsonl",
        [
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ],
    )
    subagent = _batch_subagent(
        {"results": [{"id": "a", "label": "x"}, {"id": "b", "label": "y"}]}
    )
    summary = await execute_swarm(
        _options(
            store,
            subagent_graphs={"general-purpose": subagent},
            response_schema={"type": "object", "properties": {"label": {"type": "string"}}},
            batch_size=2,
        )
    )
    # 4 rows / batch_size=2 = 2 batches → 2 subagent calls.
    # First batch is mocked to return [a, b]; second batch reuses the same mock
    # so it returns the same results — c/d will be marked failed (no match).
    assert subagent.ainvoke.call_count == 2
    assert summary.total == 4
    assert summary.completed == 2  # only a/b matched
    assert summary.failed == 2     # c/d: "No result returned for id"


async def test_execute_swarm_batch_size_requires_response_schema() -> None:
    store = _TableStore()
    store.seed("/t.jsonl", [{"id": "a"}])
    with pytest.raises(ValueError, match="batch_size requires response_schema"):
        await execute_swarm(
            _options(
                store,
                subagent_graphs={"general-purpose": _batch_subagent({"results": []})},
                batch_size=2,
            )
        )


async def test_execute_swarm_batch_size_must_be_positive() -> None:
    store = _TableStore()
    store.seed("/t.jsonl", [{"id": "a"}])
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        await execute_swarm(
            _options(
                store,
                subagent_graphs={"general-purpose": _batch_subagent({"results": []})},
                response_schema={"type": "object", "properties": {"x": {"type": "string"}}},
                batch_size=0,
            )
        )


async def test_execute_swarm_batched_results_flatten_onto_rows() -> None:
    """Schema properties should land as top-level columns on each row,
    same as the single-row path."""
    store = _TableStore()
    store.seed("/t.jsonl", [{"id": "a"}, {"id": "b"}])
    subagent = _batch_subagent(
        {
            "results": [
                {"id": "a", "label": "pos"},
                {"id": "b", "label": "neg"},
            ]
        }
    )
    await execute_swarm(
        _options(
            store,
            subagent_graphs={"general-purpose": subagent},
            response_schema={"type": "object", "properties": {"label": {"type": "string"}}},
            batch_size=2,
        )
    )
    rows = parse_table_jsonl(store.contents["/t.jsonl"])
    assert rows == [
        {"id": "a", "label": "pos"},
        {"id": "b", "label": "neg"},
    ]
