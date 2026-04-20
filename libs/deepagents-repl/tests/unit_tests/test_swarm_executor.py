"""Port of ``libs/deepagents/src/swarm/executor.test.ts``."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from deepagents.backends.protocol import BackendProtocol, WriteResult
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from deepagents_repl._swarm import (
    SwarmExecutionOptions,
    SwarmTaskSpec,
    execute_swarm,
)


def _make_mock_subagent(
    response: dict | Exception | None = None,
    delay: float = 0.0,
) -> Runnable:
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


class _MockBackend(BackendProtocol):
    """Records writes in-memory; everything else raises."""

    def __init__(self) -> None:
        self.written_files: dict[str, str] = {}

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        self.written_files[file_path] = content
        return WriteResult(path=file_path)

    # Required abstract methods — unused in executor tests.
    def ls(self, path: str) -> Any:  # pragma: no cover
        raise NotImplementedError

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> Any:  # pragma: no cover
        raise NotImplementedError

    def write(self, file_path: str, content: str) -> Any:  # pragma: no cover
        raise NotImplementedError

    def edit(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def grep(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def glob(self, pattern: str, path: str = "/") -> Any:  # pragma: no cover
        raise NotImplementedError

    def upload_files(self, files: list) -> Any:  # pragma: no cover
        raise NotImplementedError

    def download_files(self, paths: list) -> Any:  # pragma: no cover
        raise NotImplementedError


def _build_options(**overrides: Any) -> SwarmExecutionOptions:
    tasks = overrides.pop(
        "tasks",
        [
            SwarmTaskSpec(id="t1", description="Task one"),
            SwarmTaskSpec(id="t2", description="Task two"),
        ],
    )
    subagent_graphs = overrides.pop(
        "subagent_graphs",
        {"general-purpose": _make_mock_subagent()},
    )
    backend = overrides.pop("backend", _MockBackend())
    return SwarmExecutionOptions(
        tasks=tasks,
        subagent_graphs=subagent_graphs,
        backend=backend,
        **overrides,
    )


def _read_first_result(backend: _MockBackend) -> dict:
    for key, content in backend.written_files.items():
        if key.endswith("results.jsonl"):
            line = content.strip().split("\n")[0]
            return json.loads(line)
    raise AssertionError("no results.jsonl written")


class TestHappyPath:
    async def test_dispatches_all_tasks(self) -> None:
        subagent = _make_mock_subagent({"messages": [AIMessage("result")]})
        summary = await execute_swarm(
            _build_options(subagent_graphs={"general-purpose": subagent})
        )
        assert summary.total == 2
        assert summary.completed == 2
        assert summary.failed == 0
        assert summary.failed_tasks == []
        assert summary.results_dir.startswith("/swarm_runs/")
        # summary.results exposes per-task outputs for in-memory aggregation.
        assert len(summary.results) == 2
        assert {r.id for r in summary.results} == {"t1", "t2"}
        assert all(r.status == "completed" and r.result == "result" for r in summary.results)
        assert subagent.ainvoke.call_count == 2

    async def test_passes_description_as_human_message(self) -> None:
        subagent = _make_mock_subagent()
        await execute_swarm(
            _build_options(
                tasks=[SwarmTaskSpec(id="t1", description="Do the thing")],
                subagent_graphs={"general-purpose": subagent},
            )
        )
        state = subagent.ainvoke.call_args_list[0].args[0]
        messages = state["messages"]
        assert len(messages) == 1
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "Do the thing"

    async def test_routes_to_correct_subagent(self) -> None:
        general = _make_mock_subagent()
        analyst = _make_mock_subagent()
        await execute_swarm(
            _build_options(
                tasks=[
                    SwarmTaskSpec(id="t1", description="general"),
                    SwarmTaskSpec(id="t2", description="analyst", subagent_type="analyst"),
                ],
                subagent_graphs={"general-purpose": general, "analyst": analyst},
            )
        )
        assert general.ainvoke.call_count == 1
        assert analyst.ainvoke.call_count == 1


class TestResultExtraction:
    async def test_string_content(self) -> None:
        backend = _MockBackend()
        await execute_swarm(
            _build_options(
                tasks=[SwarmTaskSpec(id="t1", description="task")],
                subagent_graphs={
                    "general-purpose": _make_mock_subagent(
                        {"messages": [AIMessage("extracted text")]}
                    )
                },
                backend=backend,
            )
        )
        assert _read_first_result(backend)["result"] == "extracted text"

    async def test_structured_response_as_json(self) -> None:
        backend = _MockBackend()
        await execute_swarm(
            _build_options(
                tasks=[SwarmTaskSpec(id="t1", description="task")],
                subagent_graphs={
                    "general-purpose": _make_mock_subagent(
                        {
                            "structured_response": {"category": "bug"},
                            "messages": [AIMessage("ignored")],
                        }
                    )
                },
                backend=backend,
            )
        )
        payload = _read_first_result(backend)
        assert json.loads(payload["result"]) == {"category": "bug"}

    async def test_array_content_blocks(self) -> None:
        backend = _MockBackend()
        msg = AIMessage(
            content=[
                {"type": "text", "text": "hello"},
                {"type": "text", "text": "world"},
            ]
        )
        await execute_swarm(
            _build_options(
                tasks=[SwarmTaskSpec(id="t1", description="task")],
                subagent_graphs={"general-purpose": _make_mock_subagent({"messages": [msg]})},
                backend=backend,
            )
        )
        assert _read_first_result(backend)["result"] == "hello\nworld"

    async def test_filters_thinking_and_tool_use(self) -> None:
        backend = _MockBackend()
        msg = AIMessage(
            content=[
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "answer"},
                {"type": "tool_use", "id": "x", "name": "t", "input": {}},
            ]
        )
        await execute_swarm(
            _build_options(
                tasks=[SwarmTaskSpec(id="t1", description="task")],
                subagent_graphs={"general-purpose": _make_mock_subagent({"messages": [msg]})},
                backend=backend,
            )
        )
        assert _read_first_result(backend)["result"] == "answer"

    async def test_empty_messages_fallback(self) -> None:
        backend = _MockBackend()
        await execute_swarm(
            _build_options(
                tasks=[SwarmTaskSpec(id="t1", description="task")],
                subagent_graphs={"general-purpose": _make_mock_subagent({"messages": []})},
                backend=backend,
            )
        )
        assert _read_first_result(backend)["result"] == "Task completed"


class TestErrorHandling:
    async def test_captures_subagent_errors(self) -> None:
        summary = await execute_swarm(
            _build_options(
                tasks=[SwarmTaskSpec(id="t1", description="will fail")],
                subagent_graphs={"general-purpose": _make_mock_subagent(RuntimeError("boom"))},
            )
        )
        assert summary.failed == 1
        assert summary.completed == 0
        assert len(summary.failed_tasks) == 1
        assert summary.failed_tasks[0].id == "t1"
        assert summary.failed_tasks[0].error == "boom"

    async def test_raises_on_unknown_subagent(self) -> None:
        with pytest.raises(ValueError, match="Unknown subagent type.*nonexistent"):
            await execute_swarm(
                _build_options(
                    tasks=[
                        SwarmTaskSpec(id="t1", description="task", subagent_type="nonexistent")
                    ],
                    subagent_graphs={"general-purpose": _make_mock_subagent()},
                )
            )

    async def test_reports_available_types(self) -> None:
        with pytest.raises(ValueError, match="Available: general-purpose, analyst"):
            await execute_swarm(
                _build_options(
                    tasks=[
                        SwarmTaskSpec(id="t1", description="task", subagent_type="missing")
                    ],
                    subagent_graphs={
                        "general-purpose": _make_mock_subagent(),
                        "analyst": _make_mock_subagent(),
                    },
                )
            )

    async def test_mixed_success_and_failure(self) -> None:
        summary = await execute_swarm(
            _build_options(
                tasks=[
                    SwarmTaskSpec(id="t1", description="succeeds"),
                    SwarmTaskSpec(id="t2", description="fails", subagent_type="flaky"),
                ],
                subagent_graphs={
                    "general-purpose": _make_mock_subagent({"messages": [AIMessage("ok")]}),
                    "flaky": _make_mock_subagent(RuntimeError("failed")),
                },
            )
        )
        assert summary.total == 2
        assert summary.completed == 1
        assert summary.failed == 1


class TestFileOutput:
    async def test_writes_results_jsonl(self) -> None:
        backend = _MockBackend()
        summary = await execute_swarm(
            _build_options(
                tasks=[SwarmTaskSpec(id="t1", description="task")],
                backend=backend,
            )
        )
        results_path = f"{summary.results_dir}/results.jsonl"
        assert results_path in backend.written_files
        lines = [ln for ln in backend.written_files[results_path].split("\n") if ln.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["id"] == "t1"

    async def test_writes_tasks_jsonl_when_synthesized(self) -> None:
        backend = _MockBackend()
        synthesized = '{"id":"t1","description":"task"}\n'
        summary = await execute_swarm(
            _build_options(
                tasks=[SwarmTaskSpec(id="t1", description="task")],
                backend=backend,
                synthesized_tasks_jsonl=synthesized,
            )
        )
        assert backend.written_files[f"{summary.results_dir}/tasks.jsonl"] == synthesized

    async def test_no_tasks_jsonl_when_not_provided(self) -> None:
        backend = _MockBackend()
        summary = await execute_swarm(
            _build_options(
                tasks=[SwarmTaskSpec(id="t1", description="task")],
                backend=backend,
            )
        )
        assert f"{summary.results_dir}/tasks.jsonl" not in backend.written_files


class TestStateFiltering:
    async def test_excludes_reserved_keys(self) -> None:
        subagent = _make_mock_subagent()
        await execute_swarm(
            _build_options(
                tasks=[SwarmTaskSpec(id="t1", description="task")],
                subagent_graphs={"general-purpose": subagent},
                current_state={
                    "messages": ["should be excluded"],
                    "todos": ["should be excluded"],
                    "structured_response": "should be excluded",
                    "skills_metadata": "should be excluded",
                    "memory_contents": "should be excluded",
                    "customField": "should be kept",
                },
            )
        )
        state = subagent.ainvoke.call_args_list[0].args[0]
        assert "todos" not in state
        assert "structured_response" not in state
        assert "skills_metadata" not in state
        assert "memory_contents" not in state
        assert state["customField"] == "should be kept"
        assert len(state["messages"]) == 1  # replaced with the task HumanMessage


class TestCancellation:
    async def test_pending_tasks_short_circuit_when_event_set(self) -> None:
        """Tasks that start after cancel_event fires are failed with ``Aborted``."""
        event = asyncio.Event()
        event.set()
        summary = await execute_swarm(
            _build_options(
                tasks=[
                    SwarmTaskSpec(id="t1", description="a"),
                    SwarmTaskSpec(id="t2", description="b"),
                ],
                cancel_event=event,
            )
        )
        assert summary.failed == 2
        assert summary.completed == 0
        assert all(r.error == "Aborted" for r in summary.results)


class TestConcurrency:
    async def test_respects_concurrency_limit(self) -> None:
        state = {"max": 0, "current": 0}
        lock = asyncio.Lock()

        async def _ainvoke(_state: dict, _config: Any = None) -> dict:
            async with lock:
                state["current"] += 1
                if state["current"] > state["max"]:
                    state["max"] = state["current"]
            await asyncio.sleep(0.05)
            async with lock:
                state["current"] -= 1
            return {"messages": [AIMessage("done")]}

        subagent = AsyncMock(spec=Runnable)
        subagent.ainvoke.side_effect = _ainvoke

        tasks = [SwarmTaskSpec(id=f"t{i}", description=f"task {i}") for i in range(10)]
        await execute_swarm(
            _build_options(
                tasks=tasks,
                subagent_graphs={"general-purpose": subagent},
                concurrency=3,
            )
        )
        assert state["max"] <= 3
        assert subagent.ainvoke.call_count == 10

    async def test_clamps_to_max_concurrency(self) -> None:
        summary = await execute_swarm(
            _build_options(
                tasks=[SwarmTaskSpec(id="t1", description="task")],
                subagent_graphs={"general-purpose": _make_mock_subagent()},
                concurrency=999,
            )
        )
        assert summary.completed == 1
