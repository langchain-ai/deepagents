"""End-to-end(ish) tests for the swarm() REPL global.

Drives ``_ThreadREPL`` directly, injecting a ``SwarmBinding`` with mock
subagents + a stub backend, and verifies the JS-side ``swarm({...})``
call runs the executor and returns a parseable JSON summary.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from deepagents.backends.protocol import BackendProtocol, GlobResult, WriteResult
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from quickjs_rs import Runtime

from deepagents_repl._repl import SwarmBinding, _ThreadREPL


class _StubBackend(BackendProtocol):
    def __init__(self, glob_files: dict[str, list[dict[str, str]]] | None = None) -> None:
        self._glob_files = glob_files or {}
        self.written_files: dict[str, str] = {}

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        self.written_files[file_path] = content
        return WriteResult(path=file_path)

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        if pattern in self._glob_files:
            return GlobResult(matches=[{"path": f["path"]} for f in self._glob_files[pattern]])
        return GlobResult(matches=[])

    # Required abstracts; not exercised.
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


def _make_subagent(response_text: str = "done") -> Runnable:
    mock = AsyncMock(spec=Runnable)

    async def _ainvoke(state: dict, config: Any = None) -> dict:
        return {"messages": [AIMessage(response_text)]}

    mock.ainvoke.side_effect = _ainvoke
    return mock


@pytest.fixture
def runtime() -> Runtime:
    rt = Runtime()
    try:
        yield rt
    finally:
        rt.close()


async def test_swarm_summary_exposes_results_inline(runtime: Runtime) -> None:
    """JS-facing summary should include ``results`` so the model can aggregate
    without a second backend round-trip through ``results.jsonl``."""
    backend = _StubBackend()
    subagent = _make_subagent("classification")
    binding = SwarmBinding(backend=backend, subagent_graphs={"general-purpose": subagent})
    repl = _ThreadREPL(
        runtime, timeout=10.0, capture_console=True, swarm_binding=binding
    )

    outcome = await repl.eval_async(
        """
        const summary = await swarm({
            tasks: [
                { id: "a", description: "A." },
                { id: "b", description: "B." },
            ],
        });
        JSON.stringify(summary.results.map(r => ({ id: r.id, result: r.result })))
        """
    )
    assert outcome.error_type is None, outcome.error_message
    parsed = json.loads(outcome.result)
    assert {r["id"] for r in parsed} == {"a", "b"}
    assert all(r["result"] == "classification" for r in parsed)


async def test_swarm_direct_tasks_form(runtime: Runtime) -> None:
    backend = _StubBackend()
    subagent = _make_subagent("hello")
    binding = SwarmBinding(backend=backend, subagent_graphs={"general-purpose": subagent})
    repl = _ThreadREPL(
        runtime, timeout=10.0, capture_console=True, swarm_binding=binding
    )

    outcome = await repl.eval_async(
        """
        const summary = await swarm({
            tasks: [
                { id: "q1", description: "Say hi." },
                { id: "q2", description: "Say bye." },
            ],
        });
        JSON.stringify(summary)
        """
    )
    assert outcome.error_type is None, outcome.error_message
    parsed = json.loads(outcome.result)
    assert parsed["total"] == 2
    assert parsed["completed"] == 2
    assert parsed["failed"] == 0
    assert parsed["failedTasks"] == []
    assert parsed["resultsDir"].startswith("/swarm_runs/")

    # results.jsonl was written to the backend
    results_path = f"{parsed['resultsDir']}/results.jsonl"
    lines = [ln for ln in backend.written_files[results_path].split("\n") if ln.strip()]
    assert len(lines) == 2
    parsed_results = [json.loads(ln) for ln in lines]
    ids = {r["id"] for r in parsed_results}
    assert ids == {"q1", "q2"}
    assert all(r["status"] == "completed" and r["result"] == "hello" for r in parsed_results)
    # tasks.jsonl should NOT be written for the direct form
    assert f"{parsed['resultsDir']}/tasks.jsonl" not in backend.written_files


async def test_swarm_virtual_table_form(runtime: Runtime) -> None:
    backend = _StubBackend(
        glob_files={"feedback/*.txt": [{"path": "feedback/a.txt"}, {"path": "feedback/b.txt"}]}
    )
    subagent = _make_subagent("classified")
    binding = SwarmBinding(backend=backend, subagent_graphs={"general-purpose": subagent})
    repl = _ThreadREPL(
        runtime, timeout=10.0, capture_console=True, swarm_binding=binding
    )

    outcome = await repl.eval_async(
        """
        const summary = await swarm({
            glob: "feedback/*.txt",
            instruction: "Classify",
        });
        JSON.stringify(summary)
        """
    )
    assert outcome.error_type is None, outcome.error_message
    parsed = json.loads(outcome.result)
    assert parsed["total"] == 2
    assert parsed["completed"] == 2

    # tasks.jsonl should be written for the virtual-table form
    tasks_path = f"{parsed['resultsDir']}/tasks.jsonl"
    assert tasks_path in backend.written_files
    task_lines = [ln for ln in backend.written_files[tasks_path].split("\n") if ln.strip()]
    assert len(task_lines) == 2
    task_ids = {json.loads(ln)["id"] for ln in task_lines}
    assert task_ids == {"a.txt", "b.txt"}


async def test_swarm_routes_to_named_subagent(runtime: Runtime) -> None:
    backend = _StubBackend()
    general = _make_subagent("general-output")
    analyst = _make_subagent("analyst-output")
    binding = SwarmBinding(
        backend=backend,
        subagent_graphs={"general-purpose": general, "analyst": analyst},
    )
    repl = _ThreadREPL(
        runtime, timeout=10.0, capture_console=True, swarm_binding=binding
    )

    outcome = await repl.eval_async(
        """
        const summary = await swarm({
            tasks: [
                { id: "g", description: "general task" },
                { id: "a", description: "analyst task", subagentType: "analyst" },
            ],
        });
        JSON.stringify(summary)
        """
    )
    assert outcome.error_type is None, outcome.error_message
    parsed = json.loads(outcome.result)
    assert parsed["completed"] == 2
    assert general.ainvoke.call_count == 1
    assert analyst.ainvoke.call_count == 1


async def test_swarm_failed_tasks_populate_failed_tasks(runtime: Runtime) -> None:
    backend = _StubBackend()
    subagent = AsyncMock(spec=Runnable)

    async def _ainvoke(_state: dict, _config: Any = None) -> dict:
        raise RuntimeError("kaboom")

    subagent.ainvoke.side_effect = _ainvoke

    binding = SwarmBinding(backend=backend, subagent_graphs={"general-purpose": subagent})
    repl = _ThreadREPL(
        runtime, timeout=10.0, capture_console=True, swarm_binding=binding
    )

    outcome = await repl.eval_async(
        """
        const summary = await swarm({
            tasks: [{ id: "t1", description: "doomed" }],
        });
        JSON.stringify(summary)
        """
    )
    parsed = json.loads(outcome.result)
    assert parsed["failed"] == 1
    assert parsed["completed"] == 0
    assert parsed["failedTasks"] == [{"id": "t1", "error": "kaboom"}]


async def test_swarm_rejects_when_no_tasks_or_instruction(runtime: Runtime) -> None:
    backend = _StubBackend()
    binding = SwarmBinding(
        backend=backend, subagent_graphs={"general-purpose": _make_subagent()}
    )
    repl = _ThreadREPL(
        runtime, timeout=10.0, capture_console=True, swarm_binding=binding
    )
    outcome = await repl.eval_async(
        """
        try {
            await swarm({});
            "unexpected"
        } catch (e) {
            e.message
        }
        """
    )
    assert outcome.error_type is None
    assert "requires either" in (outcome.result or "")


async def test_swarm_rejects_unknown_subagent(runtime: Runtime) -> None:
    backend = _StubBackend()
    binding = SwarmBinding(
        backend=backend, subagent_graphs={"general-purpose": _make_subagent()}
    )
    repl = _ThreadREPL(
        runtime, timeout=10.0, capture_console=True, swarm_binding=binding
    )
    outcome = await repl.eval_async(
        """
        try {
            await swarm({
                tasks: [{ id: "t1", description: "task", subagentType: "nonexistent" }],
            });
            "unexpected"
        } catch (e) {
            e.message
        }
        """
    )
    assert outcome.error_type is None
    assert "Unknown subagent type" in (outcome.result or "")


async def test_swarm_not_registered_without_binding(runtime: Runtime) -> None:
    """When swarm_binding is None, the global is absent."""
    repl = _ThreadREPL(runtime, timeout=10.0, capture_console=True)
    outcome = await repl.eval_async("typeof swarm")
    # typeof on an unregistered global returns "undefined" in JS.
    assert outcome.result == "undefined"


async def test_swarm_cancelled_on_eval_timeout(runtime: Runtime) -> None:
    """When the outer eval_async times out, in-flight swarm subagents get aborted."""
    cancelled_seen: list[bool] = []

    subagent = AsyncMock(spec=Runnable)

    async def _slow(_state: dict, _config: Any = None) -> dict:
        try:
            await asyncio.sleep(5.0)
            return {"messages": [AIMessage("ok")]}
        except asyncio.CancelledError:
            cancelled_seen.append(True)
            raise

    subagent.ainvoke.side_effect = _slow

    backend = _StubBackend()
    binding = SwarmBinding(backend=backend, subagent_graphs={"general-purpose": subagent})
    repl = _ThreadREPL(
        runtime, timeout=0.3, capture_console=True, swarm_binding=binding
    )

    outcome = await repl.eval_async(
        """
        await swarm({ tasks: [{ id: "slow", description: "never finishes" }] });
        "unreachable"
        """
    )
    assert outcome.error_type == "Timeout"
    # The ainvoke was cancelled when eval hit its deadline.
    assert cancelled_seen == [True]
