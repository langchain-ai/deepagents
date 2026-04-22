"""End-to-end tests for the swarm.create / swarm.execute REPL namespace.

Drives ``_ThreadREPL`` directly with a ``SwarmBinding``, verifies the
JS-side namespace is wired correctly, and checks that pending-writes
buffer behaviour makes ``create`` visible to ``execute`` within one
eval.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from deepagents.backends.protocol import (
    BackendProtocol,
    GlobResult,
    ReadResult,
    WriteResult,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable
from quickjs_rs import Runtime

from deepagents_repl._repl import SwarmBinding, _ThreadREPL


class _StubBackend(BackendProtocol):
    """Backend with an in-memory file store — supports read/write/glob."""

    def __init__(
        self,
        files: dict[str, str] | None = None,
        glob_files: dict[str, list[str]] | None = None,
    ) -> None:
        self._files: dict[str, str] = dict(files or {})
        self._glob_files = glob_files or {}
        self.write_calls: list[tuple[str, str]] = []

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        self._files[file_path] = content
        self.write_calls.append((file_path, content))
        return WriteResult(path=file_path)

    async def aread(
        self, file_path: str, offset: int = 0, limit: int = 2000
    ) -> ReadResult:
        if file_path not in self._files:
            return ReadResult(error="file_not_found")
        return ReadResult(file_data={"content": self._files[file_path], "encoding": "utf-8"})

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        matches = self._glob_files.get(pattern, [])
        return GlobResult(matches=[{"path": p} for p in matches])

    # Abstract methods not exercised here.
    def ls(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def read(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def write(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def edit(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def grep(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def glob(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def upload_files(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def download_files(self, *a: Any, **k: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def _mock_subagent(response_text: str = "done") -> Runnable:
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


async def test_namespace_registered(runtime: Runtime) -> None:
    """`swarm.create` and `swarm.execute` exist on the JS global."""
    backend = _StubBackend()
    binding = SwarmBinding(
        backend=backend,
        subagent_graphs={"general-purpose": _mock_subagent()},
    )
    repl = _ThreadREPL(runtime, timeout=10.0, capture_console=True, swarm_binding=binding)

    out = await repl.eval_async(
        'typeof swarm.create + "/" + typeof swarm.execute'
    )
    assert out.result == "function/function"


async def test_namespace_absent_without_binding(runtime: Runtime) -> None:
    """When no binding is configured, the global is not injected."""
    repl = _ThreadREPL(runtime, timeout=10.0, capture_console=True)
    out = await repl.eval_async("typeof swarm")
    assert out.result == "undefined"


async def test_create_then_execute_round_trip(runtime: Runtime) -> None:
    """`swarm.create` stages the table; `swarm.execute` reads it from the
    pending buffer in the same eval; results stream into the table."""
    backend = _StubBackend()
    subagent = _mock_subagent("classified")
    binding = SwarmBinding(
        backend=backend, subagent_graphs={"general-purpose": subagent}
    )
    repl = _ThreadREPL(runtime, timeout=10.0, capture_console=True, swarm_binding=binding)

    out = await repl.eval_async(
        """
        await swarm.create("/t.jsonl", {
            tasks: [
                { id: "a", topic: "x" },
                { id: "b", topic: "y" },
            ],
        });
        const summary = JSON.parse(await swarm.execute("/t.jsonl", {
            instruction: "Classify {topic}",
            column: "label",
        }));
        JSON.stringify({
            completed: summary.completed,
            total: summary.total,
            column: summary.column,
        })
        """
    )
    assert out.error_type is None, out.error_message
    parsed = json.loads(out.result)
    assert parsed == {"completed": 2, "total": 2, "column": "label"}
    assert subagent.ainvoke.call_count == 2


async def test_pending_writes_flushed_after_eval(runtime: Runtime) -> None:
    """After eval returns, the updated table is drainable for the middleware to flush."""
    backend = _StubBackend()
    binding = SwarmBinding(
        backend=backend,
        subagent_graphs={"general-purpose": _mock_subagent("answer")},
    )
    repl = _ThreadREPL(runtime, timeout=10.0, capture_console=True, swarm_binding=binding)

    await repl.eval_async(
        """
        await swarm.create("/t.jsonl", { tasks: [{ id: "a" }] });
        await swarm.execute("/t.jsonl", { instruction: "do {id}" });
        """
    )
    drained = dict(repl._drain_pending_writes())
    assert "/t.jsonl" in drained
    rows = [json.loads(line) for line in drained["/t.jsonl"].strip().split("\n")]
    assert rows == [{"id": "a", "result": "answer"}]


async def test_filter_skips_matching_rows(runtime: Runtime) -> None:
    """`filter` clauses pass through unchanged rows to the final table."""
    backend = _StubBackend()
    subagent = _mock_subagent("processed")
    binding = SwarmBinding(
        backend=backend, subagent_graphs={"general-purpose": subagent}
    )
    repl = _ThreadREPL(runtime, timeout=10.0, capture_console=True, swarm_binding=binding)

    out = await repl.eval_async(
        """
        await swarm.create("/t.jsonl", {
            tasks: [
                { id: "a", status: "done", result: "existing" },
                { id: "b", status: "pending" },
            ],
        });
        const summary = JSON.parse(await swarm.execute("/t.jsonl", {
            instruction: "do {id}",
            filter: { column: "status", equals: "pending" },
        }));
        JSON.stringify({ completed: summary.completed, skipped: summary.skipped });
        """
    )
    parsed = json.loads(out.result)
    assert parsed == {"completed": 1, "skipped": 1}
    assert subagent.ainvoke.call_count == 1


async def test_response_schema_flows_to_factory(runtime: Runtime) -> None:
    """A `responseSchema` on the JS side is passed to the factory and
    the result is stored as parsed JSON in the column."""
    factory_calls: list[Any] = []
    variant = _mock_subagent('{"label":"bug"}')
    default = _mock_subagent()

    def factory(schema: Any) -> Any:
        factory_calls.append(schema)
        return variant

    backend = _StubBackend()
    binding = SwarmBinding(
        backend=backend,
        subagent_graphs={"general-purpose": default},
        subagent_factories={"general-purpose": factory},
    )
    repl = _ThreadREPL(runtime, timeout=10.0, capture_console=True, swarm_binding=binding)

    await repl.eval_async(
        """
        await swarm.create("/t.jsonl", { tasks: [{ id: "a" }] });
        await swarm.execute("/t.jsonl", {
            instruction: "Classify {id}",
            responseSchema: {
                type: "object",
                properties: { label: { type: "string" } },
                required: ["label"],
            },
        });
        """
    )
    assert len(factory_calls) == 1
    assert factory_calls[0] == {
        "type": "object",
        "properties": {"label": {"type": "string"}},
        "required": ["label"],
    }
    # Parsed JSON, not a string.
    drained = dict(repl._drain_pending_writes())
    rows = [json.loads(line) for line in drained["/t.jsonl"].strip().split("\n")]
    assert rows == [{"id": "a", "label": "bug"}]
    assert default.ainvoke.call_count == 0


async def test_swarm_execute_logs_summary_and_failures_to_stdout(
    runtime: Runtime,
) -> None:
    """Summary + first-3 failure details should be emitted to the REPL stdout,
    so the agent can see *why* dispatches failed instead of just a count."""
    failing = AsyncMock(spec=Runnable)
    failing.ainvoke.side_effect = RuntimeError("rate limit: 429")
    backend = _StubBackend()
    binding = SwarmBinding(
        backend=backend, subagent_graphs={"general-purpose": failing}
    )
    repl = _ThreadREPL(runtime, timeout=10.0, capture_console=True, swarm_binding=binding)

    outcome = await repl.eval_async(
        """
        await swarm.create("/t.jsonl", {
            tasks: [
                { id: "a" }, { id: "b" }, { id: "c" },
                { id: "d" }, { id: "e" },
            ],
        });
        const summary = JSON.parse(await swarm.execute("/t.jsonl", {
            instruction: "do {id}",
        }));
        summary.failed
        """
    )
    assert outcome.result == "5"
    assert outcome.stdout is not None
    # Summary line
    assert "[swarm.execute] 0 completed, 5 failed, 0 skipped" in outcome.stdout
    # First three failures listed
    assert '[swarm.execute] failure id="a"' in outcome.stdout
    assert '[swarm.execute] failure id="b"' in outcome.stdout
    assert '[swarm.execute] failure id="c"' in outcome.stdout
    assert "rate limit: 429" in outcome.stdout
    # Overflow tallied
    assert "[swarm.execute] ... and 2 more failures" in outcome.stdout


async def test_swarm_create_logs_to_stdout(runtime: Runtime) -> None:
    backend = _StubBackend()
    binding = SwarmBinding(
        backend=backend, subagent_graphs={"general-purpose": _mock_subagent()}
    )
    repl = _ThreadREPL(runtime, timeout=10.0, capture_console=True, swarm_binding=binding)
    outcome = await repl.eval_async(
        """
        await swarm.create("/t.jsonl", { tasks: [{ id: "a" }] });
        "ok"
        """
    )
    assert outcome.stdout is not None
    assert "[swarm.create] Table written to /t.jsonl." in outcome.stdout


async def test_swarm_execute_authoritative_hint_always_logged(runtime: Runtime) -> None:
    """The 'Results are authoritative' breadcrumb fires on every execute,
    not just on failure — so the model never re-dispatches to verify."""
    backend = _StubBackend()
    binding = SwarmBinding(
        backend=backend, subagent_graphs={"general-purpose": _mock_subagent("ok")}
    )
    repl = _ThreadREPL(runtime, timeout=10.0, capture_console=True, swarm_binding=binding)
    outcome = await repl.eval_async(
        """
        await swarm.create("/t.jsonl", { tasks: [{ id: "a" }] });
        await swarm.execute("/t.jsonl", { instruction: "do {id}" });
        "ok"
        """
    )
    assert outcome.stdout is not None
    assert "Results are authoritative" in outcome.stdout


async def test_swarm_execute_summary_wire_omits_results(runtime: Runtime) -> None:
    """The summary returned to the JS side should not include per-row
    results — they're on the table already, and duplicating them wastes
    context tokens."""
    backend = _StubBackend()
    binding = SwarmBinding(
        backend=backend, subagent_graphs={"general-purpose": _mock_subagent("ok")}
    )
    repl = _ThreadREPL(runtime, timeout=10.0, capture_console=True, swarm_binding=binding)
    outcome = await repl.eval_async(
        """
        await swarm.create("/t.jsonl", { tasks: [{ id: "a" }, { id: "b" }] });
        const summary = JSON.parse(await swarm.execute("/t.jsonl", { instruction: "do {id}" }));
        JSON.stringify({
            hasResults: "results" in summary,
            total: summary.total,
            completed: summary.completed,
        })
        """
    )
    parsed = json.loads(outcome.result)
    assert parsed == {"hasResults": False, "total": 2, "completed": 2}


async def test_swarm_execute_context_flows_to_subagent(runtime: Runtime) -> None:
    """`context` on swarm.execute is prepended to every subagent prompt."""
    subagent = _mock_subagent()
    backend = _StubBackend()
    binding = SwarmBinding(
        backend=backend, subagent_graphs={"general-purpose": subagent}
    )
    repl = _ThreadREPL(runtime, timeout=10.0, capture_console=True, swarm_binding=binding)
    await repl.eval_async(
        """
        await swarm.create("/t.jsonl", { tasks: [{ id: "a" }, { id: "b" }] });
        await swarm.execute("/t.jsonl", {
            instruction: "handle {id}",
            context: "These are customer reviews. Classify sentiment.",
        });
        """
    )
    assert subagent.ainvoke.call_count == 2
    first_content = subagent.ainvoke.call_args_list[0].args[0]["messages"][0].content
    assert first_content.startswith("These are customer reviews. Classify sentiment.")
    assert "---" in first_content
    assert first_content.endswith("handle a")


async def test_swarm_execute_rejects_non_string_context(runtime: Runtime) -> None:
    backend = _StubBackend()
    binding = SwarmBinding(
        backend=backend, subagent_graphs={"general-purpose": _mock_subagent()}
    )
    repl = _ThreadREPL(runtime, timeout=10.0, capture_console=True, swarm_binding=binding)
    out = await repl.eval_async(
        """
        await swarm.create("/t.jsonl", { tasks: [{ id: "a" }] });
        try {
            await swarm.execute("/t.jsonl", {
                instruction: "do {id}",
                context: 42,
            });
            "unexpected"
        } catch (e) {
            e.message
        }
        """
    )
    assert "context" in (out.result or "")


async def test_swarm_execute_batch_size_dispatches_in_batches(runtime: Runtime) -> None:
    """`batchSize` from JS groups rows into combined subagent calls."""
    subagent = AsyncMock(spec=Runnable)
    # The mock returns a wrapped batch shape regardless of input size; we
    # only need to verify the call count drops from 4 → 2.
    async def _ainvoke(state: dict, config: Any = None) -> dict:
        return {
            "messages": [
                AIMessage(
                    json.dumps(
                        {
                            "results": [
                                {"id": "a", "label": "x"},
                                {"id": "b", "label": "y"},
                            ]
                        }
                    )
                )
            ]
        }

    subagent.ainvoke.side_effect = _ainvoke
    backend = _StubBackend()
    binding = SwarmBinding(
        backend=backend, subagent_graphs={"general-purpose": subagent}
    )
    repl = _ThreadREPL(runtime, timeout=10.0, capture_console=True, swarm_binding=binding)
    await repl.eval_async(
        """
        await swarm.create("/t.jsonl", {
            tasks: [{ id: "a" }, { id: "b" }, { id: "c" }, { id: "d" }],
        });
        await swarm.execute("/t.jsonl", {
            instruction: "do {id}",
            batchSize: 2,
            responseSchema: {
                type: "object",
                properties: { label: { type: "string" } },
                required: ["label"],
            },
        });
        """
    )
    # 4 rows / batchSize=2 = 2 subagent invocations.
    assert subagent.ainvoke.call_count == 2


async def test_swarm_execute_batch_size_without_schema_rejected(runtime: Runtime) -> None:
    backend = _StubBackend()
    binding = SwarmBinding(
        backend=backend, subagent_graphs={"general-purpose": _mock_subagent()}
    )
    repl = _ThreadREPL(runtime, timeout=10.0, capture_console=True, swarm_binding=binding)
    out = await repl.eval_async(
        """
        await swarm.create("/t.jsonl", { tasks: [{ id: "a" }] });
        try {
            await swarm.execute("/t.jsonl", {
                instruction: "do {id}",
                batchSize: 2,
            });
            "unexpected"
        } catch (e) {
            e.message
        }
        """
    )
    assert "batch_size requires response_schema" in (out.result or "")


async def test_swarm_execute_rejects_missing_instruction(runtime: Runtime) -> None:
    backend = _StubBackend()
    binding = SwarmBinding(
        backend=backend, subagent_graphs={"general-purpose": _mock_subagent()}
    )
    repl = _ThreadREPL(runtime, timeout=10.0, capture_console=True, swarm_binding=binding)
    out = await repl.eval_async(
        """
        await swarm.create("/t.jsonl", { tasks: [{ id: "a" }] });
        try {
            await swarm.execute("/t.jsonl", {});
            "unexpected"
        } catch (e) {
            e.message
        }
        """
    )
    assert "instruction" in (out.result or "")
