"""Unit tests for the swarm interpreter extension.

Exercises the extension through a real ``_ThreadREPL`` with an in-memory
backend: the bundled swarm scripts import as ``import { create, rows } from
"swarm"`` and call the top-level ``__swarm*`` host functions (glob/read/
write/edit backed by ``ctx.backend``, dispatch backed by a stub) directly,
and a ``create`` → ``rows`` round-trip persists and reads a table back.
Model calls are stubbed; file ops use the fake backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from quickjs_rs import Runtime, ThreadWorker

from langchain_quickjs import InterpreterExtension, SwarmExtension, swarm
from langchain_quickjs._extensions import (
    has_on_setup,
    validate_extension_exports,
    validate_extension_hooks,
)
from langchain_quickjs._repl import _ThreadREPL
from langchain_quickjs._swarm._extension import SwarmExtension as _SwarmExtensionClass

if TYPE_CHECKING:
    from collections.abc import Iterator

from deepagents.backends.protocol import (
    BackendProtocol,
    GlobResult,
    ReadResult,
    WriteResult,
)


class _MemoryBackend(BackendProtocol):
    """Minimal in-memory backend for swarm table persistence."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        import fnmatch

        matches = [
            {"path": p} for p in sorted(self.files) if fnmatch.fnmatch(p, pattern)
        ]
        return GlobResult(matches=matches)

    async def aread(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        if file_path not in self.files:
            return ReadResult(error="file_not_found")
        return ReadResult(
            file_data={"content": self.files[file_path], "encoding": "utf-8"}
        )

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        self.files[file_path] = content
        return WriteResult(path=file_path)


@pytest.fixture
def worker() -> Iterator[ThreadWorker]:
    w = ThreadWorker()
    try:
        yield w
    finally:
        w.close()


@pytest.fixture
def runtime(worker: ThreadWorker) -> Iterator[Runtime]:
    async def _make() -> Runtime:
        return Runtime()

    rt = worker.run_sync(_make())
    try:
        yield rt
    finally:

        async def _close() -> None:
            rt.close()

        worker.run_sync(_close())


def _make_repl(
    worker: ThreadWorker,
    runtime: Runtime,
    extensions: list[InterpreterExtension],
    backend: BackendProtocol | None = None,
) -> _ThreadREPL:
    return _ThreadREPL(
        worker,
        runtime,
        timeout=10.0,
        capture_console=True,
        max_stdout_chars=8000,
        extensions=extensions,
        backend=backend,
    )


def _stub_extension(calls: list[tuple]) -> SwarmExtension:
    """A SwarmExtension whose dispatch records its args and echoes back."""

    async def _dispatch(
        description: str,
        subagent_type: str | None = None,
        response_schema: dict[str, Any] | None = None,
        mode: str | None = None,
    ) -> str:
        calls.append((description, subagent_type, response_schema, mode))
        return f"dispatched: {description}"

    return _SwarmExtensionClass(dispatch=_dispatch)


# ---------------------------------------------------------------------------
# Module + host wiring
# ---------------------------------------------------------------------------


def test_scripts_expose_table_api(worker: ThreadWorker, runtime: Runtime) -> None:
    repl = _make_repl(worker, runtime, [_stub_extension([])], _MemoryBackend())
    outcome = repl.eval_sync(
        'const m = await import("swarm");'
        " [typeof m.create, typeof m.run, typeof m.rows].join(',')"
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "function,function,function"


def test_host_functions_registered(worker: ThreadWorker, runtime: Runtime) -> None:
    # The scripts call top-level __swarm* functions directly — no
    # globalThis.tools namespace. Assert each host symbol is a global fn.
    repl = _make_repl(worker, runtime, [_stub_extension([])], _MemoryBackend())
    outcome = repl.eval_sync(
        "[typeof __swarmTask, typeof __swarmGlob, typeof __swarmReadFile,"
        " typeof __swarmWriteFile, typeof __swarmEditFile].join(',')"
    )
    assert outcome.result == "function,function,function,function,function"


def test_create_then_rows_roundtrip(worker: ThreadWorker, runtime: Runtime) -> None:
    # create({tasks}) persists a table via writeFile; rows() reads it back.
    # Exercises the real scripts + the backend-backed host adapters.
    backend = _MemoryBackend()
    repl = _make_repl(worker, runtime, [_stub_extension([])], backend)
    outcome = repl.eval_sync(
        'const { create, rows } = await import("swarm");'
        " const table = await create({ tasks: ["
        '   { id: "a", text: "first" },'
        '   { id: "b", text: "second" },'
        " ] });"
        " const r = await rows(table.id);"
        " r.length"
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "2"
    # The table was persisted through the backend.
    assert backend.files, "expected the table to be written to the backend"


def test_glob_returns_json_paths(worker: ThreadWorker, runtime: Runtime) -> None:
    backend = _MemoryBackend()
    backend.files["/a.txt"] = "x"
    backend.files["/b.txt"] = "y"
    repl = _make_repl(worker, runtime, [_stub_extension([])], backend)
    # The script JSON.parses glob output; assert our adapter returns valid JSON.
    outcome = repl.eval_sync(
        'const raw = await __swarmGlob({ pattern: "*.txt" });'
        " JSON.parse(raw).map(e => e.path).sort().join(',')"
    )
    assert outcome.result == "/a.txt,/b.txt"


def test_file_op_without_backend_errors(worker: ThreadWorker, runtime: Runtime) -> None:
    # No backend configured: a file-op host call surfaces a catchable error.
    repl = _make_repl(worker, runtime, [_stub_extension([])], backend=None)
    outcome = repl.eval_sync(
        ' let msg = "";'
        ' try { await __swarmGlob({ pattern: "*" }) }'
        ' catch (e) { msg = "errored" }'
        " msg"
    )
    assert outcome.error_type is None
    assert outcome.result == "errored"


# ---------------------------------------------------------------------------
# Extension shape / factory
# ---------------------------------------------------------------------------


def test_system_prompt_is_the_skill_doc() -> None:
    ext = _stub_extension([])
    prompt = ext.system_prompt
    assert prompt is not None
    # Sourced from the upstream SKILL.md body: full sections, not a paraphrase.
    assert "## Flow" in prompt
    assert "## API Reference" in prompt
    assert "responseSchema" in prompt
    # Import specifier adapted to "swarm"; no leftover skill path.
    assert "@/skills" not in prompt


def test_extension_validates_as_a_hook_impl() -> None:
    ext = _stub_extension([])
    assert has_on_setup(ext) is True
    validate_extension_hooks(ext)  # does not raise
    validate_extension_exports(ext)  # does not raise


def test_factory_builds_extension() -> None:
    ext = swarm(default_model="openai:gpt-4o-mini")
    assert isinstance(ext, SwarmExtension)
    assert has_on_setup(ext) is True
    assert ext.system_prompt is not None
