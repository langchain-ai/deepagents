"""Unit tests for baseline Code Interpreter extensions."""

from __future__ import annotations

import asyncio
import base64
import fnmatch
import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from deepagents.backends.protocol import (
    EditResult,
    FileUploadResponse,
    GlobResult,
    LsResult,
    ReadResult,
)
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field
from quickjs_rs import Runtime, ThreadWorker

from langchain_quickjs import (
    CodeInterpreterMiddleware,
    ExtensionContext,
    FilesystemExtension,
    GlobExtension,
    InterpreterExtension,
    LlmExtension,
    SubagentExtension,
)
from langchain_quickjs._baseline import build_baseline_extensions
from langchain_quickjs._repl import _ThreadREPL


class _TaskInput(BaseModel):
    description: str = Field()
    subagent_type: str = Field()


@dataclass
class _FakeRuntime:
    state: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str | None = "outer_eval"
    config: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    store: Any = None
    stream_writer: Any = None
    tools: list[BaseTool] = field(default_factory=list)
    execution_info: Any = None
    server_info: Any = None


class _MemoryBackend:
    def __init__(self) -> None:
        self.files: dict[str, tuple[str, str]] = {}

    async def aread(
        self, file_path: str, offset: int = 0, limit: int = 2000
    ) -> ReadResult:
        if file_path not in self.files:
            return ReadResult(error="file_not_found")
        content, encoding = self.files[file_path]
        sliced = content[offset : offset + limit] if encoding == "utf-8" else content
        return ReadResult(file_data={"content": sliced, "encoding": encoding})

    async def als(self, path: str) -> LsResult:
        prefix = "/" if path == "/" else path.rstrip("/") + "/"
        seen: set[str] = set()
        entries: list[dict[str, Any]] = []
        for file_path in sorted(self.files):
            if not file_path.startswith(prefix):
                continue
            suffix = file_path[len(prefix) :]
            if not suffix:
                continue
            child = suffix.split("/", 1)[0]
            child_path = "/" + child if prefix == "/" else prefix + child
            if child_path in seen:
                continue
            seen.add(child_path)
            entries.append({"path": child_path})
        return LsResult(entries=entries)

    async def aglob(self, pattern: str, path: str = "/") -> GlobResult:
        prefix = "/" if path == "/" else path.rstrip("/") + "/"
        matches = []
        for file_path in sorted(self.files):
            if file_path.startswith(prefix):
                rel_path = file_path[len(prefix) :]
            else:
                rel_path = file_path
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(
                file_path, pattern
            ):
                matches.append({"path": file_path})
        return GlobResult(matches=matches)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        if file_path not in self.files:
            return EditResult(error="file_not_found")
        content, encoding = self.files[file_path]
        if encoding != "utf-8":
            return EditResult(error="not supported for binary data")
        occurrences = content.count(old_string)
        if occurrences == 0:
            return EditResult(error="old string not found")
        if not replace_all and occurrences > 1:
            return EditResult(error="old string is ambiguous")
        if replace_all:
            updated = content.replace(old_string, new_string)
        else:
            updated = content.replace(old_string, new_string, 1)
        self.files[file_path] = (updated, "utf-8")
        return EditResult(path=file_path, occurrences=occurrences)

    async def aupload_files(
        self, files: list[tuple[str, bytes]]
    ) -> list[FileUploadResponse]:
        results: list[FileUploadResponse] = []
        for path, payload in files:
            try:
                text = payload.decode("utf-8")
                self.files[path] = (text, "utf-8")
            except UnicodeDecodeError:
                encoded = base64.standard_b64encode(payload).decode("ascii")
                self.files[path] = (encoded, "base64")
            results.append(FileUploadResponse(path=path, error=None))
        return results


def _task_tool(
    handler: Any,
) -> BaseTool:
    async def _run(description: str, subagent_type: str) -> str:
        result = handler(description, subagent_type)
        if asyncio.iscoroutine(result):
            return await result
        return str(result)

    return StructuredTool.from_function(
        name="task",
        description="Subagent dispatch tool.",
        coroutine=_run,
        args_schema=_TaskInput,
    )


def _make_repl(
    worker: ThreadWorker,
    runtime: Runtime,
    *,
    backend: Any = None,
    subagent_max_in_flight: int = 10,
    llm_max_in_flight: int = 10,
    llm_model: str | None = None,
) -> _ThreadREPL:
    extensions = build_baseline_extensions(
        subagent_max_in_flight=subagent_max_in_flight,
        llm_max_in_flight=llm_max_in_flight,
        subagent_timeout_s=None,
        llm_timeout_s=None,
        llm_model=llm_model,
    )
    return _ThreadREPL(
        worker,
        runtime,
        timeout=5.0,
        capture_console=True,
        max_stdout_chars=4000,
        extensions=extensions,
        backend=backend,
    )


@pytest.fixture
def worker() -> ThreadWorker:
    w = ThreadWorker()
    try:
        yield w
    finally:
        w.close()


@pytest.fixture
def runtime(worker: ThreadWorker) -> Runtime:
    async def _make() -> Runtime:
        return Runtime()

    rt = worker.run_sync(_make())
    try:
        yield rt
    finally:

        async def _close() -> None:
            rt.close()

        worker.run_sync(_close())


def test_middleware_loads_baseline_extensions_by_default() -> None:
    mw = CodeInterpreterMiddleware()
    try:
        baseline_names = [type(ext).__name__ for ext in mw._extensions[:5]]
        assert baseline_names == [
            "SubagentExtension",
            "LlmExtension",
            "FilesystemExtension",
            "GlobExtension",
            "EditFileExtension",
        ]
    finally:
        mw._registry.close()


def test_middleware_rejects_reserved_baseline_name_collision() -> None:
    class _CollidingExtension(InterpreterExtension):
        exported_globals = ("subagent",)

        def on_setup(self, ctx: ExtensionContext) -> None: ...

    with pytest.raises(ValueError, match="reserved CI baseline global names: subagent"):
        CodeInterpreterMiddleware(extensions=[_CollidingExtension()])


def test_middleware_rejects_invalid_baseline_cap_knobs() -> None:
    with pytest.raises(ValueError, match="subagent_max_in_flight"):
        CodeInterpreterMiddleware(subagent_max_in_flight=0)
    with pytest.raises(ValueError, match="llm_max_in_flight"):
        CodeInterpreterMiddleware(llm_max_in_flight=0)


def test_baseline_exports_installed_in_repl(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    repl = _make_repl(worker, runtime, llm_model="test:model")
    outcome = repl.eval_sync(
        "["
        "typeof subagent,"
        "typeof llm,"
        "typeof fs,"
        "typeof glob,"
        "typeof editFile"
        "].join(',')"
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "function,function,object,function,function"


async def test_subagent_rejects_snake_case_input(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    repl = _make_repl(worker, runtime, llm_model="test:model")
    outer_runtime = _FakeRuntime(tools=[_task_tool(lambda _d, _t: "ok")])
    with pytest.raises(RuntimeError, match=r"ERR_INVALID_ARG.*subagent_type"):
        await repl.eval_async(
            "await subagent({ description: 'x', subagent_type: 'worker' })",
            outer_runtime=outer_runtime,
        )


async def test_subagent_structured_output_parses_to_native_value(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    repl = _make_repl(worker, runtime, llm_model="test:model")
    outer_runtime = _FakeRuntime(
        tools=[_task_tool(lambda _d, _t: '{"score": 1}')]
    )
    outcome = await repl.eval_async(
        "const out = await subagent({"
        " description: 'run',"
        " subagentType: 'worker',"
        " responseSchema: { type: 'object' }"
        "});"
        " out.score + 1",
        outer_runtime=outer_runtime,
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "2"


async def test_subagent_unknown_type_message_maps_to_error_code(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    repl = _make_repl(worker, runtime, llm_model="test:model")
    outer_runtime = _FakeRuntime(
        tools=[
            _task_tool(
                lambda _d, t: (
                    f"We cannot invoke subagent {t} because it does not exist, "
                    "the only allowed types are `researcher`"
                )
            )
        ]
    )
    with pytest.raises(RuntimeError, match="ERR_SUBAGENT_TYPE_UNKNOWN"):
        await repl.eval_async(
            "await subagent({ description: 'run', subagentType: 'unknown' })",
            outer_runtime=outer_runtime,
        )


async def test_subagent_capacity_limit_enforced_per_eval(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    repl = _make_repl(
        worker,
        runtime,
        llm_model="test:model",
        subagent_max_in_flight=1,
    )

    async def _slow(_description: str, _subagent_type: str) -> str:
        await asyncio.sleep(0.05)
        return "ok"

    outer_runtime = _FakeRuntime(tools=[_task_tool(_slow)])
    with pytest.raises(RuntimeError, match="ERR_CAPACITY_EXCEEDED"):
        await repl.eval_async(
            "await Promise.all(["
            " subagent({ description: 'a', subagentType: 'worker' }),"
            " subagent({ description: 'b', subagentType: 'worker' })"
            "])",
            outer_runtime=outer_runtime,
        )


async def test_llm_structured_output_parses_to_native_value(
    worker: ThreadWorker, runtime: Runtime, monkeypatch: pytest.MonkeyPatch
) -> None:
    import langchain_quickjs._baseline.llm as baseline_llm_module

    captured: dict[str, Any] = {}

    async def _fake_invoke_one_shot(
        *,
        model: Any,
        prompt: str,
        response_schema: dict[str, Any] | None,
    ) -> str:
        captured["model"] = model
        captured["prompt"] = prompt
        captured["response_schema"] = response_schema
        return '{"value": 7}'

    monkeypatch.setattr(
        baseline_llm_module, "_invoke_one_shot", _fake_invoke_one_shot
    )
    repl = _make_repl(worker, runtime, llm_model="openai:gpt-test")
    outcome = await repl.eval_async(
        "const out = await llm({"
        " prompt: 'extract',"
        " responseSchema: { type: 'object' }"
        "});"
        " out.value"
    )
    assert outcome.error_type is None, outcome.error_message
    assert outcome.result == "7"
    assert captured["model"] == "openai:gpt-test"
    assert captured["prompt"] == "extract"
    assert captured["response_schema"] == {"type": "object"}


async def test_llm_throws_when_model_cannot_be_inferred(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    repl = _make_repl(worker, runtime)
    outer_runtime = _FakeRuntime()
    with pytest.raises(RuntimeError, match="ERR_MODEL_UNAVAILABLE"):
        await repl.eval_async(
            "await llm({ prompt: 'hello' })",
            outer_runtime=outer_runtime,
        )


async def test_fs_glob_and_editfile_roundtrip(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    backend = _MemoryBackend()
    repl = _make_repl(worker, runtime, backend=backend, llm_model="test:model")
    outcome = await repl.eval_async(
        "await fs.writeFile('/a.txt', 'hello');"
        "await fs.writeFile('/b.bin', 'aGVsbG8=', { encoding: 'base64' });"
        "const txt = await fs.readFile('/a.txt', { encoding: 'utf8' });"
        "const b64 = await fs.readFile('/b.bin', { encoding: 'base64' });"
        "const names = await fs.readdir('/');"
        "const matches = await glob('*.txt');"
        "await editFile({"
        " filePath: '/a.txt',"
        " oldString: 'hello',"
        " newString: 'hi'"
        "});"
        "const edited = await fs.readFile('/a.txt');"
        "JSON.stringify({ txt, b64, names, matches, edited });"
    )
    assert outcome.error_type is None, outcome.error_message
    parsed = json.loads(outcome.result or "{}")
    assert parsed["txt"] == "hello"
    assert parsed["b64"] == "aGVsbG8="
    assert parsed["matches"] == ["/a.txt"]
    assert parsed["edited"] == "hi"
    assert parsed["names"] == sorted(parsed["names"])


async def test_fs_writefile_wx_throws_eexist(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    backend = _MemoryBackend()
    repl = _make_repl(worker, runtime, backend=backend, llm_model="test:model")
    with pytest.raises(RuntimeError, match="EEXIST"):
        await repl.eval_async(
            "await fs.writeFile('/dup.txt', 'one');"
            "await fs.writeFile('/dup.txt', 'two', { flag: 'wx' });"
        )


async def test_backend_required_errors_when_backend_is_missing(
    worker: ThreadWorker, runtime: Runtime
) -> None:
    repl = _make_repl(worker, runtime, backend=None, llm_model="test:model")
    with pytest.raises(RuntimeError, match="ERR_BACKEND_REQUIRED"):
        await repl.eval_async("await glob('*.txt')")


def test_baseline_extension_exported_globals() -> None:
    assert FilesystemExtension.exported_globals == ("fs",)
    assert GlobExtension.exported_globals == ("glob",)
    assert SubagentExtension.exported_globals == ("subagent",)
    assert LlmExtension.exported_globals == ("llm",)
