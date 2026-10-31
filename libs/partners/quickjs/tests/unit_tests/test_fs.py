"""The `fs` capability: model code reads the agent's files through Node's `fs` API."""

from __future__ import annotations

import asyncio
import html
import json
import re
from typing import TYPE_CHECKING, Any

import pytest
from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend
from deepagents.backends.protocol import RAW_FILE_RESULT_CONFIG_KEY
from deepagents.backends.utils import create_file_data
from langchain.agents import create_agent
from langchain.tools import ToolRuntime  # noqa: TC002 — `@tool` resolves it at runtime
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool, ToolException, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import Field

from langchain_quickjs import CodeInterpreterMiddleware, FilesystemCapability, fs
from langchain_quickjs._repl import _Registry, _SnapshotRestoreError
from tests._common import FakeChatModel

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents.backends.protocol import FileDownloadResponse
    from langchain_core.messages import BaseMessage

pytestmark = pytest.mark.filterwarnings(
    "ignore:The class `CodeInterpreterMiddleware` is in beta"
)

_ROWS = json.dumps({"rows": list(range(150))}, indent=1)
_FILES = {
    "/data.json": _ROWS,
    "/docs/a.md": "hello\nTODO: write docs",
    "/docs/b.txt": "plain",
    "/secret/key.txt": "hunter2 TODO",
    "/hr/salaries.csv": "name,salary TODO",
}
_RULES = [
    FilesystemPermission(operations=["read"], paths=["/secret/**"], mode="deny"),
    FilesystemPermission(operations=["read"], paths=["/hr/**"], mode="interrupt"),
]


class _RecordingModel(FakeChatModel):
    """Fake model that keeps the system prompt of every call."""

    prompts: list[str] = Field(default_factory=list)

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.prompts.append(str(messages[0].content))
        return super()._generate(messages, *args, **kwargs)


def _model(code: str) -> _RecordingModel:
    """A model that runs one eval, then answers."""
    call = {"name": "eval", "args": {"code": code}, "id": "c0", "type": "tool_call"}
    script = [AIMessage(content="", tool_calls=[call]), AIMessage(content="done")]
    return _RecordingModel(messages=iter(script))


def _results(state: dict[str, Any]) -> list[Any]:
    """Each eval's JSON result, or its tool message when it didn't return one."""
    out = []
    for message in state["messages"]:
        if message.type != "tool" or message.name != "eval":
            continue
        match = re.fullmatch(r"<result>(.*)</result>", message.content, re.DOTALL)
        out.append(json.loads(html.unescape(match[1])) if match else message.content)
    return out


def _files() -> dict[str, Any]:
    return {path: create_file_data(text) for path, text in _FILES.items()}


def _run(
    code: str,
    *,
    run_async: bool = False,
    capability: FilesystemCapability | None = None,
    **agent_kwargs: Any,
) -> Any:
    """Run one eval in a deep agent over `_FILES` and return its result."""
    middleware = CodeInterpreterMiddleware(
        capabilities=[capability or fs()], **agent_kwargs.pop("middleware_kwargs", {})
    )
    agent = create_deep_agent(
        model=_model(code), middleware=[middleware], **agent_kwargs
    )
    payload = {"messages": [{"role": "user", "content": "go"}], "files": _files()}
    state = asyncio.run(agent.ainvoke(payload)) if run_async else agent.invoke(payload)
    return _results(state)[0]


@pytest.mark.parametrize("run_async", [False, True], ids=["invoke", "ainvoke"])
def test_reads_whole_file_unformatted(*, run_async: bool) -> None:
    code = """JSON.stringify([
      JSON.parse(fs.readFileSync("/data.json", "utf8")).rows.length,
      JSON.parse(await fs.promises.readFile("/data.json", "utf8")).rows.length,
    ])"""
    assert _run(code, run_async=run_async) == [150, 150]


def test_import_and_require_resolve_to_the_same_modules() -> None:
    code = """
    const same = async (id, name, expected) => {
      const m = await import(id);
      return [m[name], m.default[name], require(id)[name]].every((f) => f === expected);
    };
    let missing;
    try { require("child_process") } catch (e) { missing = e.code }
    JSON.stringify([
      await same("fs", "readFileSync", fs.readFileSync),
      await same("node:fs", "readFileSync", fs.readFileSync),
      await same("fs/promises", "readFile", fs.promises.readFile),
      await same("node:fs/promises", "readFile", fs.promises.readFile),
      await same("path", "join", require("path").join),
      await same("node:path", "dirname", require("node:path").dirname),
      missing,
    ])"""
    assert _run(code) == [True] * 6 + ["MODULE_NOT_FOUND"]


def test_reads_exact_bytes(tmp_path: Path) -> None:
    (tmp_path / "bin.dat").write_bytes(b"a\r\nb\x00\xff")
    (tmp_path / "crlf.txt").write_bytes(b"x\r\ny\r\n")
    (tmp_path / "data.json").write_text('{"name": "hé"}', encoding="utf-8")
    code = """JSON.stringify([
      JSON.parse(fs.readFileSync("/data.json")).name,
      String(await fs.promises.readFile("/data.json")),
      fs.readFileSync("/bin.dat") instanceof Uint8Array,
      Array.from(fs.readFileSync("/bin.dat")),
      Array.from(await fs.promises.readFile("/bin.dat")),
      fs.readFileSync("/crlf.txt", "utf8"),
      fs.readFileSync("/crlf.txt", { encoding: "UTF-8" }),
      fs.readFileSync("/bin.dat", "base64"),
    ])"""
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    assert _run(code, backend=backend) == [
        "hé",
        '{"name": "hé"}',
        True,
        [97, 13, 10, 98, 0, 255],
        [97, 13, 10, 98, 0, 255],
        "x\r\ny\r\n",
        "x\r\ny\r\n",
        "YQ0KYgD/",
    ]


@pytest.mark.parametrize("run_async", [False, True], ids=["invoke", "ainvoke"])
def test_fs_is_gated_exactly_like_tools(*, run_async: bool) -> None:
    code = """
    const sync = (p) => {
      try { fs.readFileSync(p); return "read" } catch (e) { return e.code }
    };
    const promise = (p) => fs.promises.readFile(p).then(() => "read", (e) => e.code);
    const viaTools = async (p) => {
      const out = await tools.readFile({ file_path: p });
      return out.startsWith("Error:") ? "refused" : "read";
    };
    const paths = ["/docs/b.txt", "/secret/key.txt", "/hr/salaries.csv"];
    JSON.stringify(await Promise.all(
      paths.map(async (p) => [sync(p), await promise(p), await viaTools(p)])
    ))"""
    result = _run(
        code,
        run_async=run_async,
        permissions=_RULES,
        middleware_kwargs={"ptc": ["read_file"]},
    )
    assert result == [
        ["read", "read", "read"],
        ["EACCES", "EACCES", "refused"],
        ["read", "read", "read"],
    ]


@tool("config_has_raw_request")
def _config_has_raw_request(runtime: ToolRuntime) -> bool:
    """Report whether this call's config carries a raw file result request."""
    return RAW_FILE_RESULT_CONFIG_KEY in runtime.config.get("configurable", {})


def test_raw_result_requests_stay_on_fs_calls() -> None:
    code = """
    const text = fs.readFileSync("/docs/b.txt", "utf8");
    JSON.stringify([text, await tools.configHasRawRequest({})])"""
    result = _run(
        code,
        tools=[_config_has_raw_request],
        middleware_kwargs={"ptc": ["config_has_raw_request"]},
    )
    assert result == ["plain", False]


def test_listings_show_what_the_tools_permit() -> None:
    code = """
    const globbed = [];
    for await (const p of fs.promises.glob("**/*")) globbed.push(p);
    JSON.stringify([
      fs.readdirSync("/", { withFileTypes: true })
        .map((d) => [d.name, d.isDirectory(), d.isFile(), d.parentPath])
        .sort(),
      (await fs.promises.readdir("/docs")).sort(),
      fs.readdirSync("/secret"),
      fs.globSync("**/*").sort(),
      globbed.sort(),
      fs.globSync("*.md", { cwd: "/docs" }),
      ["/docs", "docs/a.md", "/nope", "/nope/x", "/", "/secret/key.txt"]
        .map(fs.existsSync),
    ])"""
    visible = ["data.json", "docs/a.md", "docs/b.txt", "hr/salaries.csv"]
    assert _run(code, permissions=_RULES) == [
        [
            ["data.json", False, True, "/"],
            ["docs", True, False, "/"],
            ["hr", True, False, "/"],
        ],
        ["a.md", "b.txt"],
        [],
        visible,
        visible,
        ["a.md"],
        [True, True, False, False, True, False],
    ]


def test_failures_are_node_errors(tmp_path: Path) -> None:
    (tmp_path / "dir").mkdir()
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "key.txt").write_text("k")
    (tmp_path / "small.txt").write_text("plain")
    (tmp_path / "large.txt").write_text("too large")
    code = """
    const caught = (read) => {
      try { return read() } catch (e) { return [e.name, e.code, e.message] }
    };
    JSON.stringify([
      caught(() => fs.readFileSync("/missing.txt")),
      caught(() => fs.readFileSync("/dir")),
      caught(() => fs.readFileSync("/secret/key.txt")),
      caught(() => fs.readdirSync("/missing")),
      caught(() => fs.readFileSync("/small.txt", "utf32")).slice(0, 2),
      caught(() => fs.readFileSync(42)).slice(0, 2),
      ((e) => [e[0], e[1], e[2].includes("9 bytes")])(
        caught(() => fs.readFileSync("/large.txt", "utf8"))
      ),
      fs.readFileSync("/small.txt", "utf8"),
    ])"""
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    capability = fs(max_read_bytes=8)
    assert _run(code, capability=capability, backend=backend, permissions=_RULES) == [
        ["Error", "ENOENT", "ENOENT: no such file or directory, open '/missing.txt'"],
        ["Error", "EISDIR", "EISDIR: illegal operation on a directory, open '/dir'"],
        ["Error", "EACCES", "EACCES: permission denied, open '/secret/key.txt'"],
        ["Error", "EIO", "EIO: Path '/missing': path_not_found, scandir '/missing'"],
        ["TypeError", "ERR_INVALID_ARG_VALUE"],
        ["TypeError", "ERR_INVALID_ARG_TYPE"],
        ["RangeError", "ERR_FS_FILE_TOO_LARGE", True],
        "plain",
    ]


def test_fs_calls_count_toward_the_call_budget() -> None:
    code = 'fs.readFileSync("/docs/b.txt"); fs.existsSync("/a"); fs.existsSync("/a")'
    result = _run(code, middleware_kwargs={"max_ptc_calls": 2})
    assert 'type="PTCCallBudgetExceeded"' in result
    assert "function=fs.readdir" in result


class _AsyncOnlyBackend(StateBackend):
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        raise NotImplementedError

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        return super().download_files(paths)


def test_sync_reads_use_the_agents_loop_under_ainvoke() -> None:
    code = 'JSON.stringify(fs.readFileSync("/docs/b.txt", "utf8"))'
    assert _run(code, run_async=True, backend=_AsyncOnlyBackend()) == "plain"


@tool("read_file")
def _read_file_as_text(file_path: str) -> str:
    """Return the file as plain text, as a custom tool does."""
    return f"contents of {file_path}"


@pytest.mark.parametrize(
    ("capabilities", "tools", "prompt_text"),
    [
        ([fs()], [_read_file_as_text], "### Files"),
        ([fs()], [], "no built-in filesystem"),
        ([], [_read_file_as_text], "no built-in filesystem"),
    ],
    ids=["fs", "fs-without-file-tools", "no-fs"],
)
def test_globals_prompt_and_schema_follow_the_capability(
    capabilities: list[FilesystemCapability], tools: list[Any], prompt_text: str
) -> None:
    code = """
    let imported = "imported";
    try { await import("fs") } catch { imported = "failed" }
    JSON.stringify([typeof fs, typeof require, imported])
    """
    model = _model(code)
    middleware = CodeInterpreterMiddleware(capabilities=capabilities)
    agent = create_agent(model, tools=tools, middleware=[middleware])
    state = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    eval_tool = middleware.tools[0]
    schema = eval_tool.tool_call_schema.model_json_schema()["properties"]["code"]
    access = "Read-only `fs` when the agent has file tools"
    on = bool(capabilities)
    expected = (
        ["object", "function", "imported"]
        if on
        else ["undefined", "undefined", "failed"]
    )
    assert _results(state) == [expected]
    assert (access in eval_tool.description, access in schema["description"]) == (
        on,
        on,
    )
    assert prompt_text in model.prompts[0]


@tool("read_file")
def _read_file_without_raw_result(file_path: str, runtime: ToolRuntime) -> ToolMessage:
    """Read a file the way a deepagents release without raw results does."""
    return ToolMessage(f"1\t{file_path}", tool_call_id=runtime.tool_call_id)


@tool("read_file")
def _read_file_that_fails(file_path: str, runtime: ToolRuntime) -> ToolMessage:
    """Fail with an error result that carries no raw result."""
    content = f"Error: disk failed reading {file_path}"
    return ToolMessage(content, tool_call_id=runtime.tool_call_id, status="error")


def _raise_disk_failure(file_path: str) -> str:
    """Fail with an exception that `handle_tool_error` reports as text."""
    msg = f"disk failed reading {file_path}"
    raise ToolException(msg)


_read_file_that_raises = StructuredTool.from_function(
    _raise_disk_failure, name="read_file", handle_tool_error=True
)


@pytest.mark.parametrize(
    ("tools", "code", "message"),
    [
        ([], "ENOSYS", "this agent has no read_file tool"),
        ([_read_file_without_raw_result], "ENOSYS", "fs needs a deepagents release"),
        ([_read_file_as_text], "ENOSYS", "ENOSYS: the read_file tool returned no raw"),
        ([_read_file_that_fails], "EIO", "EIO: disk failed reading /a, open '/a'"),
        ([_read_file_that_raises], "EIO", "EIO: disk failed reading /a, open '/a'"),
    ],
    ids=["no-file-tools", "no-raw-result", "text-result", "error-result", "raised"],
)
def test_fs_errors_when_a_file_tool_returns_no_raw_result(
    tools: list[Any], code: str, message: str
) -> None:
    js = """
    let error;
    try { fs.readFileSync("/a", "utf8") } catch (e) { error = [e.code, e.message] }
    JSON.stringify(error)
    """
    model = _model(js)
    middleware = CodeInterpreterMiddleware(capabilities=[fs()])
    agent = create_agent(model, tools=tools, middleware=[middleware])
    state = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    thrown, text = _results(state)[0]
    assert (thrown, message in text) == (code, True)


def test_exists_sync_throws_when_the_agent_has_no_ls_tool() -> None:
    js = 'let r; try { fs.existsSync("/a") } catch (e) { r = e.code } JSON.stringify(r)'
    middleware = CodeInterpreterMiddleware(capabilities=[fs()])
    agent = create_agent(
        _model(js), tools=[_read_file_as_text], middleware=[middleware]
    )
    state = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    assert _results(state) == ["ENOSYS"]


def test_a_fresh_process_restores_state_and_replaces_a_stale_fs() -> None:
    first = (
        'globalThis.cached = fs.readFileSync("/docs/b.txt", "utf8");'
        ' globalThis.fs = "stale"; "1"'
    )
    second = 'JSON.stringify([cached, fs.readFileSync("/docs/a.md", "utf8").length])'
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "t"}}
    for code, message in ((first, {"files": _files()}), (second, {})):
        agent = create_deep_agent(
            model=_model(code),
            middleware=[CodeInterpreterMiddleware(capabilities=[fs()])],
            checkpointer=saver,
        )
        state = agent.invoke(
            {"messages": [{"role": "user", "content": "go"}], **message}, config
        )
    assert _results(state)[-1] == ["plain", len(_FILES["/docs/a.md"])]


def test_fs_works_in_an_eval_resumed_without_a_model_call() -> None:
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "t"}}
    code = 'JSON.stringify(fs.readFileSync("/docs/b.txt", "utf8"))'
    models = (_model(code), FakeChatModel(messages=iter([AIMessage(content="done")])))
    payloads: tuple[Any, Any] = (
        {"messages": [{"role": "user", "content": "go"}], "files": _files()},
        Command(resume={"decisions": [{"type": "approve"}]}),
    )
    for model, payload in zip(models, payloads, strict=True):
        agent = create_deep_agent(
            model=model,
            middleware=[CodeInterpreterMiddleware(capabilities=[fs()])],
            interrupt_on={"eval": True},
            checkpointer=saver,
        )
        state = agent.invoke(payload, config)
    assert _results(state) == ["plain"]


def test_capability_configuration_is_validated() -> None:
    with pytest.raises(ValueError, match="max_read_bytes"):
        fs(max_read_bytes=0)
    with pytest.raises(ValueError, match="at most once"):
        CodeInterpreterMiddleware(capabilities=[fs(), fs()])
    with pytest.raises(TypeError, match=r"fs\(\)"):
        CodeInterpreterMiddleware(capabilities=["fs"])


def _registry() -> _Registry:
    return _Registry(
        memory_limit=64 * 1024 * 1024,
        timeout=5,
        capture_console=True,
        max_stdout_chars=4000,
        files=fs(),
    )


def test_fs_is_installed_after_a_failed_snapshot_restore() -> None:
    registry = _registry()
    repl = registry.get("slot")
    try:
        with pytest.raises(_SnapshotRestoreError):
            repl.eval_sync("1", snapshot_payload=b"not a snapshot")
        outcome = repl.eval_sync("JSON.stringify([typeof fs, typeof require])")
    finally:
        registry.close()
    assert json.loads(outcome.result or "") == ["object", "function"]


def test_code_that_declares_its_own_fs_keeps_running() -> None:
    registry = _registry()
    repl = registry.get("slot")
    try:
        repl.eval_sync("function fs() { return 'mine' }; 1")
        outcome = repl.eval_sync(
            'JSON.stringify([fs(), typeof require("fs").readFileSync])'
        )
    finally:
        registry.close()
    assert json.loads(outcome.result or "") == ["mine", "function"]
