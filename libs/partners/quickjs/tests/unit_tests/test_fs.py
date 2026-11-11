from __future__ import annotations

import asyncio
import html
import json
import re
from typing import TYPE_CHECKING, Any

import pytest
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.protocol import FileDownloadResponse
from deepagents.backends.utils import create_file_data
from langchain.agents import create_agent
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from langchain_quickjs import CodeInterpreterMiddleware
from tests._common import FakeChatModel

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID

    from deepagents.backends.protocol import (
        BackendProtocol,
        LsResult,
    )
    from langchain_core.messages import BaseMessage

FILES = {
    "/data.json": json.dumps({"rows": list(range(150))}, indent=1),
    "/docs/a.md": "hello\nworld",
    "/docs/b.txt": "plain",
}

CODE_OF = "const codeOf = (f) => f().then(() => 'ok', (e) => e.code);\n"


class RecordingModel(FakeChatModel):
    prompts: list[str] = Field(default_factory=list)

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.prompts.append(str(messages[0].content))
        return super()._generate(messages, *args, **kwargs)


def model(*codes: str) -> RecordingModel:
    script = [
        AIMessage("", tool_calls=[{"name": "eval", "args": {"code": c}, "id": f"c{i}"}])
        for i, c in enumerate(codes)
    ]
    return RecordingModel(messages=iter([*script, AIMessage(content="done")]))


def results(state: dict[str, Any]) -> list[Any]:
    out = []
    for message in state["messages"]:
        if message.type != "tool" or message.name != "eval":
            continue
        match = re.fullmatch(r"<result>(.*)</result>", message.content, re.DOTALL)
        out.append(json.loads(html.unescape(match[1])) if match else message.content)
    return out


def files() -> dict[str, Any]:
    return {path: create_file_data(text) for path, text in FILES.items()}


def run(
    *codes: str,
    backend: BackendProtocol | None = None,
    run_async: bool = False,
    **middleware_kwargs: Any,
) -> list[Any]:
    backend = backend or StateBackend()
    agent = create_deep_agent(
        model=model(*codes),
        backend=backend,
        middleware=[CodeInterpreterMiddleware(fs=backend, **middleware_kwargs)],
    )
    payload = {"messages": [{"role": "user", "content": "go"}], "files": files()}
    if run_async:
        return results(asyncio.run(agent.ainvoke(payload)))
    return results(agent.invoke(payload))


@pytest.mark.parametrize("run_async", [False, True], ids=["invoke", "ainvoke"])
def test_reads_files_in_agent_state_as_text(*, run_async: bool) -> None:
    code = """JSON.stringify({
      read: await fs.readFile("/docs/a.md"),
      rows: JSON.parse(await fs.promises.readFile("/data.json", "utf8")).rows.length,
    })"""

    assert run(code, run_async=run_async) == [{"read": "hello\nworld", "rows": 150}]


@pytest.mark.parametrize("run_async", [False, True], ids=["invoke", "ainvoke"])
def test_writes_lists_and_deletes_files_in_agent_state(*, run_async: bool) -> None:
    code = """const out = {};
    await fs.writeFile("/w/new.txt", "made");
    out.readBack = await fs.readFile("/w/new.txt");
    out.listed = (await fs.readdir("/docs")).sort();
    out.exists = await fs.access("/docs/a.md").then(() => true, () => false);
    out.root = await fs.access("/").then(() => true, () => false);
    out.mode = await fs.access("/docs/a.md", fs.constants.R_OK).then(
      () => [fs.constants.F_OK, fs.promises.constants.W_OK],
    );
    await fs.rm("/docs/b.txt");
    out.deleted = await fs.access("/docs/b.txt").then(() => "kept", (e) => e.code);
    JSON.stringify(out)"""
    backend = StateBackend()
    agent = create_deep_agent(
        model=model(code),
        backend=backend,
        middleware=[CodeInterpreterMiddleware(fs=backend)],
    )
    payload = {"messages": [{"role": "user", "content": "go"}], "files": files()}

    state = asyncio.run(agent.ainvoke(payload)) if run_async else agent.invoke(payload)

    assert results(state) == [
        {
            "readBack": "made",
            "listed": ["a.md", "b.txt"],
            "exists": True,
            "root": True,
            "mode": [0, 2],
            "deleted": "ENOENT",
        }
    ]
    assert state["files"]["/w/new.txt"]["content"] == "made"
    assert "/docs/b.txt" not in state["files"]


@pytest.mark.parametrize("on_disk", [False, True], ids=["state", "filesystem"])
def test_write_file_replaces_an_existing_file(tmp_path: Path, *, on_disk: bool) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "b.txt").write_text("plain")
    backend = (
        FilesystemBackend(root_dir=tmp_path, virtual_mode=True) if on_disk else None
    )
    code = """await fs.writeFile("/docs/b.txt", "replaced");
    JSON.stringify(await fs.readFile("/docs/b.txt"))"""

    assert run(code, backend=backend) == ["replaced"]


def test_fs_follows_the_routes_of_a_composite_backend(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("on disk")
    disk = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    backend = CompositeBackend(default=StateBackend(), routes={"/disk/": disk})
    code = (
        CODE_OF
        + """const out = {};
    out.read = await fs.readFile("/disk/a.txt");
    out.listed = await fs.readdir("/disk");
    out.missing = await codeOf(() => fs.access("/disk/nope.txt"));
    await fs.writeFile("/disk/new/b.txt", "made");
    await fs.rm("/disk/a.txt");
    out.removed = await codeOf(() => fs.access("/disk/a.txt"));
    out.stateRead = await fs.readFile("/docs/b.txt");
    JSON.stringify(out)"""
    )

    assert run(code, backend=backend) == [
        {
            "read": "on disk",
            "listed": ["a.txt"],
            "missing": "ENOENT",
            "removed": "ENOENT",
            "stateRead": "plain",
        }
    ]
    assert (tmp_path / "new" / "b.txt").read_text() == "made"
    assert not (tmp_path / "a.txt").exists()


def test_failures_are_node_errors(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("hello")
    (tmp_path / "latin1.txt").write_bytes("café".encode("latin-1"))
    code = (
        CODE_OF
        + """JSON.stringify({
      missing: await codeOf(() => fs.readFile("/missing.txt")),
      readDirectory: await codeOf(() => fs.readFile("/docs")),
      underAFile: await codeOf(() => fs.readFile("/docs/a.md/x")),
      trailingSlash: await codeOf(() => fs.readFile("/docs/a.md/")),
      notUtf8: await codeOf(() => fs.readFile("/latin1.txt")),
      listFile: await codeOf(() => fs.readdir("/docs/a.md")),
      listMissing: await codeOf(() => fs.readdir("/nope/deeper")),
      accessUnderFile: await codeOf(() => fs.access("/docs/a.md/x")),
      accessFileAsDirectory: await codeOf(() => fs.access("/docs/a.md/")),
      writeDirectory: await codeOf(() => fs.writeFile("/docs", "x")),
      writeDirectoryPath: await codeOf(() => fs.writeFile("/new/", "x")),
      rmMissing: await codeOf(() => fs.rm("/missing.txt")),
      rmMissingForced: await codeOf(() => fs.rm("/missing.txt", { force: true })),
      rmDirectory: await codeOf(() => fs.rm("/docs")),
      rmDirectoryRecursive: await codeOf(() => fs.rm("/docs", { recursive: true })),
    })"""
    )
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)

    assert run(code, backend=backend) == [
        {
            "missing": "ENOENT",
            "readDirectory": "EISDIR",
            "underAFile": "ENOENT",
            "trailingSlash": "ENOTDIR",
            "notUtf8": "ERR_ENCODING_INVALID_ENCODED_DATA",
            "listFile": "ENOTDIR",
            "listMissing": "ENOENT",
            "accessUnderFile": "ENOENT",
            "accessFileAsDirectory": "ENOTDIR",
            "writeDirectory": "EISDIR",
            "writeDirectoryPath": "EISDIR",
            "rmMissing": "ENOENT",
            "rmMissingForced": "ok",
            "rmDirectory": "ERR_FS_EISDIR",
            "rmDirectoryRecursive": "ok",
        }
    ]
    assert (tmp_path / "latin1.txt").read_bytes() == "café".encode("latin-1")
    assert not (tmp_path / "docs").exists()
    assert not (tmp_path / "new").exists()


def test_agent_state_failures_are_node_errors() -> None:
    code = (
        CODE_OF
        + """const syscall = (f) => f().catch((e) => e.syscall);
    JSON.stringify({
      readDirectory: await codeOf(() => fs.readFile("/docs")),
      accessMissing: await codeOf(() => fs.access("/missing.txt")),
      accessFileAsDirectory: await codeOf(() => fs.access("/docs/a.md/")),
      writeDirectoryPath: await codeOf(() => fs.writeFile("/docs/", "x")),
      rmMissing: await codeOf(() => fs.rm("/missing.txt")),
      rmDirectory: await codeOf(() => fs.rm("/docs")),
      syscalls: [
        await syscall(() => fs.access("/missing.txt")),
        await syscall(() => fs.rm("/missing.txt")),
      ],
    })"""
    )

    assert run(code) == [
        {
            "readDirectory": "EISDIR",
            "accessMissing": "ENOENT",
            "accessFileAsDirectory": "ENOTDIR",
            "writeDirectoryPath": "EISDIR",
            "rmMissing": "ENOENT",
            "rmDirectory": "ERR_FS_EISDIR",
            "syscalls": ["access", "lstat"],
        }
    ]


def test_dot_dot_segments_stay_inside_the_backend_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "inside.txt").write_text("in")
    (tmp_path / "outside.txt").write_text("out")
    code = (
        CODE_OF
        + """JSON.stringify([
      await fs.readFile("/a/../../inside.txt"),
      await codeOf(() => fs.readFile("/../outside.txt")),
    ])"""
    )
    backend = FilesystemBackend(root_dir=root, virtual_mode=True)

    assert run(code, backend=backend) == [["in", "ENOENT"]]


class _CountingDisk(FilesystemBackend):
    listings = 0

    def ls(self, path: str) -> LsResult:
        self.listings += 1
        return super().ls(path)


def test_a_deep_missing_path_costs_a_few_listings(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x")
    backend = _CountingDisk(root_dir=tmp_path, virtual_mode=True)
    deep = '"d/".repeat(300) + "x"'
    code = (
        CODE_OF
        + f"""JSON.stringify([
      await codeOf(() => fs.readFile("/" + {deep})),
      await codeOf(() => fs.readFile("/f.txt/" + {deep})),
    ])"""
    )

    assert run(code, backend=backend) == [["ENOENT", "ENOENT"]]
    assert backend.listings == 2


def test_the_host_function_rejects_arguments_that_are_not_strings() -> None:
    code = (
        CODE_OF
        + """JSON.stringify([
      (await __lc_fs("readFile", { path: 42 })).error.code,
      (await __lc_fs("writeFile", { path: "/n.txt", data: 42 })).error.code,
      await codeOf(() => fs.access("/n.txt")),
    ])"""
    )

    assert run(code) == [["ERR_INVALID_ARG_TYPE", "ERR_INVALID_ARG_TYPE", "ENOENT"]]


class _Broken(StateBackend):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        raise self.error


@pytest.mark.parametrize(
    ("error", "expected"),
    [(NotImplementedError(), "ENOSYS"), (OSError("disk on fire"), "EIO")],
    ids=["unsupported", "failing"],
)
def test_a_backend_that_raises_rejects_one_call(
    error: Exception, expected: str
) -> None:
    code = CODE_OF + "JSON.stringify([await codeOf(() => fs.readFile('/a')), 1])"

    assert run(code, backend=_Broken(error)) == [[expected, 1]]


def test_arguments_fs_cannot_honor_are_refused() -> None:
    code = (
        CODE_OF
        + """JSON.stringify({
      base64: await codeOf(() => fs.readFile("/docs/b.txt", "base64")),
      bytes: await codeOf(() => fs.writeFile("/w.txt", new Uint8Array([1]))),
      append: await codeOf(() => fs.writeFile("/w.txt", "x", { flag: "a" })),
      replace: await codeOf(() => fs.writeFile("/w.txt", "x", { flag: "w" })),
      fileTypes: await codeOf(() => fs.readdir("/docs", { withFileTypes: true })),
      recursive: await codeOf(() => fs.readdir("/docs", { recursive: true })),
      buffer: await codeOf(() => fs.readdir("/docs", "buffer")),
      callback: await codeOf(() => fs.readFile("/docs/b.txt", "utf8", () => {})),
      notAPath: await codeOf(() => fs.readFile(42)),
    })"""
    )

    assert run(code) == [
        {
            "base64": "ERR_INVALID_ARG_VALUE",
            "bytes": "ERR_INVALID_ARG_TYPE",
            "append": "ERR_INVALID_ARG_VALUE",
            "replace": "ok",
            "fileTypes": "ERR_INVALID_ARG_VALUE",
            "recursive": "ERR_INVALID_ARG_VALUE",
            "buffer": "ERR_INVALID_ARG_VALUE",
            "callback": "ERR_INVALID_ARG_TYPE",
            "notAPath": "ERR_INVALID_ARG_TYPE",
        }
    ]


def test_sync_and_dropped_functions_throw_an_error_naming_the_replacement() -> None:
    code = """const thrown = (f) => {
      try { f(); return null } catch (e) { return [e.code, e.message] }
    };
    const dropped = `readFileSync writeFileSync readdirSync existsSync exists
      accessSync stat statSync lstat glob rmSync unlink rmdir appendFile mkdir`;
    JSON.stringify({
      dropped: Object.fromEntries(dropped.split(/\\s+/).map(
        (name) => [name, thrown(() => fs.promises[name]("/docs/a.md"))],
      )),
      require: thrown(() => require("fs"))[0],
      promisesReadFile: typeof fs.promises.readFile,
    })"""

    [result] = run(code)

    dropped = result.pop("dropped")
    assert len(dropped) == 15
    assert {code for code, _ in dropped.values()} == {"ERR_METHOD_NOT_IMPLEMENTED"}
    assert "Use await fs.readFile(path)." in dropped["readFileSync"][1]
    assert "Use await fs.access(path)," in dropped["existsSync"][1]
    assert result == {"require": "MODULE_NOT_FOUND", "promisesReadFile": "function"}


@pytest.mark.parametrize("enabled", [True, False], ids=["backend", "default"])
def test_fs_and_its_prompt_section_need_a_backend(*, enabled: bool) -> None:
    backend = StateBackend()
    llm = model("JSON.stringify([typeof fs, typeof require])")
    middleware = CodeInterpreterMiddleware(fs=backend if enabled else None)
    agent = create_deep_agent(model=llm, backend=backend, middleware=[middleware])

    state = agent.invoke({"messages": [{"role": "user", "content": "go"}]})

    expected = ["object", "function"] if enabled else ["undefined", "undefined"]
    assert results(state) == [expected]
    assert ("### Files: `fs`" in llm.prompts[0]) is enabled
    assert ("No filesystem" in middleware.tools[0].description) is not enabled


def test_an_oversized_read_ends_only_that_eval(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("a" * (16 * 1024 * 1024))
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)

    out = run(
        '(await fs.readFile("/big.txt")).length',
        "1 + 1",
        backend=backend,
        memory_limit=8 * 1024 * 1024,
    )

    assert 'type="MarshalError"' in out[0]
    assert out[1] == 2


def test_a_host_result_nested_too_deep_ends_only_that_eval() -> None:
    @tool
    def nested() -> dict[str, Any]:
        """Return a deeply nested object."""
        value: dict[str, Any] = {}
        for _ in range(300):
            value = {"x": value}
        return value

    agent = create_agent(
        model=model("typeof (await tools.nested({}))", "1 + 1"),
        tools=[nested],
        middleware=[CodeInterpreterMiddleware(ptc=["nested"])],
    )

    out = results(agent.invoke({"messages": [{"role": "user", "content": "go"}]}))

    assert 'type="MarshalError"' in out[0]
    assert "too deeply nested" in out[0]
    assert out[1] == 2


def test_fs_calls_count_toward_the_call_budget() -> None:
    code = """for (const p of ["/docs/a.md", "/docs/b.txt", "/data.json"]) {
      await fs.readFile(p);
    }"""

    assert run(code, max_ptc_calls=2) == [
        '<error type="PTCCallBudgetExceeded">PTC call budget exceeded '
        "(limit=2, attempted=3, function=fs.readFile)</error>"
    ]


class _Runs(BaseCallbackHandler):
    def __init__(self) -> None:
        self.names: dict[UUID, str] = {}
        self.outputs: dict[str, list[Any]] = {}

    def on_chain_start(self, serialized: Any, inputs: Any, **kwargs: Any) -> None:
        self.names[kwargs["run_id"]] = kwargs.get("name") or ""

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        name = self.names[kwargs["run_id"]]
        self.outputs.setdefault(name, []).append(outputs)


def test_file_contents_stay_out_of_traces(tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("TOPSECRET")
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    agent = create_deep_agent(
        model=model('(await fs.readFile("/secret.txt")).length'),
        backend=backend,
        middleware=[CodeInterpreterMiddleware(fs=backend)],
    )
    runs = _Runs()

    agent.invoke(
        {"messages": [{"role": "user", "content": "go"}]}, {"callbacks": [runs]}
    )

    assert runs.outputs["fs.readFile"] == [None]


@pytest.mark.parametrize("run_async", [False, True], ids=["invoke", "ainvoke"])
def test_fs_works_in_a_repl_restored_from_a_snapshot(*, run_async: bool) -> None:
    backend = StateBackend()
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "restored"}}

    def turn(code: str) -> list[Any]:
        agent = create_deep_agent(
            model=model(code),
            backend=backend,
            middleware=[CodeInterpreterMiddleware(fs=backend)],
            checkpointer=saver,
        )
        payload = {"messages": [{"role": "user", "content": "go"}], "files": files()}
        if run_async:
            return results(asyncio.run(agent.ainvoke(payload, config)))
        return results(agent.invoke(payload, config))

    turn('const read = () => fs.readFile("/docs/b.txt"); 1')
    out = turn("JSON.stringify([await read(), await fs.readFile('/docs/a.md')])")

    assert out[-1] == ["plain", "hello\nworld"]


@pytest.mark.parametrize("function", ["access", "rm"])
def test_a_budget_error_names_the_fs_function_the_code_called(function: str) -> None:
    code = f'await fs.readFile("/docs/a.md"); await fs.{function}("/docs/b.txt")'

    [message] = run(code, max_ptc_calls=1)

    assert f"attempted=2, function=fs.{function})" in message


@pytest.mark.parametrize("run_async", [False, True], ids=["invoke", "ainvoke"])
@pytest.mark.parametrize(
    ("kind", "path", "expected"),
    [
        ("state", "/docs/b.txt", "plain"),
        ("filesystem", "/a.txt", "on disk"),
        ("composite", "/d/a.txt", "on disk"),
    ],
)
def test_fs_true_reads_through_the_agents_backend(
    tmp_path: Path, kind: str, path: str, expected: str, *, run_async: bool
) -> None:
    (tmp_path / "a.txt").write_text("on disk")
    disk = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    composite = CompositeBackend(default=StateBackend(), routes={"/d/": disk})
    backend = {"state": None, "filesystem": disk, "composite": composite}[kind]
    agent = create_deep_agent(
        model=model(f"JSON.stringify(await fs.readFile({path!r}))"),
        middleware=[CodeInterpreterMiddleware(fs=True)],
        **({} if backend is None else {"backend": backend}),
    )
    payload = {"messages": [{"role": "user", "content": "go"}], "files": files()}

    state = asyncio.run(agent.ainvoke(payload)) if run_async else agent.invoke(payload)

    assert results(state) == [expected]


def test_an_explicit_fs_backend_wins_over_the_agents(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "b.txt").write_text("own disk")
    own = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    agent = create_deep_agent(
        model=model('JSON.stringify(await fs.readFile("/docs/b.txt"))'),
        middleware=[CodeInterpreterMiddleware(fs=own)],
    )

    state = agent.invoke(
        {"messages": [{"role": "user", "content": "go"}], "files": files()}
    )

    assert results(state) == ["own disk"]


def test_one_fs_true_middleware_serves_one_agents_files(tmp_path: Path) -> None:
    shared = CodeInterpreterMiddleware(fs=True)
    create_deep_agent(model=model("1"), middleware=[shared])
    create_deep_agent(model=model("1"), middleware=[shared])
    disk = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    other = CodeInterpreterMiddleware(fs=True)
    create_deep_agent(model=model("1"), backend=disk, middleware=[other])

    with pytest.raises(ValueError, match="one middleware per agent"):
        create_deep_agent(
            model=model("1"),
            backend=FilesystemBackend(root_dir=tmp_path / "x", virtual_mode=True),
            middleware=[other],
        )


def test_fs_accepts_a_backend_that_does_not_subclass_the_protocol() -> None:
    class Duck:
        async def adownload_files(self, paths: list[str]) -> list[Any]:
            return [FileDownloadResponse(path=p, content=b"duck") for p in paths]

    agent = create_deep_agent(
        model=model('JSON.stringify(await fs.readFile("/a.txt"))'),
        middleware=[CodeInterpreterMiddleware(fs=Duck())],
    )

    state = agent.invoke({"messages": [{"role": "user", "content": "go"}]})

    assert results(state) == ["duck"]


def test_fs_true_without_a_bound_backend_fails_on_the_first_model_call() -> None:
    agent = create_agent(
        model=model("1"), middleware=[CodeInterpreterMiddleware(fs=True)]
    )

    with pytest.raises(ValueError, match="has no backend"):
        agent.invoke({"messages": [{"role": "user", "content": "go"}]})
