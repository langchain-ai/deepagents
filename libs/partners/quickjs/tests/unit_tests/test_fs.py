from __future__ import annotations

import asyncio
import html
import json
import re
from typing import TYPE_CHECKING, Any

import pytest
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend
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
        FileDownloadResponse,
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
        }
    ]


def test_a_directory_in_agent_state_is_eisdir() -> None:
    code = CODE_OF + 'JSON.stringify(await codeOf(() => fs.readFile("/docs")))'

    assert run(code) == ["EISDIR"]


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
    code = 'JSON.stringify((await __lc_fs("readFile", { path: 42 })).error.code)'

    assert run(code) == ["ERR_INVALID_ARG_TYPE"]


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
      callback: await codeOf(() => fs.readFile("/docs/b.txt", "utf8", () => {})),
      notAPath: await codeOf(() => fs.readFile(42)),
    })"""
    )

    assert run(code) == [
        {
            "base64": "ERR_INVALID_ARG_VALUE",
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
    assert result == {"require": "MODULE_NOT_FOUND", "promisesReadFile": "function"}


def test_code_has_no_fs_without_a_backend() -> None:
    agent = create_deep_agent(
        model=model("JSON.stringify([typeof fs, typeof require])"),
        middleware=[CodeInterpreterMiddleware()],
    )

    state = agent.invoke({"messages": [{"role": "user", "content": "go"}]})

    assert results(state) == [["undefined", "undefined"]]


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
