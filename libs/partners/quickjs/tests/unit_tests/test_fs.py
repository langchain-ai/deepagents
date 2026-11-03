"""`fs`: model code reads the agent's files through Node's `fs` API."""

from __future__ import annotations

import asyncio
import html
import json
import re
import time
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import pytest
from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend
from deepagents.backends.utils import create_file_data
from deepagents.middleware.filesystem import FilesystemMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from langchain_quickjs import CodeInterpreterMiddleware
from langchain_quickjs._repl import _Registry
from tests._common import FakeChatModel, answer_pauses

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents.backends.protocol import FileDownloadResponse
    from langchain_core.messages import BaseMessage

_FILES = {
    "/data.json": json.dumps({"rows": list(range(150))}, indent=1),
    "/docs/a.md": "hello\nTODO: write docs",
    "/docs/b.txt": "plain",
    "/secret/key.txt": "hunter2",
}
_FS_TOOLS = ["read_file", "ls"]
_DENY_SECRET = [
    FilesystemPermission(operations=["read"], paths=["/secret/**"], mode="deny")
]


class _RecordingModel(FakeChatModel):
    """Fake model that keeps the system prompt of every call."""

    prompts: list[str] = Field(default_factory=list)

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.prompts.append(str(messages[0].content))
        return super()._generate(messages, *args, **kwargs)


def _model(*codes: str) -> _RecordingModel:
    """A model that runs each eval in turn, then answers."""
    script = [
        AIMessage("", tool_calls=[{"name": "eval", "args": {"code": c}, "id": f"c{i}"}])
        for i, c in enumerate(codes)
    ]
    return _RecordingModel(messages=iter([*script, AIMessage(content="done")]))


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


def _run(*codes: str, run_async: bool = False, **agent_kwargs: Any) -> Any:
    """Run evals in a deep agent over `_FILES` and return the last result."""
    middleware = CodeInterpreterMiddleware(
        ptc=_FS_TOOLS, **agent_kwargs.pop("middleware_kwargs", {})
    )
    agent = create_deep_agent(
        model=_model(*codes), middleware=[middleware], **agent_kwargs
    )
    payload = {"messages": [{"role": "user", "content": "go"}], "files": _files()}
    state = asyncio.run(agent.ainvoke(payload)) if run_async else agent.invoke(payload)
    return _results(state)[-1]


@pytest.mark.parametrize("run_async", [False, True], ids=["invoke", "ainvoke"])
def test_reads_whole_file_unformatted(*, run_async: bool) -> None:
    code = """JSON.stringify([
      JSON.parse(fs.readFileSync("/data.json", "utf8")).rows.length,
      JSON.parse(await fs.promises.readFile("data.json", "utf8")).rows.length,
    ])"""
    assert _run(code, run_async=run_async) == [150, 150]


def test_without_an_encoding_reads_bytes_that_decode_as_utf8(tmp_path: Path) -> None:
    (tmp_path / "bin.dat").write_bytes(b"a\r\nb\x00\xff")
    (tmp_path / "data.json").write_text('{"name": "hé"}', encoding="utf-8")
    code = """JSON.stringify([
      JSON.parse(fs.readFileSync("/data.json")).name,
      String(await fs.promises.readFile("/data.json")),
      fs.readFileSync("/bin.dat") instanceof Uint8Array,
      Array.from(fs.readFileSync("/bin.dat")),
      fs.readFileSync("/bin.dat", { encoding: "UTF-8" }) === "a\\r\\nb\\0\\ufffd",
      fs.readFileSync("/bin.dat", "base64"),
    ])"""
    expected = [
        "hé",
        '{"name": "hé"}',
        True,
        [97, 13, 10, 98, 0, 255],
        True,
        "YQ0KYgD/",
    ]
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    assert _run(code, backend=backend) == expected


def test_failures_are_node_errors(tmp_path: Path) -> None:
    (tmp_path / "dir").mkdir()
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "key.txt").write_text("k")
    (tmp_path / "large.txt").write_bytes(b"x" * (8 * 1024 * 1024 + 1))
    code = """
    const caught = (f) => {
      try { f() } catch (e) { return [e.name, e.code, e.message, e.syscall, e.path] }
    };
    const denied = await tools.readFile({ file_path: "/secret/key.txt" })
      .catch((e) => [e.name, e.code]);
    const invalid = await tools.readFile({}).catch((e) => [e.name, e.code]);
    JSON.stringify([
      caught(() => fs.readFileSync("/missing.txt")),
      caught(() => fs.readFileSync("/dir")).slice(1, 3),
      caught(() => fs.readFileSync("/secret/key.txt")).slice(1, 3),
      caught(() => fs.readFileSync("/large.txt")).slice(1, 2),
      caught(() => fs.readFileSync("/dir", "utf32")).slice(0, 2),
      caught(() => fs.readFileSync(42)).slice(0, 2),
      denied,
      invalid,
    ])"""
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    assert _run(code, backend=backend, permissions=_DENY_SECRET) == [
        [
            "Error",
            "ENOENT",
            "ENOENT: no such file or directory, open '/missing.txt'",
            "open",
            "/missing.txt",
        ],
        ["EISDIR", "EISDIR: illegal operation on a directory, open '/dir'"],
        ["EACCES", "EACCES: permission denied, open '/secret/key.txt'"],
        ["EIO"],
        ["TypeError", "ERR_INVALID_ARG_VALUE"],
        ["TypeError", "ERR_INVALID_ARG_TYPE"],
        ["ToolError", "EIO"],
        ["ToolError", "EIO"],
    ]


def test_fs_calls_count_toward_the_call_budget() -> None:
    code = 'for (let i = 0; i < 3; i++) fs.readFileSync("/docs/b.txt")'
    result = _run(code, middleware_kwargs={"max_ptc_calls": 2})
    assert 'type="PTCCallBudgetExceeded"' in result
    assert "function=fs.readFile" in result


@tool("read_file")
def _read_file_as_text(file_path: str) -> str:
    """Return the file as plain text, as a custom tool does."""
    return f"contents of {file_path}"


@pytest.mark.parametrize(
    ("ptc", "tools", "on", "prompt_text"),
    [
        (["read_file"], [_read_file_as_text], True, "fs.readFileSync(path"),
        (["read_file"], [], False, "no built-in filesystem"),
        (None, [_read_file_as_text], False, "no built-in filesystem"),
    ],
    ids=["listed", "listed-without-the-tool", "not-listed"],
)
def test_globals_prompt_and_schema_follow_ptc(
    ptc: list[str] | None, tools: list[Any], *, on: bool, prompt_text: str
) -> None:
    model = _model("JSON.stringify(typeof fs)")
    middleware = CodeInterpreterMiddleware(ptc=ptc)
    agent = create_agent(model, tools=tools, middleware=[middleware])
    state = agent.invoke({"messages": [{"role": "user", "content": "go"}]})
    assert _results(state) == ["object" if on else "undefined"]
    assert prompt_text in model.prompts[0]
    tool = middleware.tools[0]
    schema = tool.tool_call_schema.model_json_schema()["properties"]["code"]
    no_files = ("No filesystem" in tool.description, "No fs" in schema["description"])
    assert no_files == (ptc is None, ptc is None)


@tool("read_file", response_format="content_and_artifact")
def _read_file_with_own_artifact(file_path: str) -> tuple[str, dict[str, Any]]:
    """Return text with an artifact of its own, as a custom tool may."""
    return file_path, {"bytes": 1}


@pytest.mark.parametrize(
    ("tools", "message"),
    [
        ([_read_file_as_text], "ENOSYS: the read_file tool returned no raw result"),
        ([_read_file_with_own_artifact], "EIO: the read_file tool returned a raw"),
        ([], "ENOSYS: function not implemented; add 'read_file' to `ptc`, open '/a'"),
    ],
    ids=["custom-text", "custom-artifact", "no-longer-listed"],
)
def test_a_read_file_tool_fs_cannot_use_fails_with_a_node_error(
    tools: list[Any], message: str
) -> None:
    read = "try { fs.readFileSync('/a') } catch (e) { e.message }"
    assert _evals(("1", [_read_file_as_text]), (read, tools)).startswith(message)


async def test_without_the_tool_call_chain_fs_calls_the_tool_directly(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("direct")
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    file_tools = FilesystemMiddleware(backend=backend).tools
    middleware = CodeInterpreterMiddleware(ptc=_FS_TOOLS)
    eval_tool = middleware.tools[0]
    state = {"_quickjs_slot_id": "direct"}
    middleware._prepare_for_call(SimpleNamespace(tools=file_tools, state=state))
    runtime = ToolRuntime(
        state=state,
        context={},
        config={},
        stream_writer=lambda _: None,
        tools=[eval_tool, *file_tools],
        tool_call_id="outer",
        store=None,
    )
    code = """const missing = await fs.promises.readFile("/b.txt").catch((e) => e.code);
    JSON.stringify([fs.readFileSync("/a.txt", "utf8"), missing])"""
    assert eval_tool.coroutine is not None
    try:
        result = await eval_tool.coroutine(runtime=runtime, code=code)
    finally:
        middleware._registry.close()
    assert result.content == '<result>["direct","ENOENT"]</result>'


def test_a_fresh_process_restores_the_heap_and_fs_still_reads() -> None:
    first = """globalThis.cached = fs.readFileSync("/docs/b.txt", "utf8"); fs = 0;
    globalThis.__lc_fs_sync = () => [0, "fake"];
    const fake = { value: async () => [0, "fake"] };
    try { Object.defineProperty(globalThis, "__lc_fs_async", fake) } catch {}"""
    second = """JSON.stringify([
      cached,
      fs.readFileSync("/docs/a.md", "utf8").length,
      await fs.promises.readFile("/docs/b.txt", "utf8"),
    ])"""
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "t"}}
    for code, files in ((first, {"files": _files()}), (second, {})):
        agent = create_deep_agent(
            model=_model(code),
            middleware=[CodeInterpreterMiddleware(ptc=_FS_TOOLS)],
            checkpointer=saver,
        )
        state = agent.invoke(
            {"messages": [{"role": "user", "content": "go"}], **files}, config
        )
    assert _results(state)[-1] == ["plain", len(_FILES["/docs/a.md"]), "plain"]


def _evals(*evals: tuple[str, list[Any]], timeout: float = 5) -> str | None:
    """Run evals in one REPL, each with its bound tools; return the last result."""
    registry = _Registry(
        memory_limit=64 * 1024 * 1024,
        timeout=timeout,
        capture_console=True,
        max_stdout_chars=4000,
    )
    try:
        repl = registry.get("slot")
        return [repl.eval_sync(code, tools=tools) for code, tools in evals][-1].result
    finally:
        registry.close()


def test_code_that_declares_its_own_fs_keeps_running() -> None:
    listed = [_read_file_as_text]
    mine = "function fs() { return 'mine' }; 1"
    assert _evals((mine, listed), ("fs()", listed)) == "mine"


@tool("slow")
async def _slow() -> str:
    """Wait longer than the eval's timeout, which time awaiting async calls skips."""
    await asyncio.sleep(0.4)
    return "slow"


@pytest.mark.parametrize(
    "code",
    [
        "await tools.slow({}); READ",
        "(async () => { await tools.slow({}); return READ })()",
    ],
    ids=["top-level-await", "async-function"],
)
def test_time_awaiting_async_calls_does_not_count_against_a_sync_read(
    code: str,
) -> None:
    read = "(() => { try { fs.readFileSync('/a') } catch (e) { return e.code } })()"
    tools = [_slow, _read_file_as_text]
    assert _evals((code.replace("READ", read), tools), timeout=0.2) == "ENOSYS"


_notes: list[str] = []


@tool("note")
def _note(x: str) -> str:
    """Record a note."""
    _notes.append(x)
    return x


def test_a_call_left_queued_by_an_eval_that_returned_does_not_run_in_the_next() -> None:
    late = "tools.slow({}).then(() => tools.note({ x: 'late' })); 1"
    own = "await tools.note({ x: 'own' }); 2"
    tools = [_slow, _note]
    _notes.clear()
    assert _evals((late, tools), (own, tools)) == "2"
    assert _notes == ["own"]


_reads: list[str] = []


@tool("read_file")
def _blocked_read_file(file_path: str) -> str:
    """Finish twice the eval's timeout after the read starts."""
    time.sleep(0.4)
    _reads.append(file_path)
    return file_path


@pytest.mark.parametrize("run_async", [False, True], ids=["invoke", "ainvoke"])
def test_once_a_sync_read_times_out_no_call_from_code_runs(*, run_async: bool) -> None:
    code = """for (const read of [
      () => fs.readFileSync("/a"), () => fs.readFileSync("/b"),
      () => fs.promises.readFile("/c"), () => tools.readFile({ file_path: "/d" }),
    ]) { try { await read() } catch {} }"""
    middleware = CodeInterpreterMiddleware(ptc=["read_file"], timeout=0.2)
    model = _model(code, "1")
    agent = create_agent(model, tools=[_blocked_read_file], middleware=[middleware])
    _reads.clear()
    payload = {"messages": [{"role": "user", "content": "go"}]}
    state = asyncio.run(agent.ainvoke(payload)) if run_async else agent.invoke(payload)
    message = (
        "fs.readFileSync did not finish in time; the call may still have completed"
    )
    assert _results(state) == [f'<error type="Timeout">{message}</error>', 1]
    assert _reads == ["/a"]


class _CountingBackend(StateBackend):
    """State backend that records the file downloads behind raw reads."""

    downloads: ClassVar[list[str]] = []

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        self.downloads.extend(paths)
        return super().download_files(paths)


async def _answer_pauses(
    code: str,
    gated: str,
    decision: dict[str, Any],
    *,
    durability: Literal["async", "exit"] = "async",
) -> tuple[list[Any], list[Any]]:
    """Run `code` where `gated` calls need per-call approval, giving each `decision`."""
    agent = create_agent(
        model=_model(code),
        middleware=[
            CodeInterpreterMiddleware(ptc=_FS_TOOLS),
            FilesystemMiddleware(backend=_CountingBackend()),
            HumanInTheLoopMiddleware(
                interrupt_on={gated: True}, interrupt_mode="per_call"
            ),
        ],
        checkpointer=InMemorySaver(),
    )
    out, pauses = await answer_pauses(
        agent,
        lambda _: decision,
        invoke_mode="invoke",
        durability=durability,
        max_pauses=3,
        payload={"messages": [{"role": "user", "content": "go"}], "files": _files()},
    )
    return _results(out.value), [[(p["name"], p["args"]) for p in ps] for ps in pauses]


@pytest.mark.parametrize(
    "call",
    [
        "fs.readFileSync('/docs/b.txt', 'utf8')",
        "await fs.promises.readFile('/docs/b.txt', 'utf8')",
        "await tools.readFile({ file_path: '/docs/b.txt' })",
    ],
)
async def test_an_interrupt_on_rule_pauses_fs_reads_like_tools_reads(call: str) -> None:
    code = f"""let r;
    try {{ r = {call} }} catch (e) {{ r = [e.name, e.code, e.message] }}
    JSON.stringify(r)"""
    approved = await _answer_pauses(code, "read_file", {"type": "approve"})
    rejected = await _answer_pauses(
        code, "read_file", {"type": "reject", "message": "no"}
    )
    pause = [[("read_file", {"file_path": "/docs/b.txt"})]]
    assert approved[1] == rejected[1] == pause
    assert "plain" in approved[0][0]
    name, code_, message = rejected[0][0]
    assert name == ("ToolError" if "tools." in call else "Error")
    assert code_ == "EACCES"
    assert "User rejected the tool call for `read_file` with reason: no" in message


@pytest.mark.parametrize("durability", ["async", "exit"])
async def test_finished_fs_calls_are_not_run_again_after_a_resume(
    durability: Literal["async", "exit"],
) -> None:
    code = """const before = String.fromCharCode(...fs.readFileSync("/docs/b.txt"));
    const listed = await tools.ls({ path: "/docs" });
    const after = await fs.promises.readFile("/docs/a.md", "utf8");
    JSON.stringify([before, listed.includes("a.md"), after.length])"""
    _CountingBackend.downloads.clear()
    results, pauses = await _answer_pauses(
        code, "ls", {"type": "approve"}, durability=durability
    )
    assert pauses == [[("ls", {"path": "/docs"})]]
    assert results == [["plain", True, len(_FILES["/docs/a.md"])]]
    assert _CountingBackend.downloads == ["/docs/b.txt", "/docs/a.md"]
