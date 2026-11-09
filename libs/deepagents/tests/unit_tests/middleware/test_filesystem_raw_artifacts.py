import asyncio
import errno
import os
from collections.abc import Awaitable, Callable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import ToolRuntime
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.types import Command

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import RAW_FILE_RESULT_CONFIG_KEY, DeleteResult, FileDownloadResponse, LsResult, RawFileResultRequest, WriteResult
from deepagents.backends.utils import create_file_data
from deepagents.middleware.filesystem import FilesystemMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel

_LONG = "".join(f"line {i}\r\n" for i in range(300))
_BINARY = b"a\r\nb\x00\xff"
_RAW = {"tool_call_id": "c1"}
_NEEDS_OS_PERMISSIONS = pytest.mark.skipif(not (hasattr(os, "geteuid") and os.geteuid() != 0), reason="needs a non-root POSIX user")


@pytest.fixture
def tools(tmp_path: Path) -> Iterator[dict[str, BaseTool]]:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("alpha")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "key.txt").write_text("k")
    (tmp_path / "link.txt").symlink_to(tmp_path / "secret" / "key.txt")
    (tmp_path / "data.bin").write_bytes(_BINARY)
    (tmp_path / "long.txt").write_bytes(_LONG.encode())
    (tmp_path / "readonly").mkdir()
    (tmp_path / "readonly" / "kept.txt").write_text("k")
    (tmp_path / "readonly").chmod(0o500)
    rules = [
        FilesystemPermission(operations=["read"], paths=["/secret", "/secret/**"], mode="deny"),
        FilesystemPermission(operations=["write"], paths=["/locked", "/locked/**"], mode="deny"),
    ]
    middleware = FilesystemMiddleware(backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True), permissions=rules)
    yield {t.name: t for t in middleware.tools}
    (tmp_path / "readonly").chmod(0o700)


def _call(tool: BaseTool, args: dict[str, Any], *, run_async: bool, request: Mapping[str, Any] | str | None = _RAW) -> ToolMessage:
    if isinstance(request, Mapping):
        request = {"tool_name": tool.name, **request}
    configurable = {} if request is None else {RAW_FILE_RESULT_CONFIG_KEY: request}
    runtime = ToolRuntime(state={}, context=None, tool_call_id="c1", store=None, stream_writer=lambda _: None, config={"configurable": configurable})
    payload = {**args, "runtime": runtime}
    return asyncio.run(tool.ainvoke(payload)) if run_async else tool.invoke(payload)


@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])
def test_read_file_artifact_is_the_whole_file_byte_for_byte(tools: dict[str, BaseTool], *, run_async: bool) -> None:
    binary = _call(tools["read_file"], {"file_path": "/data.bin"}, run_async=run_async)
    long = _call(tools["read_file"], {"file_path": "/long.txt", "limit": 5}, run_async=run_async)
    assert (binary.status, binary.artifact) == ("success", {"content": _BINARY})
    assert (long.status, long.artifact) == ("success", {"content": _LONG.encode()})


@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])
def test_ls_artifact_lists_the_permitted_entries(tools: dict[str, BaseTool], *, run_async: bool) -> None:
    ls = _call(tools["ls"], {"path": "/"}, run_async=run_async)
    assert sorted(ls.artifact["entries"], key=lambda e: e["path"]) == [
        {"path": "/data.bin", "is_dir": False},
        {"path": "/docs/", "is_dir": True},
        {"path": "/long.txt", "is_dir": False},
        {"path": "/readonly/", "is_dir": True},
    ]


@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("name", "args", "code"),
    [
        ("read_file", {"file_path": "/secret/key.txt"}, "permission_denied"),
        ("read_file", {"file_path": "/link.txt"}, "permission_denied"),
        ("read_file", {"file_path": "/missing.txt"}, "file_not_found"),
        ("read_file", {"file_path": "/docs"}, "is_directory"),
        ("read_file", {"file_path": "/../etc/passwd"}, "invalid_path"),
        ("ls", {"path": "/secret"}, "permission_denied"),
        ("ls", {"path": "/../x"}, "invalid_path"),
        ("ls", {"path": "/missing"}, "file_not_found"),
        ("ls", {"path": "/docs/a.md"}, "not_a_directory"),
    ],
)
def test_read_failures_carry_a_code(tools: dict[str, BaseTool], name: str, args: dict[str, Any], code: str | None, *, run_async: bool) -> None:
    message = _call(tools[name], args, run_async=run_async)
    assert (message.status, message.artifact) == ("error", {"code": code})


@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("name", "args", "code"),
    [
        ("write_file", {"file_path": "/../x.txt", "content": "x"}, "invalid_path"),
        ("write_file", {"file_path": "/locked/a.txt", "content": "x"}, "permission_denied"),
        ("write_file", {"file_path": "/docs", "content": "x"}, "is_directory"),
        pytest.param(
            "write_file",
            {"file_path": "/readonly/new.txt", "content": "x"},
            "permission_denied",
            marks=_NEEDS_OS_PERMISSIONS,
            id="write_file-os-permission",
        ),
        ("delete", {"file_path": "/../x.txt"}, "invalid_path"),
        ("delete", {"file_path": "/locked/a.txt"}, "permission_denied"),
        ("delete", {"file_path": "/missing.txt"}, None),
        pytest.param("delete", {"file_path": "/readonly/kept.txt"}, "permission_denied", marks=_NEEDS_OS_PERMISSIONS, id="delete-os-permission"),
    ],
)
def test_mutation_failures_carry_a_code_and_keep_the_model_text(
    tools: dict[str, BaseTool], name: str, args: dict[str, Any], code: str | None, *, run_async: bool
) -> None:
    message = _call(tools[name], args, run_async=run_async)
    model_view = _call(tools[name], args, run_async=run_async, request=None)
    assert (message.status, message.artifact) == ("error", {"code": code})
    assert (model_view.status, model_view.content, model_view.artifact) == ("error", message.content, None)


@pytest.mark.parametrize(
    ("name", "args", "request_", "content"),
    [
        ("ls", {"path": "/missing"}, None, "Error: cannot list '/missing': not found"),
        ("ls", {"path": "/docs/a.md"}, None, "Error: cannot list '/docs/a.md': not a directory"),
        ("write_file", {"file_path": "/docs", "content": "x"}, None, "Error: cannot write '/docs': is a directory"),
        ("read_file", {"file_path": "/missing.txt"}, _RAW, "Error: cannot read '/missing.txt': not found"),
    ],
)
def test_a_backend_error_code_is_spelled_out_for_the_model(
    tools: dict[str, BaseTool], name: str, args: dict[str, Any], request_: Mapping[str, Any] | None, content: str
) -> None:
    assert _call(tools[name], args, run_async=False, request=request_).content == content


@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "request_",
    [
        {"tool_call_id": "another call"},
        {"tool_call_id": "c1", "tool_name": "glob"},
        "c1",
        None,
    ],
    ids=["another-call", "another-tool", "not-a-request", "no-request"],
)
@pytest.mark.parametrize(
    ("name", "args"),
    [("read_file", {"file_path": "/long.txt"}), ("read_file", {"file_path": "/missing.txt"}), ("ls", {"path": "/"})],
)
def test_a_call_the_request_does_not_name_returns_the_model_text(
    tools: dict[str, BaseTool], name: str, args: dict[str, Any], request_: Mapping[str, Any] | str | None, *, run_async: bool
) -> None:
    message = _call(tools[name], args, run_async=run_async, request=request_)
    model_view = _call(tools[name], args, run_async=run_async, request=None)
    assert (message.status, message.content, message.artifact) == (model_view.status, model_view.content, None)


def test_a_runtime_without_config_reads_like_a_model_call(tools: dict[str, BaseTool]) -> None:
    runtime = SimpleNamespace(tool_call_id="c1", state={})
    message = tools["read_file"].func(file_path="/long.txt", runtime=runtime)
    assert (message.status, message.artifact) == ("success", None)
    assert message.content.startswith("@@ lines 1-100 of 300")


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX FIFOs")
@pytest.mark.timeout(10)
def test_a_raw_read_of_a_fifo_fails_without_waiting_for_a_writer(tools: dict[str, BaseTool], tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "pipe")

    message = _call(tools["read_file"], {"file_path": "/pipe"}, run_async=False)

    assert (message.status, message.artifact) == ("error", {"code": "file_not_found"})


class _FailingDisk(FilesystemBackend):
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        raise OSError(errno.EIO, "Input/output error")


@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])
def test_an_unmapped_os_error_fails_the_raw_read_without_a_code(tmp_path: Path, *, run_async: bool) -> None:
    middleware = FilesystemMiddleware(backend=_FailingDisk(root_dir=tmp_path, virtual_mode=True))
    tool = next(t for t in middleware.tools if t.name == "read_file")

    message = _call(tool, {"file_path": "/a.txt"}, run_async=run_async)

    assert (message.status, message.artifact) == ("error", {"code": None})
    assert "Input/output error" in message.content


class _UncodedFailures(FilesystemBackend):
    def ls(self, path: str) -> LsResult:
        return LsResult(error=f"cannot list {path}")

    def write(self, file_path: str, content: str) -> WriteResult:
        return WriteResult(error=f"cannot write {file_path}")

    def delete(self, file_path: str) -> DeleteResult:
        return DeleteResult(error=f"cannot delete {file_path}")


@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("name", "args", "content"),
    [
        ("ls", {"path": "/"}, "Error: cannot list /"),
        ("write_file", {"file_path": "/a.txt", "content": "x"}, "cannot write /a.txt"),
        ("delete", {"file_path": "/a.txt"}, "cannot delete /a.txt"),
    ],
)
def test_a_backend_error_without_a_code_keeps_its_text_and_carries_an_artifact(
    tmp_path: Path, name: str, args: dict[str, Any], content: str, *, run_async: bool
) -> None:
    middleware = FilesystemMiddleware(backend=_UncodedFailures(root_dir=tmp_path, virtual_mode=True))
    tool = next(t for t in middleware.tools if t.name == name)

    message = _call(tool, args, run_async=run_async)

    assert (message.status, message.content, message.artifact) == ("error", content, {"code": None})


class _RequestRawResult(AgentMiddleware):
    def __init__(self, names: Callable[[str], str]) -> None:
        super().__init__()
        self.names = names

    def _with_request(self, request: ToolCallRequest) -> ToolCallRequest:
        config = request.runtime.config
        call = request.tool_call
        raw: RawFileResultRequest = {"tool_call_id": self.names(call["id"]), "tool_name": call["name"]}
        configurable = {**config.get("configurable", {}), RAW_FILE_RESULT_CONFIG_KEY: raw}
        return request.override(runtime=replace(request.runtime, config={**config, "configurable": configurable}))

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
        return handler(self._with_request(request))

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
    ) -> ToolMessage | Command:
        return await handler(self._with_request(request))


def _run_state_agent(calls: list[dict[str, Any]], *, run_async: bool, middleware: list[AgentMiddleware]) -> dict[str, ToolMessage]:
    tool_calls = [{**call, "type": "tool_call"} for call in calls]
    model = GenericFakeChatModel(messages=iter([AIMessage(content="", tool_calls=tool_calls), AIMessage(content="done")]))
    agent = create_deep_agent(model=model, middleware=middleware)
    files = {"/notes.txt": create_file_data("hello"), "/docs/a.md": create_file_data("alpha"), "/long.txt": create_file_data(_LONG)}
    payload = {"messages": [{"role": "user", "content": "go"}], "files": files}
    state = asyncio.run(agent.ainvoke(payload)) if run_async else agent.invoke(payload)
    return {m.tool_call_id: m for m in state["messages"] if isinstance(m, ToolMessage)}


@pytest.mark.parametrize("run_async", [False, True], ids=["invoke", "ainvoke"])
def test_tool_node_delivers_the_request_to_the_named_call_and_ignores_model_args(*, run_async: bool) -> None:
    forged = {"raw_artifact": True, RAW_FILE_RESULT_CONFIG_KEY: {"tool_call_id": "model"}}
    calls = [
        {"name": "read_file", "args": {"file_path": "/long.txt", **forged}, "id": "model"},
        {"name": "read_file", "args": {"file_path": "/long.txt"}, "id": "code"},
    ]
    results = _run_state_agent(calls, run_async=run_async, middleware=[_RequestRawResult(lambda _: "code")])
    assert results["model"].artifact is None
    assert "line 0" in results["model"].content
    assert "line 299" not in results["model"].content
    assert results["code"].artifact == {"content": _LONG.encode()}


_STATE_CALLS: list[dict[str, Any]] = [
    {"id": "write-invalid", "name": "write_file", "args": {"file_path": "/../x.txt", "content": "x"}},
    {"id": "delete-missing", "name": "delete", "args": {"file_path": "/gone.txt"}},
    {"id": "ls", "name": "ls", "args": {"path": "/"}},
]


@pytest.mark.parametrize("run_async", [False, True], ids=["invoke", "ainvoke"])
def test_state_backend_failures_carry_a_code_and_ls_lists_entries(*, run_async: bool) -> None:
    raw = _run_state_agent(_STATE_CALLS, run_async=run_async, middleware=[_RequestRawResult(lambda call_id: call_id)])
    model_view = _run_state_agent(_STATE_CALLS, run_async=run_async, middleware=[])
    codes = {call_id: raw[call_id].artifact for call_id in ("write-invalid", "delete-missing")}
    assert codes == {"write-invalid": {"code": "invalid_path"}, "delete-missing": {"code": None}}
    assert all((raw[k].status, raw[k].content) == (model_view[k].status, model_view[k].content) for k in codes)
    assert sorted(raw["ls"].artifact["entries"], key=lambda e: e["path"]) == [
        {"path": "/docs/", "is_dir": True},
        {"path": "/long.txt", "is_dir": False},
        {"path": "/notes.txt", "is_dir": False},
    ]
