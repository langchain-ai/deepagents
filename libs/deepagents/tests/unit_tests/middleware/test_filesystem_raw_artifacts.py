import asyncio
import os
from collections.abc import Awaitable, Callable, Iterator
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
from deepagents.backends.protocol import RAW_FILE_RESULT_CONFIG_KEY, RawFileResultRequest
from deepagents.backends.utils import create_file_data
from deepagents.middleware.filesystem import FilesystemMiddleware
from tests.unit_tests.chat_model import GenericFakeChatModel

_LONG = "".join(f"line {i}\r\n" for i in range(300))
_BINARY = b"a\r\nb\x00\xff"
_MAX_BYTES = 1 << 20
_RAW: RawFileResultRequest = {"tool_call_id": "c1", "max_bytes": _MAX_BYTES}
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


def _call(tool: BaseTool, args: dict[str, Any], *, run_async: bool, request: RawFileResultRequest | None = _RAW) -> ToolMessage:
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
        ("delete", {"file_path": "/missing.txt"}, "file_not_found"),
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


@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "request_",
    [{"tool_call_id": "another call", "max_bytes": _MAX_BYTES}, None],
    ids=["another-call", "no-request"],
)
@pytest.mark.parametrize(
    ("name", "args"),
    [("read_file", {"file_path": "/long.txt"}), ("read_file", {"file_path": "/missing.txt"}), ("ls", {"path": "/"})],
)
def test_a_call_the_request_does_not_name_returns_the_model_text(
    tools: dict[str, BaseTool], name: str, args: dict[str, Any], request_: RawFileResultRequest | None, *, run_async: bool
) -> None:
    message = _call(tools[name], args, run_async=run_async, request=request_)
    model_view = _call(tools[name], args, run_async=run_async, request=None)
    assert (message.status, message.content, message.artifact) == (model_view.status, model_view.content, None)


def test_a_runtime_without_config_reads_like_a_model_call(tools: dict[str, BaseTool]) -> None:
    runtime = SimpleNamespace(tool_call_id="c1", state={})
    message = tools["read_file"].func(file_path="/long.txt", runtime=runtime)
    assert (message.status, message.artifact) == ("success", None)
    assert message.content.startswith("@@ lines 1-100 of 300")


@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])
def test_max_bytes_keeps_a_larger_file_off_the_artifact(tools: dict[str, BaseTool], *, run_async: bool) -> None:
    def read(max_bytes: int) -> ToolMessage:
        request: RawFileResultRequest = {"tool_call_id": "c1", "max_bytes": max_bytes}
        return _call(tools["read_file"], {"file_path": "/data.bin"}, run_async=run_async, request=request)

    at_limit, over_limit = read(len(_BINARY)), read(len(_BINARY) - 1)
    assert (at_limit.status, at_limit.artifact) == ("success", {"content": _BINARY})
    assert (over_limit.status, over_limit.artifact) == ("error", {"code": "file_too_large"})
    assert f"{len(_BINARY)} bytes" in over_limit.content


class _RequestRawResult(AgentMiddleware):
    def __init__(self, names: Callable[[str], str]) -> None:
        super().__init__()
        self.names = names

    def _with_request(self, request: ToolCallRequest) -> ToolCallRequest:
        config = request.runtime.config
        raw: RawFileResultRequest = {"tool_call_id": self.names(request.tool_call["id"]), "max_bytes": _MAX_BYTES}
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
    forged = {"raw_artifact": True, RAW_FILE_RESULT_CONFIG_KEY: {"tool_call_id": "model", "max_bytes": _MAX_BYTES}}
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
    assert codes == {"write-invalid": {"code": "invalid_path"}, "delete-missing": {"code": "file_not_found"}}
    assert all((raw[k].status, raw[k].content) == (model_view[k].status, model_view[k].content) for k in codes)
    assert sorted(raw["ls"].artifact["entries"], key=lambda e: e["path"]) == [
        {"path": "/docs/", "is_dir": True},
        {"path": "/long.txt", "is_dir": False},
        {"path": "/notes.txt", "is_dir": False},
    ]
