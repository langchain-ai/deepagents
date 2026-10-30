"""`read_file`, `ls`, and `glob` put raw results on `ToolMessage.artifact` for the call a config request names."""

import asyncio
from collections.abc import Awaitable, Callable
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
_RAW: RawFileResultRequest = {"tool_call_id": "c1", "max_bytes": None}


@pytest.fixture
def tools(tmp_path: Path) -> dict[str, BaseTool]:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("a")
    (tmp_path / "secret").mkdir()
    (tmp_path / "secret" / "key.txt").write_text("k")
    (tmp_path / "link.txt").symlink_to(tmp_path / "secret" / "key.txt")
    (tmp_path / "data.bin").write_bytes(_BINARY)
    (tmp_path / "long.txt").write_bytes(_LONG.encode())
    rules = [FilesystemPermission(operations=["read"], paths=["/secret", "/secret/**"], mode="deny")]
    middleware = FilesystemMiddleware(backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True), permissions=rules)
    return {t.name: t for t in middleware.tools}


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
def test_listings_carry_permitted_entries_and_keep_the_model_text(tools: dict[str, BaseTool], *, run_async: bool) -> None:
    ls_args = {"path": "/"}
    glob_args = {"pattern": "**/*"}
    ls = _call(tools["ls"], ls_args, run_async=run_async)
    globbed = _call(tools["glob"], glob_args, run_async=run_async)
    assert sorted(ls.artifact["entries"], key=lambda e: e["path"]) == [
        {"path": "/data.bin", "is_dir": False},
        {"path": "/docs/", "is_dir": True},
        {"path": "/long.txt", "is_dir": False},
    ]
    assert (sorted(globbed.artifact["paths"]), globbed.artifact["truncated"]) == (["/data.bin", "/docs/a.md", "/long.txt"], False)
    for message, name, args in ((ls, "ls", ls_args), (globbed, "glob", glob_args)):
        model_view = _call(tools[name], args, run_async=run_async, request=None)
        assert (message.content, model_view.artifact) == (model_view.content, None)


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
        ("ls", {"path": "/missing"}, None),
        ("glob", {"pattern": "*", "path": "/secret"}, "permission_denied"),
        ("glob", {"pattern": "*", "path": "/../x"}, "invalid_path"),
    ],
)
def test_failures_carry_a_code(tools: dict[str, BaseTool], name: str, args: dict[str, Any], code: str | None, *, run_async: bool) -> None:
    message = _call(tools[name], args, run_async=run_async)
    assert (message.status, message.artifact) == ("error", {"code": code})


@pytest.mark.parametrize("run_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    "request_",
    [{"tool_call_id": "another call", "max_bytes": None}, None],
    ids=["another-call", "no-request"],
)
@pytest.mark.parametrize(
    ("name", "args"),
    [("read_file", {"file_path": "/long.txt"}), ("read_file", {"file_path": "/missing.txt"}), ("ls", {"path": "/"}), ("glob", {"pattern": "**/*"})],
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


class _RequestRawResultFor(AgentMiddleware):
    """Puts a raw-result request naming one tool call on every tool call's runtime config, as inherited config would."""

    def __init__(self, tool_call_id: str) -> None:
        super().__init__()
        self.request: RawFileResultRequest = {"tool_call_id": tool_call_id, "max_bytes": None}

    def _with_request(self, request: ToolCallRequest) -> ToolCallRequest:
        config = request.runtime.config
        configurable = {**config.get("configurable", {}), RAW_FILE_RESULT_CONFIG_KEY: self.request}
        return request.override(runtime=replace(request.runtime, config={**config, "configurable": configurable}))

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command]) -> ToolMessage | Command:
        return handler(self._with_request(request))

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
    ) -> ToolMessage | Command:
        return await handler(self._with_request(request))


@pytest.mark.parametrize("run_async", [False, True], ids=["invoke", "ainvoke"])
def test_tool_node_delivers_the_request_to_the_named_call_and_ignores_model_args(*, run_async: bool) -> None:
    forged = {"raw_artifact": True, RAW_FILE_RESULT_CONFIG_KEY: {"tool_call_id": "model", "max_bytes": None}}
    calls = [
        {"name": "read_file", "args": {"file_path": "/long.txt", **forged}, "id": "model", "type": "tool_call"},
        {"name": "read_file", "args": {"file_path": "/long.txt"}, "id": "code", "type": "tool_call"},
    ]
    model = GenericFakeChatModel(messages=iter([AIMessage(content="", tool_calls=calls), AIMessage(content="done")]))
    agent = create_deep_agent(model=model, middleware=[_RequestRawResultFor("code")])
    payload = {"messages": [{"role": "user", "content": "go"}], "files": {"/long.txt": create_file_data(_LONG)}}
    state = asyncio.run(agent.ainvoke(payload)) if run_async else agent.invoke(payload)
    results = {m.tool_call_id: m for m in state["messages"] if isinstance(m, ToolMessage)}
    assert results["model"].artifact is None
    assert "line 0" in results["model"].content
    assert "line 299" not in results["model"].content
    assert results["code"].artifact == {"content": _LONG.encode()}
