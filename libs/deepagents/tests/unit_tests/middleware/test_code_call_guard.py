import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import RAW_FILE_RESULT_CONFIG_KEY
from deepagents.graph import create_deep_agent
from deepagents.middleware import CodeCallGuardMiddleware
from deepagents.middleware.code_call_guard import _guard_calls_from_code
from deepagents.middleware.filesystem import FilesystemPermission
from tests.unit_tests.chat_model import GenericFakeChatModel


@tool
def run_code(calls: list[dict[str, Any]]) -> str:
    """Run tool calls from code."""
    return f"{len(calls)} calls"


class _CodeMode(AgentMiddleware[Any, Any, Any]):
    def __init__(self, *, raw: bool = False) -> None:
        super().__init__()
        self.raw = raw
        self.results: list[Any] = []

    def _nested(self, request: ToolCallRequest, seq: int, call: dict[str, Any]) -> ToolCallRequest:
        call_id = f"{request.tool_call['id']}.code{seq}"
        config = request.runtime.config
        if self.raw:
            raw = {"tool_call_id": call_id, "max_bytes": 1024}
            config = {**config, "configurable": {**config.get("configurable", {}), RAW_FILE_RESULT_CONFIG_KEY: raw}}
        tool = next(t for t in request.runtime.tools if t.name == call["name"])
        return replace(
            request,
            tool_call={"name": call["name"], "args": call["args"], "id": call_id, "type": "tool_call"},
            tool=tool,
            runtime=replace(request.runtime, tool_call_id=call_id, config=config),
        )

    def _summary(self, request: ToolCallRequest, results: list[ToolMessage | Command[Any]]) -> ToolMessage:
        self.results = results
        return ToolMessage(content=f"{len(results)} calls", name="run_code", tool_call_id=request.tool_call["id"])

    def wrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]]
    ) -> ToolMessage | Command[Any]:
        if request.tool_call["name"] != "run_code":
            return handler(request)
        calls = request.tool_call["args"]["calls"]
        return self._summary(request, [handler(self._nested(request, i, c)) for i, c in enumerate(calls)])

    async def awrap_tool_call(
        self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]]
    ) -> ToolMessage | Command[Any]:
        if request.tool_call["name"] != "run_code":
            return await handler(request)
        calls = request.tool_call["args"]["calls"]
        return self._summary(request, [await handler(self._nested(request, i, c)) for i, c in enumerate(calls)])


def _model(*tool_calls: dict[str, Any]) -> GenericFakeChatModel:
    calls = [{**call, "id": f"call_{i}", "type": "tool_call"} for i, call in enumerate(tool_calls)]
    return GenericFakeChatModel(messages=iter([AIMessage(content="", tool_calls=calls), AIMessage(content="done")]))


def _code(*calls: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    return {"name": "run_code", "args": {"calls": [{"name": name, "args": args} for name, args in calls]}}


def _write(path: str) -> tuple[str, dict[str, Any]]:
    return ("write_file", {"file_path": path, "content": "x"})


def _run(tmp_path: Path, model: GenericFakeChatModel, *, run_async: bool = False, **kwargs: Any) -> dict[str, Any]:
    (tmp_path / "notes.md").write_text("hello")
    agent = create_deep_agent(
        model=model,
        tools=[run_code],
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        checkpointer=InMemorySaver(),
        **kwargs,
    )
    payload = {"messages": [HumanMessage("go")]}
    config = {"configurable": {"thread_id": "t"}}
    return asyncio.run(agent.ainvoke(payload, config)) if run_async else agent.invoke(payload, config)


@pytest.mark.parametrize("run_async", [False, True], ids=["invoke", "ainvoke"])
def test_gated_call_from_code_is_refused_and_ungated_calls_run(tmp_path: Path, *, run_async: bool) -> None:
    code_mode = _CodeMode()
    model = _model(_code(_write("/a.txt"), ("read_file", {"file_path": "/notes.md"})))

    result = _run(tmp_path, model, run_async=run_async, middleware=[code_mode], interrupt_on={"write_file": True})

    write, read = code_mode.results
    assert "__interrupt__" not in result
    assert write.status == "error"
    assert write.content == "Error: `write_file` needs human approval, which a call from code cannot request. Call `write_file` directly instead."
    assert not (tmp_path / "a.txt").exists()
    assert read.status == "success"
    assert "hello" in read.content


@pytest.mark.parametrize("raw", [True, False], ids=["raw-result-request", "no-request"])
def test_refusal_reports_permission_denied_to_a_raw_result_request(tmp_path: Path, *, raw: bool) -> None:
    code_mode = _CodeMode(raw=raw)

    _run(tmp_path, _model(_code(_write("/a.txt"))), middleware=[code_mode], interrupt_on={"write_file": True})

    (write,) = code_mode.results
    assert write.status == "error"
    assert write.artifact == ({"code": "permission_denied"} if raw else None)


def test_path_scoped_interrupt_rule_refuses_only_matching_paths(tmp_path: Path) -> None:
    code_mode = _CodeMode()
    rule = FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="interrupt")

    result = _run(tmp_path, _model(_code(_write("/secrets/key.txt"), _write("/work/out.txt"))), middleware=[code_mode], permissions=[rule])

    secret, work = code_mode.results
    assert "__interrupt__" not in result
    assert secret.status == "error"
    assert not (tmp_path / "secrets" / "key.txt").exists()
    assert work.status == "success"
    assert (tmp_path / "work" / "out.txt").read_text() == "x"


def test_per_call_approval_does_not_pause_calls_from_code(tmp_path: Path) -> None:
    code_mode = _CodeMode()
    approval = HumanInTheLoopMiddleware(interrupt_on={"write_file": True}, interrupt_mode="per_call")

    result = _run(tmp_path, _model(_code(_write("/a.txt"))), middleware=[code_mode, approval])

    (write,) = code_mode.results
    assert "__interrupt__" not in result
    assert "needs human approval" in write.content
    assert not (tmp_path / "a.txt").exists()


@pytest.mark.parametrize("mode", ["batched", "per_call"])
def test_model_call_to_a_gated_tool_still_pauses_for_approval(tmp_path: Path, mode: str) -> None:
    approval = HumanInTheLoopMiddleware(interrupt_on={"write_file": True}, interrupt_mode=mode)
    model = _model({"name": "write_file", "args": {"file_path": "/a.txt", "content": "x"}})

    result = _run(tmp_path, model, middleware=[_CodeMode(), approval])

    assert "__interrupt__" in result
    assert not (tmp_path / "a.txt").exists()


def test_approval_middleware_subclass_is_guarded(tmp_path: Path) -> None:
    class _Approval(HumanInTheLoopMiddleware):
        pass

    code_mode = _CodeMode()

    result = _run(tmp_path, _model(_code(_write("/a.txt"))), middleware=[code_mode, _Approval(interrupt_on={"write_file": True})])

    assert "__interrupt__" not in result
    assert code_mode.results[0].status == "error"
    assert not (tmp_path / "a.txt").exists()


def test_a_hand_placed_guard_is_not_doubled(tmp_path: Path) -> None:
    code_mode = _CodeMode()
    approval = HumanInTheLoopMiddleware(interrupt_on={"write_file": True})
    stack = [code_mode, CodeCallGuardMiddleware(approval), approval]

    result = _run(tmp_path, _model(_code(_write("/a.txt"))), middleware=stack)

    assert _guard_calls_from_code(stack) == stack
    assert "__interrupt__" not in result
    assert code_mode.results[0].status == "error"


def test_a_dispatcher_listed_after_the_approval_middleware_is_not_guarded(tmp_path: Path) -> None:
    code_mode = _CodeMode()
    approval = HumanInTheLoopMiddleware(interrupt_on={"write_file": True})

    _run(tmp_path, _model(_code(_write("/a.txt"))), middleware=[approval, code_mode])

    assert code_mode.results[0].status == "success"
    assert (tmp_path / "a.txt").read_text() == "x"


def test_subagent_interrupt_on_refuses_its_calls_from_code(tmp_path: Path) -> None:
    code_mode = _CodeMode()
    worker = {
        "name": "worker",
        "description": "Runs code.",
        "system_prompt": "Run code.",
        "model": _model(_code(_write("/a.txt"))),
        "tools": [run_code],
        "middleware": [code_mode],
        "interrupt_on": {"write_file": True},
    }
    parent = _model({"name": "task", "args": {"description": "write", "subagent_type": "worker"}})

    result = _run(tmp_path, parent, subagents=[worker])

    assert "__interrupt__" not in result
    assert code_mode.results[0].status == "error"
    assert not (tmp_path / "a.txt").exists()
