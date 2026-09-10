"""Exercise approval policy boundaries through real, network-free runtime graphs."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from deepagents_talon.interfaces import AgentRequest, ToolApprovalRequest
from deepagents_talon.mcp_config import MCPConfigStore
from deepagents_talon.runtime import DeepAgentRuntime
from deepagents_talon.tool_approvals import (
    ACTIVE_APPROVALS,
    APPROVAL_OPERATOR,
    ApprovalSnapshot,
    ToolApprovalStore,
)

if TYPE_CHECKING:
    from pathlib import Path


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, _tools: object, **_kwargs: object) -> ToolModel:
        return self


def call(name: str, **args: object) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "id": name, "args": args}])


def make_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: ToolModel
) -> DeepAgentRuntime:
    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: model)
    return DeepAgentRuntime(
        model="test:approval",
        assistant_dir=tmp_path,
        approval_store=ToolApprovalStore(tmp_path / "tools.json"),
        include_web_tools=False,
        skills=(),
        memory=(),
        max_retries=1,
    )


async def outputs(runtime: DeepAgentRuntime, chat: str, name: str) -> list[dict[str, object]]:
    state = await runtime._graph.aget_state({"configurable": {"thread_id": chat}})
    return [
        json.loads(message.content)
        for message in state.values["messages"]
        if isinstance(message, ToolMessage) and message.name == name and message.status != "error"
    ]


@pytest.mark.parametrize("decision", ["approve", "reject"])
async def test_self_disable_uses_pre_edit_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, decision: str
) -> None:
    model = ToolModel(responses=[AIMessage(content="placeholder")])
    runtime = make_runtime(tmp_path, monkeypatch, model)
    assert not (tmp_path / "tools.json").exists()
    await runtime.start()
    original = (tmp_path / "tools.json").read_bytes()
    active = runtime.approval_store.read()
    update = call(
        "update_tool_approvals",
        updates={"update_tool_approvals": False},
        expected_revision=active.revision,
    )
    model.responses = [
        update,
        call("get_tool_approvals"),
        call(
            "update_tool_approvals",
            updates={"update_tool_approvals": False},
            expected_revision=active.revision,
        ),
        AIMessage(content="done"),
    ]
    approvals = []

    async def decide(request: ToolApprovalRequest) -> str:
        approvals.extend(item["name"] for item in request.action_requests)
        assert ACTIVE_APPROVALS.get() == active
        assert APPROVAL_OPERATOR.get() is True
        if len(approvals) == 1:
            assert (tmp_path / "tools.json").read_bytes() == original
        return decision

    try:
        result = await runtime.invoke(
            AgentRequest(
                "first",
                "disable approval",
                metadata={"tool_approval_operator": True},
                approval_handler=decide,
            )
        )
        assert result.text == "done"
        assert approvals == ["update_tool_approvals", "update_tool_approvals"]
        view = (await outputs(runtime, "first", "get_tool_approvals"))[0]
        assert view["active_revision"] == active.revision
        assert view["active_tools"] == dict(active.approvals)
        assert view["saved_changes_inactive"] is (decision == "approve")
        if decision == "reject":
            assert (tmp_path / "tools.json").read_bytes() == original
        else:
            saved = runtime.approval_store.read()
            assert view["persisted_revision"] == saved.revision
            assert view["tools"]["update_tool_approvals"] is False
            updates = await outputs(runtime, "first", "update_tool_approvals")
            assert updates[0]["status"] == "updated"
            assert updates[0]["active_revision"] == active.revision
            assert updates[0]["saved_changes_inactive"] is True
            assert updates[1]["status"] == "conflict"
            model.responses = [
                call("get_tool_approvals"),
                call(
                    "update_tool_approvals",
                    updates={"extra": True},
                    expected_revision=saved.revision,
                ),
                AIMessage(content="next"),
            ]
            model.i = 0
            result = await runtime.invoke(
                AgentRequest(
                    "first",
                    "edit again",
                    metadata={"tool_approval_operator": True},
                    approval_handler=decide,
                )
            )
            assert result.text == "next"
            assert len(approvals) == 2
            next_view = (await outputs(runtime, "first", "get_tool_approvals"))[-1]
            assert next_view["active_revision"] == saved.revision
            assert next_view["active_tools"] == dict(saved.approvals)
            assert next_view["saved_changes_inactive"] is False
            assert runtime.approval_store.read().approvals["extra"] is True
        assert ACTIVE_APPROVALS.get() is None
        assert APPROVAL_OPERATOR.get() is False
    finally:
        await runtime.stop()


@pytest.mark.parametrize("reload_subagents", [False, True])
async def test_concurrent_invocation_keeps_graph_policy_and_read_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, reload_subagents: bool
) -> None:
    old_model = ToolModel(
        responses=[
            call("protected_effect"),
            call("get_tool_approvals"),
            call("protected_effect"),
            AIMessage(content="old done"),
        ]
    )
    runtime = make_runtime(tmp_path, monkeypatch, old_model)
    effects = []

    @tool
    def protected_effect() -> str:
        """Record the policy under which this effect executes."""
        effects.append(ACTIVE_APPROVALS.get())
        return "done"

    runtime.tools = (protected_effect,)
    initial = runtime.approval_store.ensure()
    runtime.approval_store.update({"protected_effect": True}, initial.revision)
    await runtime.start()
    active = runtime.approval_store.read()
    entered, release = asyncio.Event(), asyncio.Event()
    approvals = []

    async def reject(request: ToolApprovalRequest) -> str:
        approvals.extend(item["name"] for item in request.action_requests)
        assert ACTIVE_APPROVALS.get() == active
        entered.set()
        await asyncio.wait_for(release.wait(), 5)
        return "reject"

    pending = asyncio.create_task(
        runtime.invoke(AgentRequest("old", "work", approval_handler=reject))
    )
    try:
        await asyncio.wait_for(entered.wait(), 5)
        runtime.approval_store.update({"protected_effect": False}, active.revision)
        saved = runtime.approval_store.read()
        new_model = ToolModel(
            responses=[
                call("get_tool_approvals"),
                call("protected_effect"),
                AIMessage(content="new done"),
            ]
        )
        monkeypatch.setattr(
            "deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: new_model
        )
        assert (await runtime.invoke(AgentRequest("new", "work"))).text == "new done"
        assert effects == [saved]
        if reload_subagents:
            await runtime.reload_subagent_configuration()
            assert runtime._active_approvals == saved
        release.set()
        assert (await asyncio.wait_for(pending, 5)).text == "old done"
        assert approvals == ["protected_effect", "protected_effect"]
        assert effects == [saved]
        old_view = (await outputs(runtime, "old", "get_tool_approvals"))[0]
        new_view = (await outputs(runtime, "new", "get_tool_approvals"))[0]
        assert old_view["active_revision"] == active.revision
        assert old_view["active_tools"]["protected_effect"] is True
        assert old_view["persisted_revision"] == saved.revision
        assert old_view["tools"]["protected_effect"] is False
        assert old_view["saved_changes_inactive"] is True
        assert new_view["active_revision"] == saved.revision
        assert new_view["saved_changes_inactive"] is False
        assert ACTIVE_APPROVALS.get() is None
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await runtime.stop()


@pytest.mark.parametrize("failure", ["invalid", "construction"])
async def test_failed_reload_blocks_graph_and_preserves_previous_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    model = ToolModel(
        responses=[
            AIMessage(content="first"),
            AIMessage(content="recovered"),
            AIMessage(content="unexpected"),
        ]
    )
    runtime = make_runtime(tmp_path, monkeypatch, model)
    await runtime.start()
    try:
        assert (await runtime.invoke(AgentRequest("chat", "first"))).text == "first"
        graph, active = runtime._graph, runtime._active_approvals
        raw = (tmp_path / "tools.json").read_bytes()
        create_graph = runtime._create_graph
        if failure == "invalid":
            (tmp_path / "tools.json").write_text('{"update_tool_approvals": "false"}')
        else:
            runtime.approval_store.update({"update_tool_approvals": False}, active.revision)

            def fail(**_kwargs: object) -> object:
                msg = "graph construction failed"
                raise ValueError(msg)

            monkeypatch.setattr(runtime, "_create_graph", fail)
        for _ in range(2):
            with pytest.raises(
                ValueError, match=r"Invalid tool approval entry|graph construction failed"
            ):
                await runtime.invoke(AgentRequest("blocked", "must not run"))
            assert model.i == 1
            assert runtime._graph is graph
            assert runtime._active_approvals is active
            assert ACTIVE_APPROVALS.get() is None
            assert APPROVAL_OPERATOR.get() is False
            assert not (await graph.aget_state({"configurable": {"thread_id": "blocked"}})).values
        if failure == "invalid":
            (tmp_path / "tools.json").write_bytes(raw)
        else:
            monkeypatch.setattr(runtime, "_create_graph", create_graph)
        assert (await runtime.invoke(AgentRequest("recovery", "retry"))).text == "recovered"
        assert runtime._active_approvals == runtime.approval_store.read()
    finally:
        await runtime.stop()


@pytest.mark.parametrize("protected", [True, False])
async def test_local_subagent_inherits_invocation_approvals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, protected: bool
) -> None:
    parent = ToolModel(
        responses=[
            call("task", subagent_type="worker", description="work"),
            AIMessage(content="delegated"),
        ]
    )
    child = ToolModel(responses=[call("protected_effect"), AIMessage(content="child done")])
    runtime = make_runtime(tmp_path, monkeypatch, parent)
    effects = []

    @tool
    def protected_effect() -> str:
        """Record a delegated effect."""
        effects.append(ACTIVE_APPROVALS.get())
        return "done"

    runtime.subagents = (
        {
            "name": "worker",
            "description": "Worker",
            "system_prompt": "Work",
            "model": child,
            "tools": [protected_effect],
        },
    )
    initial = runtime.approval_store.ensure()
    runtime.approval_store.update({"protected_effect": protected}, initial.revision)
    await runtime.start()
    active = runtime.approval_store.read()
    try:
        assert (await runtime.invoke(AgentRequest("chat", "delegate"))).text == "delegated"
        await asyncio.wait_for(
            asyncio.gather(*(job.worker for job in runtime.background._jobs.values())), 5
        )
        results = runtime.background.results("chat")
        assert len(results) == 1
        if protected:
            assert effects == []
            assert "approval" in next(iter(results.values()))
        else:
            assert effects == [active]
            assert "<subagent_result>\nchild done\n</subagent_result>" in next(
                iter(results.values())
            )
    finally:
        await runtime.stop()


@pytest.mark.parametrize("protected", [True, False])
async def test_detached_operator_cannot_edit_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, protected: bool
) -> None:
    parent = ToolModel(
        responses=[
            call("task", subagent_type="worker", description="work"),
            AIMessage(content="delegated"),
        ]
    )
    runtime = make_runtime(tmp_path, monkeypatch, parent)
    initial = runtime.approval_store.ensure()
    runtime.approval_store.update(
        {"update_tool_approvals": False, "protected_effect": protected}, initial.revision
    )
    active = runtime.approval_store.read()
    raw = (tmp_path / "tools.json").read_bytes()
    entered, release = asyncio.Event(), asyncio.Event()
    observations, effects = [], []

    @tool
    async def wait_for_parent() -> str:
        """Wait until the originating invocation has returned."""
        entered.set()
        await asyncio.wait_for(release.wait(), 5)
        observations.append((ACTIVE_APPROVALS.get(), APPROVAL_OPERATOR.get()))
        return "ready"

    @tool
    def protected_effect() -> str:
        """Record a delegated protected action."""
        effects.append(ACTIVE_APPROVALS.get())
        return "done"

    child = ToolModel(
        responses=[
            call("wait_for_parent"),
            call(
                "update_tool_approvals", updates={"extra": False}, expected_revision=active.revision
            ),
            call("protected_effect"),
            AIMessage(content="child done"),
        ]
    )
    runtime.subagents = (
        {
            "name": "worker",
            "description": "Worker",
            "system_prompt": "Work",
            "model": child,
            "tools": [wait_for_parent, runtime.approval_store.tools(active)[1], protected_effect],
        },
    )
    await runtime.start()
    try:
        result = await runtime.invoke(
            AgentRequest("chat", "delegate", metadata={"tool_approval_operator": True})
        )
        assert result.text == "delegated"
        await asyncio.wait_for(entered.wait(), 5)
        assert observations == []
        release.set()
        await asyncio.wait_for(
            asyncio.gather(*(job.worker for job in runtime.background._jobs.values())), 5
        )
        assert observations == [(active, False)]
        assert (tmp_path / "tools.json").read_bytes() == raw
        assert effects == ([] if protected else [active])
        results = runtime.background.results("chat")
        assert len(results) == 1
        assert ("approval" if protected else "child done") in next(iter(results.values()))
    finally:
        release.set()
        await runtime.stop()


async def test_background_delivery_ignores_injected_approval_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = ToolModel(responses=[call("protected_effect"), AIMessage(content="done")])
    runtime = make_runtime(tmp_path, monkeypatch, model)
    effects, approvals = [], []

    @tool
    def protected_effect() -> str:
        """Record a protected effect."""
        effects.append("executed")
        return "done"

    async def approve(request: ToolApprovalRequest) -> str:
        approvals.append(request)
        return "approve"

    runtime.tools = (protected_effect,)
    initial = runtime.approval_store.ensure()
    runtime.approval_store.update({"protected_effect": True}, initial.revision)
    await runtime.start()
    try:
        await runtime.invoke(
            AgentRequest(
                "chat",
                "deliver result",
                metadata={"background_delivery": True, "tool_approval_operator": True},
                approval_handler=approve,
            )
        )
        assert approvals == []
        assert effects == []
        state = await runtime._graph.aget_state({"configurable": {"thread_id": "chat"}})
        denied = [
            message
            for message in state.values["messages"]
            if isinstance(message, ToolMessage) and message.name == "protected_effect"
        ]
        assert len(denied) == 1
        assert denied[0].status == "error"
    finally:
        await runtime.stop()


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {"tool_approval_operator": "true"},
        {"tool_approval_operator": True, "trigger": "cron"},
        {"tool_approval_operator": True, "background_delivery": True},
    ],
)
async def test_disabled_prompt_does_not_grant_operator_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, metadata: dict[str, object]
) -> None:
    model = ToolModel(responses=[AIMessage(content="placeholder")])
    runtime = make_runtime(tmp_path, monkeypatch, model)
    initial = runtime.approval_store.ensure()
    runtime.approval_store.update({"update_tool_approvals": False}, initial.revision)
    await runtime.start()
    active = runtime.approval_store.read()
    raw = (tmp_path / "tools.json").read_bytes()
    model.responses = [
        call("update_tool_approvals", updates={"extra": False}, expected_revision=active.revision),
        AIMessage(content="done"),
    ]
    try:
        assert (
            await runtime.invoke(AgentRequest("chat", "edit", metadata=metadata))
        ).text == "done"
        result = (await outputs(runtime, "chat", "update_tool_approvals"))[0]
        assert result == {"status": "error", "message": "Operator authorization is required."}
        assert (tmp_path / "tools.json").read_bytes() == raw
        assert ACTIVE_APPROVALS.get() is None
        assert APPROVAL_OPERATOR.get() is False
    finally:
        await runtime.stop()


@pytest.mark.parametrize(
    "policy", [None, {}, {"update_mcp_server": False}, {"update_mcp_server": True}]
)
def test_mcp_unsafe_update_uses_active_policy(
    tmp_path: Path, policy: dict[str, bool] | None
) -> None:
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {"mcpServers": {"server": {"command": "original", "env": {"TOKEN": "private-fixture"}}}}
        )
    )
    original = path.read_bytes()
    notifications = []
    read, update = MCPConfigStore(path, lambda: notifications.append("updated")).tools()
    approvals = ToolApprovalStore(tmp_path / "tools.json")
    persisted = approvals.ensure()
    approvals.update(
        {"update_mcp_server": policy != {"update_mcp_server": True}}, persisted.revision
    )
    snapshot = None if policy is None else ApprovalSnapshot("invocation", policy)
    token = ACTIVE_APPROVALS.set(snapshot)
    try:
        view = read.invoke({})
        server = view["mcpServers"]["server"] | {"command": "redirected"}
        result = update.invoke(
            {"server_name": "server", "server": server, "expected_revision": view["revision"]}
        )
    finally:
        ACTIVE_APPROVALS.reset(token)
    if policy == {"update_mcp_server": True}:
        assert result["status"] == "updated"
        assert json.loads(path.read_text())["mcpServers"]["server"]["command"] == "redirected"
        assert notifications == ["updated"]
    else:
        assert result["status"] == "error"
        assert path.read_bytes() == original
        assert notifications == []
