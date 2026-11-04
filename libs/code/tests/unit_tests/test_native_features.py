"""Safety and opt-in coverage for native provider tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock

import pytest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage

from deepagents_code import native_features
from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.config_manifest import get_option
from deepagents_code.memory_guard import ManagedMemoryGuardMiddleware
from unit_tests.conftest import resolve_option_for_test

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents.middleware.subagents import CompiledSubAgent, SubAgent
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.runnables import RunnableConfig


def _request(name: str, args: dict | None = None, **context: object) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": args or {}, "id": "call", "type": "tool_call"},
        tool=None,
        state={},
        runtime=Mock(context=context, store=None),
    )


@pytest.mark.parametrize("name", ["apply_patch", "native_shell"])
def test_native_options_default_off_and_resolve_toml(name: str) -> None:
    option = get_option(f"features.{name}")
    assert option is not None
    assert resolve_option_for_test(option, toml_data={})[0] is False
    assert (
        resolve_option_for_test(option, toml_data={"features": {name: True}})[0] is True
    )
    assert (
        resolve_option_for_test(
            option,
            toml_data={"features": {name: True}},
            managed_toml_data={"features": {name: False}},
        )[0]
        is False
    )


def test_disabled_features_do_not_load_sdk_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native_features, "_enabled", lambda _: False)
    factory = Mock()
    monkeypatch.setattr(native_features, "_extend_native_stack", factory)
    native_features.configure_native_features(
        backend=Mock(),
        model="test",
        tools=[],
        middleware=[],
        subagents=[],
        fs_tools=None,
        shell_allow_list=None,
    )
    factory.assert_not_called()


@pytest.mark.parametrize(
    ("feature", "fs_tools", "allow_list"),
    [
        ("native_shell", None, ["ls"]),
        ("native_shell", ["read_file"], None),
        ("apply_patch", ["read_file"], None),
    ],
)
def test_native_features_reject_restricted_access(
    monkeypatch: pytest.MonkeyPatch,
    feature: str,
    fs_tools: list[str] | None,
    allow_list: list[str] | None,
) -> None:
    monkeypatch.setattr(
        native_features, "_enabled", lambda key: key == f"features.{feature}"
    )
    with pytest.raises(ValueError, match=r"shell\.allow_list|allow-fs-tools"):
        native_features.configure_native_features(
            backend=Mock(),
            model="test",
            tools=[],
            middleware=[],
            subagents=[],
            fs_tools=fs_tools,
            shell_allow_list=allow_list,
        )


def test_session_stack_isolated_and_compiled_subagents_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native_features, "_enabled", lambda _: True)
    added = object()
    factory = Mock(side_effect=lambda **kwargs: kwargs["middleware"].append(added))
    monkeypatch.setattr(native_features, "_extend_native_stack", factory)
    stack = []
    child_stack = []
    custom_tool = Mock()
    child: SubAgent = {
        "name": "reader",
        "description": "read",
        "middleware": child_stack,
    }
    subagents: list[SubAgent | CompiledSubAgent] = [
        child,
        {"name": "compiled", "description": "compiled", "runnable": Mock()},
    ]
    native_features.configure_native_features(
        backend=Mock(),
        model="test",
        tools=[custom_tool],
        middleware=stack,
        subagents=subagents,
        fs_tools=None,
        shell_allow_list=None,
    )
    assert stack == [added]
    assert child_stack == []
    assert child["middleware"] == [added]
    assert "middleware" not in subagents[1]
    assert factory.call_args_list[0].kwargs["tools"] == [custom_tool]
    assert factory.call_args_list[1].kwargs["tools"] == [custom_tool]


@pytest.mark.parametrize("name", ["shell", "apply_patch"])
@pytest.mark.parametrize(
    "mode", [ApprovalMode.MANUAL, ApprovalMode.YOLO, ApprovalMode.AUTO]
)
async def test_native_guard_respects_live_approval_mode(
    monkeypatch: pytest.MonkeyPatch, name: str, mode: ApprovalMode
) -> None:
    from deepagents_code import agent

    monkeypatch.setattr(agent, "_aresolve_approval_mode", AsyncMock(return_value=mode))
    guard = native_features._native_safety_middleware([], frozenset({name}))
    handler = AsyncMock(return_value=ToolMessage(content="ran", tool_call_id="call"))
    result = await guard.awrap_tool_call(_request(name), handler)
    if mode is ApprovalMode.AUTO:
        assert result.status == "error"
        assert "Auto approval" in result.content
        handler.assert_not_called()
    else:
        handler.assert_awaited_once()


def test_native_guard_preserves_tool_use_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    from deepagents_code import agent

    monkeypatch.setattr(agent, "_resolve_approval_mode", lambda *_: ApprovalMode.YOLO)
    guard = native_features._native_safety_middleware([], frozenset({"shell"}))
    handler = Mock()
    result = guard.wrap_tool_call(
        _request("shell", hooks_server_events=["PreToolUse"]), handler
    )
    assert result.status == "error"
    assert "tool-use hooks" in result.content
    handler.assert_not_called()


def test_native_patch_preserves_managed_memory_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code import agent

    monkeypatch.setattr(agent, "_resolve_approval_mode", lambda *_: ApprovalMode.MANUAL)
    memory = tmp_path / "AGENTS.md"
    guard = native_features._native_safety_middleware(
        [ManagedMemoryGuardMiddleware([memory])], frozenset({"apply_patch"})
    )
    handler = Mock()
    result = guard.wrap_tool_call(
        _request(
            "apply_patch", {"operation": {"type": "update_file", "path": str(memory)}}
        ),
        handler,
    )
    assert result.status == "error"
    assert "memory-block guard" in result.content
    handler.assert_not_called()


def test_native_guard_does_not_change_custom_tool() -> None:
    guard = native_features._native_safety_middleware([], frozenset({"shell"}))
    handler = Mock(return_value=ToolMessage(content="custom", tool_call_id="call"))
    assert (
        guard.wrap_tool_call(
            _request("apply_patch", approval_mode="auto"), handler
        ).content
        == "custom"
    )
    handler.assert_called_once()


def test_native_tools_share_existing_approval_predicate() -> None:
    from deepagents_code.agent import _add_interrupt_on

    interrupts = _add_interrupt_on()
    for name in ("apply_patch", "shell"):
        assert interrupts[name]["when"] is interrupts["execute"]["when"]
        assert interrupts[name]["allowed_decisions"] == ["approve", "reject"]


@pytest.mark.parametrize("auto_approve", [False, True])
async def test_cli_native_graph_executes_patch_and_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, auto_approve: bool
) -> None:
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command

    from deepagents_code.agent import create_cli_agent

    if not getattr(ChatOpenAI, "supports_native_agent_tools", False):
        pytest.skip("Requires the native-tools OpenAI adapter companion change")
    monkeypatch.setattr(native_features, "_enabled", lambda _: True)
    target = tmp_path / "native.txt"
    replies = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "apply_patch",
                        "id": "patch-call",
                        "args": {
                            "operation": {
                                "type": "create_file",
                                "path": str(target),
                                "diff": "+native",
                            }
                        },
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "shell",
                        "id": "shell-call",
                        "args": {
                            "action": {
                                "commands": ["printf shell-ok"],
                                "timeout_ms": 1000,
                            }
                        },
                    }
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    bound_tools = []

    def generate(*_args: object, **kwargs: object) -> ChatResult:
        bound_tools.append(kwargs.get("tools"))
        return ChatResult(generations=[ChatGeneration(message=next(replies))])

    monkeypatch.setattr(ChatOpenAI, "_agenerate", AsyncMock(side_effect=generate))
    monkeypatch.setattr(
        ChatOpenAI, "get_num_tokens_from_messages", lambda *_args, **_kwargs: 50
    )
    model = ChatOpenAI(model="gpt-4o", api_key="test", use_responses_api=True)
    graph, _ = create_cli_agent(
        model=model,
        assistant_id="native-test",
        cwd=tmp_path,
        system_prompt="test",
        enable_memory=False,
        enable_skills=False,
        enable_ask_user=False,
        auto_approve=auto_approve,
        checkpointer=InMemorySaver(),
    )
    context = {
        "approval_mode": "yolo" if auto_approve else "manual",
        "auto_approve": auto_approve,
    }
    config: RunnableConfig = {"configurable": {"thread_id": "native-test"}}
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "do it"}]},
        config=config,
        context=context,
    )
    if not auto_approve:
        assert not target.exists()
        for _ in range(2):
            assert result.get("__interrupt__")
            result = await graph.ainvoke(
                Command(resume={"decisions": [{"type": "approve"}]}),
                config=config,
                context=context,
            )
        assert not result.get("__interrupt__")
    assert target.read_text() == "native\n"
    outputs = {
        message.tool_call_id: message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    }
    assert outputs["patch-call"].status == "success"
    assert outputs["shell-call"].status == "success"
    assert "shell-ok" in str(outputs["shell-call"].content)
    assert bound_tools
    assert any(tool.get("type") == "apply_patch" for tool in bound_tools[0])
    assert any(tool.get("type") == "shell" for tool in bound_tools[0])


def test_native_factory_keeps_readonly_subagent_permissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents.backends.local_shell import LocalShellBackend
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.middleware.native_tools import NativeToolsMiddleware

    monkeypatch.setattr(native_features, "_enabled", lambda _: True)
    backend = LocalShellBackend(root_dir=tmp_path)
    reader_stack: list[AgentMiddleware[Any, Any]] = [
        FilesystemMiddleware(backend=backend, tools=["read_file"])
    ]
    reader: SubAgent = {
        "name": "reader",
        "description": "read only",
        "middleware": reader_stack,
    }
    native_features.configure_native_features(
        backend=backend,
        model="openai:gpt-4o",
        tools=[],
        middleware=[],
        subagents=[reader],
        fs_tools=None,
        shell_allow_list=None,
    )
    native = next(
        item for item in reader["middleware"] if isinstance(item, NativeToolsMiddleware)
    )
    for name in ("apply_patch", "shell"):
        assert not native._handles(_request(name))
    assert (
        len(
            [
                item
                for item in reader["middleware"]
                if isinstance(item, FilesystemMiddleware)
            ]
        )
        == 1
    )


def test_native_factory_preserves_harness_exclusions_and_custom_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents import HarnessProfile
    from deepagents.backends.local_shell import LocalShellBackend
    from deepagents.middleware.native_tools import NativeToolsMiddleware
    from langchain_core.tools import StructuredTool

    from deepagents_code import agent

    profile = HarnessProfile(excluded_tools=frozenset({"execute"}))
    monkeypatch.setattr(agent, "_get_cli_harness_profile", lambda _: profile)
    monkeypatch.setattr(native_features, "_enabled", lambda _: True)
    custom = StructuredTool.from_function(
        lambda: "custom", name="apply_patch", description="Custom patch"
    )
    stack = []
    native_features.configure_native_features(
        backend=LocalShellBackend(root_dir=tmp_path),
        model="openai:gpt-4o",
        tools=[custom],
        middleware=stack,
        subagents=[],
        fs_tools=None,
        shell_allow_list=None,
    )
    native = next(item for item in stack if isinstance(item, NativeToolsMiddleware))
    assert not native.patch
    assert not native.shell
    assert profile.enable_native_apply_patch is None
    assert profile.enable_native_shell is None
    assert profile.excluded_tools == frozenset({"execute"})


def test_released_adapter_without_native_capability_fails_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents.backends.local_shell import LocalShellBackend
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from deepagents.middleware.native_tools import NativeToolsMiddleware
    from langchain.agents.middleware.types import ModelRequest
    from langchain_openai import ChatOpenAI

    monkeypatch.setattr(ChatOpenAI, "supports_native_agent_tools", False, raising=False)
    backend = LocalShellBackend(root_dir=tmp_path)
    middleware = NativeToolsMiddleware(backend=backend, enable_shell=True)
    model = ChatOpenAI(model="gpt-4o", api_key="test", use_responses_api=True)
    request = ModelRequest(
        model=model,
        messages=[],
        tools=list(FilesystemMiddleware(backend=backend).tools),
    )
    handler = Mock()
    with pytest.raises(ValueError, match="Responses-compatible"):
        middleware.wrap_model_call(request, handler)
    handler.assert_not_called()
