"""Native tool execution, safety, and provider-neutral integration tests."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from deepagents import create_deep_agent
from deepagents.backends.composite import CompositeBackend
from deepagents.backends.filesystem import FilesystemBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse
from deepagents.backends.state import StateBackend
from deepagents.middleware._native_patch import execute_patch
from deepagents.middleware.filesystem import FilesystemMiddleware, FilesystemPermission
from deepagents.middleware.native_tools import NativeToolsMiddleware, _responses_compatible, _shell_batch, native_tools_middleware
from deepagents.profiles.harness.harness_profiles import HarnessProfile, HarnessProfileConfig, _merge_profiles
from tests.unit_tests.chat_model import GenericFakeChatModel


@pytest.fixture
def backend(tmp_path: Path) -> FilesystemBackend:
    return FilesystemBackend(root_dir=tmp_path, virtual_mode=True)


def test_patch_create_update_delete(backend: FilesystemBackend, tmp_path: Path) -> None:
    execute_patch(backend, {"type": "create_file", "path": "file", "diff": "+one\n+two\n"}, [])
    assert (tmp_path / "file").read_bytes() == b"one\ntwo\n"
    execute_patch(backend, {"type": "update_file", "path": "file", "diff": "@@\n one\n-two\n+three\n"}, [])
    assert (tmp_path / "file").read_bytes() == b"one\nthree\n"
    execute_patch(backend, {"type": "delete_file", "path": "file"}, [])
    assert not (tmp_path / "file").exists()


@pytest.mark.parametrize(
    "operation",
    [
        {"type": "create_file", "path": "file", "diff": "+replace"},
        {"type": "update_file", "path": "file", "diff": "@@\n-missing\n+new"},
        {"type": "update_file", "path": "file", "diff": "@@\n-one\n+new\n@@\n-bad\n+bad"},
        {"type": "update_file", "path": "../escape", "diff": "@@\n-one\n+new"},
        {"type": "create_file", "path": "new", "diff": "not an added line"},
        {"type": "delete_file", "path": "file", "call_id": "wrong"},
    ],
)
def test_rejected_patch_does_not_write(backend: FilesystemBackend, tmp_path: Path, operation: dict) -> None:
    (tmp_path / "file").write_text("one\ntwo\n")
    with pytest.raises(ValueError, match=r"existing|context|traversal|added|Unexpected"):
        execute_patch(backend, operation, [])
    assert (tmp_path / "file").read_text() == "one\ntwo\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["file"]


@pytest.mark.parametrize("trailing", ["", "\n"])
def test_patch_preserves_final_newline(backend: FilesystemBackend, tmp_path: Path, trailing: str) -> None:
    (tmp_path / "file").write_text("old" + trailing)
    execute_patch(backend, {"type": "update_file", "path": "file", "diff": "@@\n-old\n+new"}, [])
    assert (tmp_path / "file").read_text() == "new" + trailing


def test_ambiguous_context_is_rejected(backend: FilesystemBackend, tmp_path: Path) -> None:
    (tmp_path / "file").write_text("same\nsame\n")
    with pytest.raises(ValueError, match="ambiguous"):
        execute_patch(backend, {"type": "update_file", "path": "file", "diff": "@@\n-same\n+new"}, [])
    execute_patch(backend, {"type": "update_file", "path": "file", "diff": "@@\n-same\n+new\n*** End of File"}, [])
    assert (tmp_path / "file").read_text() == "same\nnew\n"


def test_patch_honors_permissions_and_backend_write(backend: FilesystemBackend, tmp_path: Path) -> None:
    rules = [FilesystemPermission(paths=["/**"], operations=["write"], mode="deny")]
    with pytest.raises(ValueError, match="permission denied"):
        execute_patch(backend, {"type": "create_file", "path": "file", "diff": "+new"}, rules)
    assert not (tmp_path / "file").exists()
    with patch.object(backend, "write", return_value=Mock(error="read only")) as write:
        with pytest.raises(ValueError, match="read only"):
            execute_patch(backend, {"type": "create_file", "path": "file", "diff": "+new"}, [])
        write.assert_called_once()


def test_profile_roundtrip_and_explicit_disable() -> None:
    config = HarnessProfileConfig(enable_native_apply_patch=True, enable_native_shell=True)
    assert HarnessProfileConfig.from_dict(config.to_dict()) == config
    assert HarnessProfileConfig.from_harness_profile(config.to_harness_profile()) == config
    merged = _merge_profiles(config.to_harness_profile(), HarnessProfile(enable_native_shell=False))
    assert merged.enable_native_apply_patch is True
    assert merged.enable_native_shell is False
    with pytest.raises(TypeError):
        HarnessProfileConfig.from_dict({"enable_native_shell": "yes"})


def test_native_binding_preserves_custom_and_readonly_tools(tmp_path: Path) -> None:
    backend = LocalShellBackend(root_dir=tmp_path)
    filesystem = FilesystemMiddleware(backend=backend)
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    request = ModelRequest(model=model, messages=[], tools=list(filesystem.tools))
    native = NativeToolsMiddleware(backend=backend, enable_apply_patch=True, enable_shell=True)
    with patch("deepagents.middleware.native_tools._responses_compatible", return_value=True):
        tools = native._tools(request)
        assert {"type": "apply_patch"} in tools
        assert {"type": "shell", "environment": {"type": "local"}} in tools
        assert not any(getattr(tool, "name", None) == "execute" for tool in tools)
        custom = NativeToolsMiddleware(backend=backend, enable_shell=True, custom_tool_names=frozenset({"execute"}))
        assert custom._tools(request) == list(filesystem.tools)
        readonly = NativeToolsMiddleware(backend=backend, enable_apply_patch=True, excluded_tools=frozenset({"write_file"}))
        assert readonly._tools(request) == list(filesystem.tools)
    with pytest.raises(ValueError, match="Responses-compatible"):
        native._tools(request)
    composite = NativeToolsMiddleware(backend=CompositeBackend(default=backend, routes={"/memory/": StateBackend()}), enable_shell=True)
    assert composite.shell


def test_shell_validation_deadline_and_full_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    backend = LocalShellBackend(root_dir=tmp_path)
    execute = Mock(return_value=ExecuteResponse(output="display", exit_code=124, stdout="abcdef", stderr="error"))
    monkeypatch.setattr(backend, "execute", execute)
    visible, full = _shell_batch(backend, {"commands": ["one", "two"], "timeout_ms": 500, "max_output_length": 4})
    assert execute.call_count == 2
    assert 0 < execute.call_args.kwargs["timeout"] <= 0.5
    assert visible[0] == {"stdout": "abcd", "stderr": "", "outcome": {"type": "exit", "exit_code": 124}}
    assert full[1]["stdout"] == "abcdef"
    execute.reset_mock()
    with pytest.raises(ValueError, match="commands must"):
        _shell_batch(backend, {"commands": ["valid", 3]})
    execute.assert_not_called()
    assert _shell_batch(backend, {"commands": []}) == ([], [])
    execute.assert_not_called()
    execute.return_value = ExecuteResponse(output="timeout", timed_out=True, stdout="partial", stderr="")
    visible, _ = _shell_batch(backend, {"commands": ["one", "two"]})
    assert visible == [{"stdout": "partial", "stderr": "", "outcome": {"type": "timeout"}}]
    assert execute.call_count == 1


def _model() -> GenericFakeChatModel:
    return GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "apply_patch", "args": {"operation": {"type": "create_file", "path": "file", "diff": "+new"}}, "id": "call_native"}
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )


def test_graph_native_execution_and_human_edit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deepagents.middleware.native_tools._responses_compatible", lambda _: True)
    profile = HarnessProfile(enable_native_apply_patch=True)
    with patch("deepagents.graph._harness_profile_for_model", return_value=profile):
        graph = create_deep_agent(
            model=_model(),
            backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
            interrupt_on={"apply_patch": True},
            checkpointer=InMemorySaver(),
        )
    config = {"configurable": {"thread_id": "native"}}
    graph.invoke({"messages": [{"role": "user", "content": "patch"}]}, config)
    assert not (tmp_path / "file").exists()
    result = graph.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "apply_patch",
                            "args": {"operation": {"type": "create_file", "path": "edited", "diff": "+changed"}},
                        },
                    }
                ]
            }
        ),
        config,
    )
    assert not (tmp_path / "file").exists()
    assert (tmp_path / "edited").read_text() == "changed\n"
    output = next(message for message in result["messages"] if isinstance(message, ToolMessage))
    assert output.tool_call_id == "call_native"
    assert any(isinstance(block, dict) and block.get("type") == "apply_patch_call_output" for block in output.content), repr(output.content)


async def test_async_graph_native_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("deepagents.middleware.native_tools._responses_compatible", lambda _: True)
    with patch("deepagents.graph._harness_profile_for_model", return_value=HarnessProfile(enable_native_apply_patch=True)):
        graph = create_deep_agent(model=_model(), backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True))
    await graph.ainvoke({"messages": [{"role": "user", "content": "patch"}]})
    assert (tmp_path / "file").read_text() == "new\n"


def test_native_call_runs_downstream_wrapper(backend: FilesystemBackend) -> None:
    native = NativeToolsMiddleware(backend=backend, enable_apply_patch=True)
    request = ToolCallRequest(tool_call={"name": "apply_patch", "args": {"operation": {}}, "id": "call_native"}, tool=None, state={}, runtime=Mock())
    handler = Mock(return_value=ToolMessage(content="rejected", tool_call_id="call_native"))
    result = native.wrap_tool_call(request, handler)
    assert result.content == "rejected"
    assert handler.call_args.args[0].tool is not None


def test_relative_patch_uses_physical_cwd(tmp_path: Path) -> None:
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=False)
    execute_patch(backend, {"type": "create_file", "path": "native-relative", "diff": "+inside"}, [])
    assert (tmp_path / "native-relative").read_text() == "inside\n"


def test_readonly_factory_refuses_hallucinated_shell(tmp_path: Path) -> None:
    backend = LocalShellBackend(root_dir=tmp_path)
    filesystem = FilesystemMiddleware(backend=backend)
    filesystem.tools = [tool for tool in filesystem.tools if tool.name != "execute"]
    native = native_tools_middleware(HarnessProfile(enable_native_shell=True), backend, [], [filesystem])[0]
    request = ToolCallRequest(
        tool_call={"name": "shell", "args": {"action": {"commands": ["touch unsafe"]}}, "id": "call_shell"}, tool=None, state={}, runtime=Mock()
    )
    handler = Mock(return_value=ToolMessage(content="unknown tool", tool_call_id="call_shell"))
    with patch.object(backend, "execute") as execute:
        result = native.wrap_tool_call(request, handler)
    execute.assert_not_called()
    assert result.content == "unknown tool"
    assert handler.call_args.args[0].tool is None


def test_incompatible_backend_is_explicit_unless_custom_tool_owned(backend: FilesystemBackend) -> None:
    with pytest.raises(ValueError, match="stock LocalShellBackend"):
        NativeToolsMiddleware(backend=backend, enable_shell=True)
    native = NativeToolsMiddleware(backend=backend, enable_shell=True, custom_tool_names=frozenset({"execute"}))
    assert not native.shell


@pytest.mark.parametrize(("enabled", "missing"), [("enable_shell", {"execute"}), ("enable_apply_patch", {"delete"})])
def test_missing_stock_tools_fail_closed(tmp_path: Path, enabled: str, missing: set[str]) -> None:
    backend = LocalShellBackend(root_dir=tmp_path)
    filesystem = FilesystemMiddleware(backend=backend)
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    request = ModelRequest(model=model, messages=[], tools=[tool for tool in filesystem.tools if tool.name not in missing])
    native = NativeToolsMiddleware(backend=backend, **{enabled: True})
    with (
        patch("deepagents.middleware.native_tools._responses_compatible", return_value=True),
        pytest.raises(ValueError, match="requires the built-in"),
    ):
        native._tools(request)


def test_native_restore_keeps_exactly_one_provider_block() -> None:
    block = {"type": "apply_patch_call_output", "status": "completed", "output": "done"}
    result = NativeToolsMiddleware._restore_output(
        ToolMessage(content="custom diagnostic", tool_call_id="call", artifact={"native_tool_output": block})
    )
    assert result.content == [block]
    assert result.artifact["native_tool_diagnostic"] == "custom diagnostic"


async def test_async_incompatible_model_fails_before_handler(tmp_path: Path) -> None:
    backend = LocalShellBackend(root_dir=tmp_path)
    native = NativeToolsMiddleware(backend=backend, enable_shell=True)
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    request = ModelRequest(model=model, messages=[], tools=list(FilesystemMiddleware(backend=backend).tools))
    handler = Mock()
    with pytest.raises(ValueError, match="Responses-compatible"):
        await native.awrap_model_call(request, handler)
    handler.assert_not_called()


def test_model_request_restores_outer_approval_notice(tmp_path: Path) -> None:
    backend = LocalShellBackend(root_dir=tmp_path)
    native = NativeToolsMiddleware(backend=backend, enable_apply_patch=True)
    block = {"type": "apply_patch_call_output", "status": "completed", "output": "done"}
    message = ToolMessage(content=[{"type": "text", "text": "reviewer edited"}, block], tool_call_id="call", artifact={"native_tool_output": block})
    model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
    request = ModelRequest(model=model, messages=[message], tools=list(FilesystemMiddleware(backend=backend).tools))
    handler = Mock()
    with patch("deepagents.middleware.native_tools._responses_compatible", return_value=True):
        native.wrap_model_call(request, handler)
    restored = handler.call_args.args[0].messages[0]
    assert restored.content == [block]
    assert restored.artifact["native_tool_diagnostic"] == message.content
    assert len(message.content) == 2


def test_native_adapter_capability_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    model = ChatOpenAI(model="test-model", api_key="test-key", use_responses_api=True)
    monkeypatch.setattr(ChatOpenAI, "supports_native_agent_tools", False, raising=False)
    assert not _responses_compatible(model)
    monkeypatch.setattr(ChatOpenAI, "supports_native_agent_tools", True)
    assert _responses_compatible(model)
    assert not _responses_compatible(model.model_copy(update={"use_responses_api": False}))
