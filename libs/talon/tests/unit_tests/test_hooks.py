"""Exercise shared hooks through real Talon graphs without a model service."""

from __future__ import annotations

import asyncio
import json
import shlex
import sys
from typing import TYPE_CHECKING

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage

from deepagents_talon.hooks import TalonHooks
from deepagents_talon.interfaces import AgentRequest, ToolApprovalRequest
from deepagents_talon.runtime import DeepAgentRuntime

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_talon.interfaces import ToolApprovalDecision


class ToolModel(FakeMessagesListChatModel):
    def bind_tools(self, _tools: object, **_kwargs: object) -> ToolModel:
        return self


def configure_hooks(
    home: Path, events: list[str], response: dict[str, object] | None = None
) -> Path:
    home.mkdir(parents=True, exist_ok=True)
    script = home / "hook.py"
    script.write_text(
        "import json, sys\n"
        "with open(sys.argv[1], 'a') as log:\n"
        "    log.write(json.dumps(json.load(sys.stdin)) + '\\n')\n"
        "print(sys.argv[2])\n"
    )
    log = home / "events.jsonl"
    command = shlex.join([sys.executable, str(script), str(log), json.dumps(response or {})])
    (home / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    event: [
                        {
                            **({"matcher": "Bash"} if event != "Stop" else {}),
                            "hooks": [{"type": "command", "command": command}],
                        }
                    ]
                    for event in events
                }
            }
        )
    )
    return log


def make_runtime(
    home: Path, monkeypatch: pytest.MonkeyPatch, *, command: str = "printf original"
) -> DeepAgentRuntime:
    model = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[{"name": "execute", "args": {"command": command}, "id": "shell"}],
            ),
            AIMessage(content="done"),
        ]
    )
    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: model)
    return DeepAgentRuntime(
        model="test:hooks",
        assistant_dir=home,
        include_web_tools=False,
        skills=(),
        memory=(),
        max_retries=1,
        env={"DEEPAGENTS_TALON_WORKSPACE": str(home)},
    )


async def tool_results(runtime: DeepAgentRuntime, thread_id: str = "chat") -> list[ToolMessage]:
    state = await runtime._graph.aget_state({"configurable": {"thread_id": thread_id}})
    return [message for message in state.values["messages"] if isinstance(message, ToolMessage)]


async def test_tool_and_stop_hooks_receive_wire_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = configure_hooks(tmp_path, ["PreToolUse", "PostToolUse", "Stop"])
    target = tmp_path / "effect"
    command = "echo original | tee -a " + shlex.quote(str(target))
    runtime = make_runtime(tmp_path, monkeypatch, command=command)
    await runtime.start()
    result = await runtime.invoke(AgentRequest(conversation_id="chat", text="run"))

    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert result.text == "done"
    assert [event["hook_event_name"] for event in events] == ["PreToolUse", "PostToolUse", "Stop"]
    assert events[0]["tool_name"] == "Bash"
    assert events[0]["tool_input"]["command"] == command
    assert {event["session_id"] for event in events} == {"chat"}
    assert all(event["transcript_path"] for event in events)
    assert "original" in (await tool_results(runtime))[0].content
    assert target.read_text() == "original\n"


async def test_pre_tool_hook_can_block_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_hooks(
        tmp_path,
        ["PreToolUse"],
        {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny"}},
    )
    target = tmp_path / "effect"
    runtime = make_runtime(tmp_path, monkeypatch, command=shlex.join(["touch", str(target)]))
    snapshot = runtime.approval_store.ensure()
    runtime.approval_store.update({"execute": True}, snapshot.revision)
    approvals: list[ToolApprovalRequest] = []

    async def decide(request: ToolApprovalRequest) -> ToolApprovalDecision:
        approvals.append(request)
        return "approve"

    await runtime.start()
    await runtime.invoke(AgentRequest(conversation_id="chat", text="run", approval_handler=decide))

    assert approvals == []
    assert not target.exists()
    assert (await tool_results(runtime))[0].status == "error"


@pytest.mark.parametrize("decision", ["approve", "reject"])
@pytest.mark.parametrize("obsolete_value", ["", " , ", "bash, execute,custom/mcp"])
async def test_hook_allow_does_not_bypass_talon_approval(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: ToolApprovalDecision,
    obsolete_value: str,
) -> None:
    log = configure_hooks(
        tmp_path,
        ["PreToolUse"],
        {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow"}},
    )
    target = tmp_path / "effect"
    runtime = make_runtime(tmp_path, monkeypatch, command=shlex.join(["touch", str(target)]))
    runtime.env.update(
        {
            "DEEPAGENTS_TALON_INTERRUPT_ON_TOOLS": obsolete_value,
            "DEEPAGENTS_TALON_MCP_CONFIG_AUTO_APPROVE": "true",
        }
    )
    snapshot = runtime.approval_store.ensure()
    runtime.approval_store.update({"execute": True}, snapshot.revision)
    approvals: list[ToolApprovalRequest] = []

    async def decide(request: ToolApprovalRequest) -> ToolApprovalDecision:
        assert log.exists()
        assert len(log.read_text().splitlines()) == 1
        approvals.append(request)
        return decision

    await runtime.start()
    await runtime.invoke(AgentRequest(conversation_id="chat", text="run", approval_handler=decide))

    assert len(approvals) == 1
    assert approvals[0].action_requests[0]["name"] == "execute"
    assert target.exists() is (decision == "approve")
    assert len(log.read_text().splitlines()) == 1


async def test_hook_denial_does_not_skip_approval_for_other_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_hooks(
        tmp_path,
        ["PreToolUse"],
        {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny"}},
    )
    blocked = tmp_path / "blocked"
    allowed = tmp_path / "allowed"
    runtime = make_runtime(tmp_path, monkeypatch)
    model = ToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "execute",
                        "args": {"command": shlex.join(["touch", str(blocked)])},
                        "id": "blocked",
                    },
                    {
                        "name": "write_file",
                        "args": {"file_path": str(allowed), "content": "approved"},
                        "id": "allowed",
                    },
                ],
            ),
            AIMessage(content="done"),
        ]
    )
    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: model)
    snapshot = runtime.approval_store.ensure()
    runtime.approval_store.update({"execute": True, "write_file": True}, snapshot.revision)
    approvals: list[ToolApprovalRequest] = []

    async def decide(request: ToolApprovalRequest) -> ToolApprovalDecision:
        approvals.append(request)
        return "approve"

    await runtime.start()
    await runtime.invoke(AgentRequest(conversation_id="chat", text="run", approval_handler=decide))
    assert [action["name"] for request in approvals for action in request.action_requests] == [
        "write_file"
    ]
    assert not blocked.exists()
    assert allowed.read_text() == "approved"


async def test_post_failure_hook_receives_failed_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = configure_hooks(tmp_path, ["PostToolUseFailure"])
    runtime = make_runtime(tmp_path, monkeypatch, command="exit 1")
    await runtime.start()
    await runtime.invoke(AgentRequest(conversation_id="chat", text="run"))

    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert [event["hook_event_name"] for event in events] == ["PostToolUseFailure"]


async def test_workspace_hooks_are_not_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = configure_hooks(tmp_path / ".deepagents", ["PreToolUse"])
    configure_hooks(tmp_path, [])
    runtime = make_runtime(tmp_path, monkeypatch)
    await runtime.start()
    await runtime.invoke(AgentRequest(conversation_id="chat", text="run"))

    assert not log.exists()
    assert "original" in (await tool_results(runtime))[0].content


async def test_concurrent_turns_keep_hook_contexts_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = configure_hooks(tmp_path, ["Stop"])
    runtime = make_runtime(tmp_path, monkeypatch)
    await runtime.start()
    await asyncio.gather(
        runtime.invoke(AgentRequest(conversation_id="first", text="run")),
        runtime.invoke(AgentRequest(conversation_id="second", text="run")),
    )

    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert {event["session_id"] for event in events} == {"first", "second"}
    assert len({event["transcript_path"] for event in events}) == 2
    assert runtime._hooks.get() is None


def test_missing_hooks_configuration_is_disabled(tmp_path: Path) -> None:
    assert TalonHooks.load(tmp_path, tmp_path, "chat") is None


async def test_hook_configuration_reloads_between_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_hooks(tmp_path, ["PreToolUse"])
    target = tmp_path / "effect"
    runtime = make_runtime(tmp_path, monkeypatch, command=shlex.join(["touch", str(target)]))
    await runtime.start()
    await runtime.invoke(AgentRequest(conversation_id="chat", text="run"))
    assert target.exists()
    target.unlink()
    configure_hooks(
        tmp_path,
        ["PreToolUse"],
        {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny"}},
    )
    await runtime.invoke(AgentRequest(conversation_id="chat", text="run again"))

    assert not target.exists()
