"""Unit tests for the Hooks v2 execution engine."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks import dispatch_hook
from deepagents_code.hooks.engine import HookEngine
from deepagents_code.hooks.migration import migrate_legacy_hooks
from deepagents_code.hooks.models.adapters import HOOK_WIRE_INPUT_ADAPTER
from deepagents_code.hooks.models.config import HooksConfig
from deepagents_code.hooks.models.domain import (
    AgentIdentity,
    CompactTrigger,
    DcodeNotification,
    DcodeNotificationKind,
    HookContext,
    HookDiagnostic,
    HookEvent,
    HookInvocation,
    NotificationEvent,
    PermissionRequestDecision,
    PermissionRequestEvent,
    PostToolUseDecision,
    PostToolUseEvent,
    PreCompactDecision,
    PreCompactEvent,
    PreToolUseDecision,
    PreToolUseEvent,
    SessionEndCause,
    SessionEndEvent,
    SessionStartCause,
    SessionStartDecision,
    SessionStartEvent,
    StopDecision,
    StopEvent,
    SubagentStartDecision,
    SubagentStartEvent,
    SubagentStopDecision,
    SubagentStopEvent,
    ToolCallData,
    UserPromptSubmitDecision,
    UserPromptSubmitEvent,
)
from deepagents_code.hooks.models.wire import HookWireOutput
from deepagents_code.hooks.projection import project_hook_input, serialize_hook_input
from deepagents_code.hooks.reducer import reduce_hook_results
from deepagents_code.hooks.runner import HandlerResult, run_command_handler
from deepagents_code.hooks.snapshot import HookHandler, HooksSnapshot
from deepagents_code.hooks.tools import to_wire_call

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.hooks.models.domain import HookDomainEvent
    from deepagents_code.json_types import JsonObject


def _context(tmp_path: Path, *, agent: AgentIdentity | None = None) -> HookContext:
    return HookContext(
        thread_id="thread-1",
        cwd=tmp_path,
        prompt_id=uuid4(),
        approval_mode=ApprovalMode.MANUAL,
        effort="high",
        agent=agent,
    )


def _transcript_path(tmp_path: Path) -> Path:
    return tmp_path / "thread.jsonl"


def _agent_transcript_path(tmp_path: Path) -> Path:
    return tmp_path / "agent.jsonl"


def _invocation(tmp_path: Path, event: HookDomainEvent) -> HookInvocation:
    agent = getattr(event, "agent", None)
    if not isinstance(agent, AgentIdentity):
        agent = None
    return HookInvocation(context=_context(tmp_path, agent=agent), event=event)


def _config(hooks: dict[str, object]) -> HooksConfig:
    return HooksConfig.model_validate({"hooks": hooks})


def _handler(
    tmp_path: Path,
    command: str,
    *,
    timeout: float | None = None,
) -> HookHandler:
    snapshot = HooksSnapshot.from_config(
        _config(
            {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": command,
                                "timeout": timeout,
                            }
                        ]
                    }
                ]
            }
        )
    )
    invocation = _invocation(
        tmp_path,
        SessionStartEvent(
            event=HookEvent.SESSION_START, cause=SessionStartCause.STARTUP
        ),
    )
    return snapshot.match(invocation).handlers[0]


def test_snapshot_preserves_order_and_stable_ids(tmp_path: Path) -> None:
    snapshot = HooksSnapshot.from_config(
        _config(
            {
                "PreToolUse": [
                    {
                        "matcher": "^Bash$",
                        "hooks": [
                            {"type": "command", "command": "first"},
                            {"type": "command", "command": "second"},
                        ],
                    },
                    {"hooks": [{"type": "command", "command": "all"}]},
                ]
            }
        )
    )
    invocation = _invocation(
        tmp_path,
        PreToolUseEvent(
            event=HookEvent.PRE_TOOL_USE,
            call=ToolCallData(id="call-1", name="Bash", args={}),
        ),
    )

    match = snapshot.match(invocation)

    assert [handler.command for handler in match.handlers] == [
        "first",
        "second",
        "all",
    ]
    assert [handler.id for handler in match.handlers] == [
        "PreToolUse:0:0",
        "PreToolUse:0:1",
        "PreToolUse:1:0",
    ]


def test_snapshot_matches_notification_and_skips_tool_mismatch(tmp_path: Path) -> None:
    snapshot = HooksSnapshot.from_config(
        _config(
            {
                "Notification": [
                    {
                        "matcher": "permission_.*",
                        "hooks": [{"type": "command", "command": "notify"}],
                    }
                ],
                "PermissionRequest": [
                    {
                        "matcher": "Write",
                        "hooks": [{"type": "command", "command": "write"}],
                    }
                ],
            }
        )
    )
    notification = _invocation(
        tmp_path,
        NotificationEvent(
            event=HookEvent.NOTIFICATION,
            notification=DcodeNotification(
                type=DcodeNotificationKind.PERMISSION_REQUIRED,
                message="Approve",
            ),
        ),
    )
    permission = _invocation(
        tmp_path,
        PermissionRequestEvent(
            event=HookEvent.PERMISSION_REQUEST,
            call=ToolCallData(id="call-1", name="Bash", args={}),
        ),
    )

    assert [item.command for item in snapshot.match(notification).handlers] == [
        "notify"
    ]
    assert snapshot.match(permission).handlers == ()


def test_snapshot_matches_compaction_trigger(tmp_path: Path) -> None:
    snapshot = HooksSnapshot.from_config(
        _config(
            {
                "PreCompact": [
                    {
                        "matcher": "manual",
                        "hooks": [{"type": "command", "command": "manual"}],
                    },
                    {"hooks": [{"type": "command", "command": "all"}]},
                ]
            }
        )
    )
    invocation = _invocation(
        tmp_path,
        PreCompactEvent(
            event=HookEvent.PRE_COMPACT,
            trigger=CompactTrigger.MANUAL,
        ),
    )

    assert [item.command for item in snapshot.match(invocation).handlers] == [
        "manual",
        "all",
    ]


def test_snapshot_matches_native_tool_names_via_wire_adapter(tmp_path: Path) -> None:
    snapshot = HooksSnapshot.from_config(
        _config(
            {
                "PreToolUse": [
                    {
                        "matcher": "Bash|Write",
                        "hooks": [{"type": "command", "command": "policy"}],
                    }
                ]
            }
        )
    )
    execute = _invocation(
        tmp_path,
        PreToolUseEvent(
            event=HookEvent.PRE_TOOL_USE,
            call=ToolCallData(id="call-1", name="execute", args={"command": "pwd"}),
        ),
    )
    write = _invocation(
        tmp_path,
        PreToolUseEvent(
            event=HookEvent.PRE_TOOL_USE,
            call=ToolCallData(id="call-2", name="write_file", args={}),
        ),
    )
    read = _invocation(
        tmp_path,
        PreToolUseEvent(
            event=HookEvent.PRE_TOOL_USE,
            call=ToolCallData(id="call-3", name="read_file", args={}),
        ),
    )

    assert [item.command for item in snapshot.match(execute).handlers] == ["policy"]
    assert [item.command for item in snapshot.match(write).handlers] == ["policy"]
    assert snapshot.match(read).handlers == ()


def test_snapshot_exact_matcher_does_not_substring_match(tmp_path: Path) -> None:
    snapshot = HooksSnapshot.from_config(
        _config(
            {
                "PreToolUse": [
                    {
                        "matcher": "Write",
                        "hooks": [{"type": "command", "command": "write"}],
                    }
                ]
            }
        )
    )
    # Exact matchers must not treat "Write" as a regex substring of "WriteFile".
    invocation = _invocation(
        tmp_path,
        PreToolUseEvent(
            event=HookEvent.PRE_TOOL_USE,
            call=ToolCallData(id="call-1", name="WriteFile", args={}),
        ),
    )

    assert snapshot.match(invocation).handlers == ()


def test_snapshot_rejects_invalid_matcher_at_compile_time(tmp_path: Path) -> None:
    snapshot = HooksSnapshot.from_config(
        _config(
            {
                "SessionEnd": [
                    {
                        "matcher": "[",
                        "hooks": [{"type": "command", "command": "bad"}],
                    },
                    {
                        "matcher": "logout",
                        "hooks": [{"type": "command", "command": "good"}],
                    },
                    {
                        "hooks": [{"type": "command", "command": "all"}],
                    },
                ]
            }
        )
    )
    invocation = _invocation(
        tmp_path,
        SessionEndEvent(event=HookEvent.SESSION_END, cause=SessionEndCause.OTHER),
    )

    match = snapshot.match(invocation)

    assert [handler.command for handler in match.handlers] == ["all"]
    assert [diagnostic.code for diagnostic in snapshot.diagnostics] == [
        "invalid_matcher"
    ]
    assert match.diagnostics == ()


def test_snapshot_rejects_matcher_for_unmatchable_event() -> None:
    snapshot = HooksSnapshot.from_config(
        _config(
            {
                "Stop": [
                    {
                        "matcher": "Bash",
                        "hooks": [{"type": "command", "command": "invalid"}],
                    },
                    {"hooks": [{"type": "command", "command": "valid"}]},
                ]
            }
        )
    )

    assert [item.command for item in snapshot.handlers[HookEvent.STOP]] == ["valid"]
    assert [item.code for item in snapshot.diagnostics] == ["unsupported_matcher"]


@pytest.mark.parametrize(
    ("event", "expected"),
    [
        (
            SessionStartEvent(
                event=HookEvent.SESSION_START,
                cause=SessionStartCause.RESUME,
                model="provider:model",
            ),
            {"hook_event_name": "SessionStart", "source": "resume"},
        ),
        (
            UserPromptSubmitEvent(
                event=HookEvent.USER_PROMPT_SUBMIT,
                prompt="Review this change",
            ),
            {
                "hook_event_name": "UserPromptSubmit",
                "prompt": "Review this change",
            },
        ),
        (
            SessionEndEvent(event=HookEvent.SESSION_END, cause=SessionEndCause.LOGOUT),
            {"hook_event_name": "SessionEnd", "reason": "logout"},
        ),
        (
            PermissionRequestEvent(
                event=HookEvent.PERMISSION_REQUEST,
                call=ToolCallData(id="call-1", name="Bash", args={"command": "pwd"}),
            ),
            {"hook_event_name": "PermissionRequest", "tool_name": "Bash"},
        ),
        (
            NotificationEvent(
                event=HookEvent.NOTIFICATION,
                notification=DcodeNotification(
                    type=DcodeNotificationKind.PERMISSION_REQUIRED,
                    message="Approve",
                ),
            ),
            {
                "hook_event_name": "Notification",
                "notification_type": "permission_prompt",
            },
        ),
        (
            PreToolUseEvent(
                event=HookEvent.PRE_TOOL_USE,
                call=ToolCallData(id="call-1", name="Write", args={}),
            ),
            {"hook_event_name": "PreToolUse", "tool_use_id": "call-1"},
        ),
        (
            PostToolUseEvent(
                event=HookEvent.POST_TOOL_USE,
                call=ToolCallData(id="call-2", name="Bash", args={}),
                result=ToolMessage(content="done", tool_call_id="call-2"),
            ),
            {"hook_event_name": "PostToolUse", "tool_use_id": "call-2"},
        ),
        (
            PreCompactEvent(
                event=HookEvent.PRE_COMPACT,
                trigger=CompactTrigger.MANUAL,
                custom_instructions="Keep the plan",
            ),
            {
                "hook_event_name": "PreCompact",
                "trigger": "manual",
                "custom_instructions": "Keep the plan",
            },
        ),
        (
            StopEvent(
                event=HookEvent.STOP,
                continuation_count=1,
                last_assistant_message="Done",
            ),
            {"hook_event_name": "Stop", "stop_hook_active": True},
        ),
        (
            SubagentStartEvent(
                event=HookEvent.SUBAGENT_START,
                agent=AgentIdentity(id="agent-1", name="researcher"),
            ),
            {"hook_event_name": "SubagentStart", "agent_id": "agent-1"},
        ),
        (
            SubagentStopEvent(
                event=HookEvent.SUBAGENT_STOP,
                agent=AgentIdentity(id="agent-1", name="researcher"),
                continuation_count=0,
                last_assistant_message="Done",
            ),
            {"hook_event_name": "SubagentStop", "agent_id": "agent-1"},
        ),
    ],
)
def test_projects_all_wire_events(
    tmp_path: Path,
    event: HookDomainEvent,
    expected: dict[str, object],
) -> None:
    invocation = _invocation(tmp_path, event)

    payload = HOOK_WIRE_INPUT_ADAPTER.dump_python(
        project_hook_input(
            invocation,
            transcript_path=_transcript_path(tmp_path),
            agent_transcript_path=(
                _agent_transcript_path(tmp_path)
                if isinstance(event, SubagentStopEvent)
                else None
            ),
        ),
        mode="json",
        by_alias=True,
        exclude_none=True,
    )

    assert payload.items() >= expected.items()
    assert payload["session_id"] == "thread-1"
    assert payload["permission_mode"] == "default"
    assert payload["effort"] == {"level": "high"}
    assert payload["transcript_path"].endswith("thread.jsonl")


def test_projects_native_tool_names_to_wire(tmp_path: Path) -> None:
    invocation = _invocation(
        tmp_path,
        PreToolUseEvent(
            event=HookEvent.PRE_TOOL_USE,
            call=ToolCallData(
                id="call-1",
                name="execute",
                args={"command": "pwd"},
            ),
        ),
    )

    payload = HOOK_WIRE_INPUT_ADAPTER.dump_python(
        project_hook_input(
            invocation,
            transcript_path=_transcript_path(tmp_path),
        ),
        mode="json",
        by_alias=True,
        exclude_none=True,
    )

    assert payload["tool_name"] == "Bash"
    assert payload["tool_input"] == {"command": "pwd"}

    agent = AgentIdentity(id="agent-9", name="researcher")
    nested = HookInvocation(
        context=_context(tmp_path, agent=agent),
        event=PreToolUseEvent(
            event=HookEvent.PRE_TOOL_USE,
            call=ToolCallData(id="call-2", name="ls", args={"path": "."}),
        ),
    )
    nested_payload = HOOK_WIRE_INPUT_ADAPTER.dump_python(
        project_hook_input(
            nested,
            transcript_path=_transcript_path(tmp_path),
        ),
        mode="json",
        by_alias=True,
        exclude_none=True,
    )
    assert nested_payload["agent_id"] == "agent-9"
    assert nested_payload["agent_type"] == "researcher"

    invocation = _invocation(
        tmp_path,
        PostToolUseEvent.model_construct(
            event=HookEvent.POST_TOOL_USE,
            call=ToolCallData(id="call-1", name="Bash", args={}),
            result=Command(update={"result": "done"}),
        ),
    )

    payload = json.loads(
        serialize_hook_input(
            invocation,
            transcript_path=_transcript_path(tmp_path),
        )
    )

    assert payload["tool_response"]["update"] == {"result": "done"}
    assert payload["tool_response"]["goto"] == []


def test_projection_rejects_unknown_notification_and_projects_auto_mode(
    tmp_path: Path,
) -> None:
    unknown = HookInvocation(
        context=HookContext(
            thread_id="thread",
            cwd=tmp_path,
            approval_mode=ApprovalMode.MANUAL,
        ),
        event=NotificationEvent(
            event=HookEvent.NOTIFICATION,
            notification=DcodeNotification(type="invented", message="notice"),
        ),
    )
    with pytest.raises(ValueError, match="Unsupported notification type"):
        project_hook_input(
            unknown,
            transcript_path=_transcript_path(tmp_path),
        )

    automatic = HookInvocation(
        context=HookContext(
            thread_id="thread",
            cwd=tmp_path,
            approval_mode=ApprovalMode.AUTO,
        ),
        event=SessionStartEvent(
            event=HookEvent.SESSION_START,
            cause=SessionStartCause.STARTUP,
        ),
    )
    payload = HOOK_WIRE_INPUT_ADAPTER.dump_python(
        project_hook_input(
            automatic,
            transcript_path=_transcript_path(tmp_path),
        ),
        mode="json",
        by_alias=True,
        exclude_none=True,
    )
    assert payload["permission_mode"] == "auto"


async def test_engine_accepts_auto_permission_mode(
    tmp_path: Path,
) -> None:
    invocation = _invocation(
        tmp_path,
        SessionStartEvent(
            event=HookEvent.SESSION_START,
            cause=SessionStartCause.STARTUP,
        ),
    )
    snapshot = HooksSnapshot.from_config(HooksConfig(hooks={}))

    automatic = HookInvocation(
        context=invocation.context.model_copy(
            update={"approval_mode": ApprovalMode.AUTO}
        ),
        event=invocation.event,
    )
    auto = await HookEngine(snapshot).run(
        automatic,
        transcript_path=_transcript_path(tmp_path),
    )

    assert auto.diagnostics == []


@pytest.mark.parametrize(
    ("name", "args", "wire_name", "wire_input"),
    [
        (
            "execute",
            {"command": "pytest", "timeout": 30},
            "Bash",
            {"command": "pytest", "timeout": 30_000},
        ),
        (
            "write_file",
            {"file_path": "/tmp/result.txt", "content": "done"},
            "Write",
            {"file_path": "/tmp/result.txt", "content": "done"},
        ),
        (
            "edit_file",
            {
                "file_path": "/tmp/result.txt",
                "old_string": "before",
                "new_string": "after",
                "replace_all": True,
            },
            "Edit",
            {
                "file_path": "/tmp/result.txt",
                "old_string": "before",
                "new_string": "after",
                "replace_all": True,
            },
        ),
        (
            "read_file",
            {"file_path": "/tmp/result.txt", "offset": 0, "limit": 100},
            "Read",
            {"file_path": "/tmp/result.txt", "offset": 1, "limit": 100},
        ),
        (
            "glob",
            {"pattern": "**/*.py", "path": "/tmp"},
            "Glob",
            {"pattern": "**/*.py", "path": "/tmp"},
        ),
        (
            "grep",
            {
                "pattern": "result.*",
                "path": "/tmp",
                "glob": "*.txt",
                "output_mode": "content",
                "max_count": 20,
            },
            "Grep",
            {
                "pattern": "result\\.\\*",
                "path": "/tmp",
                "glob": "*.txt",
                "output_mode": "content",
                "head_limit": 20,
            },
        ),
        ("ls", {"path": "/tmp"}, "LS", {"path": "/tmp"}),
        ("custom", {"value": 1}, "custom", {"value": 1}),
    ],
)
def test_adapts_native_tool_calls_to_wire(
    name: str,
    args: JsonObject,
    wire_name: str,
    wire_input: JsonObject,
) -> None:
    call = ToolCallData(id="call-1", name=name, args=args)

    assert to_wire_call(call) == (wire_name, wire_input)


def test_serializes_native_tool_message_as_json(tmp_path: Path) -> None:
    invocation = _invocation(
        tmp_path,
        PostToolUseEvent(
            event=HookEvent.POST_TOOL_USE,
            call=ToolCallData(id="call-1", name="Bash", args={}),
            result=ToolMessage(
                content=[{"type": "text", "text": "done"}],
                tool_call_id="call-1",
            ),
        ),
    )

    payload = json.loads(
        serialize_hook_input(
            invocation,
            transcript_path=_transcript_path(tmp_path),
        )
    )

    assert payload["tool_response"]["content"] == [{"type": "text", "text": "done"}]


async def test_runner_accepts_json_and_uses_invocation_cwd(tmp_path: Path) -> None:
    code = "import json,os; print(json.dumps({'systemMessage': os.getcwd()}))"
    handler = _handler(tmp_path, f"{sys.executable} -c {json.dumps(code)}")

    result = await run_command_handler(handler, b"{}", cwd=tmp_path)

    assert result.output is not None
    assert result.output.system_message == str(tmp_path)
    assert result.diagnostics == ()


async def test_runner_executes_shell_syntax(tmp_path: Path) -> None:
    out = tmp_path / "shell.txt"
    command = f"printf '%s' '{{\"systemMessage\":\"ok\"}}' > {out} && cat {out}"
    handler = _handler(tmp_path, command)

    result = await run_command_handler(handler, b"{}", cwd=tmp_path)

    assert result.output is not None
    assert result.output.system_message == "ok"
    assert out.read_text() == '{"systemMessage":"ok"}'


@pytest.mark.parametrize(
    ("code", "expected_code"),
    [
        ("pass", None),
        ("print('[]')", "invalid_output"),
        ("raise SystemExit(3)", "nonzero_exit"),
    ],
)
async def test_runner_protocol_failures_are_structured(
    tmp_path: Path,
    code: str,
    expected_code: str | None,
) -> None:
    handler = _handler(tmp_path, f"{sys.executable} -c {json.dumps(code)}")

    result = await run_command_handler(handler, b"{}", cwd=tmp_path)

    assert result.output is None
    assert [item.code for item in result.diagnostics] == (
        [] if expected_code is None else [expected_code]
    )


async def test_runner_session_start_plain_stdout_is_context(tmp_path: Path) -> None:
    code = "print('plain')"
    handler = _handler(tmp_path, f"{sys.executable} -c {json.dumps(code)}")

    result = await run_command_handler(
        handler,
        b"{}",
        cwd=tmp_path,
    )

    assert result.output is None
    assert result.plain_output == "plain"
    assert result.diagnostics == ()


async def test_reducer_applies_pretool_plain_stdout_policy(tmp_path: Path) -> None:
    code = "print('not json')"
    snapshot = HooksSnapshot.from_config(
        _config(
            {
                "PreToolUse": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": f"{sys.executable} -c {json.dumps(code)}",
                            }
                        ]
                    }
                ]
            }
        )
    )
    handler = snapshot.handlers[HookEvent.PRE_TOOL_USE][0]

    result = await run_command_handler(
        handler,
        b"{}",
        cwd=tmp_path,
    )
    invocation = _invocation(
        tmp_path,
        PreToolUseEvent(
            event=HookEvent.PRE_TOOL_USE,
            call=ToolCallData(id="call", name="execute", args={}),
        ),
    )
    decision = reduce_hook_results(invocation, [result])

    assert result.output is None
    assert result.plain_output == "not json"
    assert result.diagnostics == ()
    assert [item.code for item in decision.diagnostics] == ["malformed_json"]


async def test_runner_turns_exit_two_stderr_into_block(tmp_path: Path) -> None:
    code = "import sys; print('protected', file=sys.stderr); raise SystemExit(2)"
    handler = _handler(tmp_path, f"{sys.executable} -c {json.dumps(code)}")

    result = await run_command_handler(handler, b"{}", cwd=tmp_path)

    assert result.output == HookWireOutput(decision="block", reason="protected")


async def test_runner_times_out_and_reaps_process(tmp_path: Path) -> None:
    code = "import time; time.sleep(10)"
    handler = _handler(
        tmp_path,
        f"{sys.executable} -c {json.dumps(code)}",
        timeout=0.01,
    )

    result = await run_command_handler(handler, b"{}", cwd=tmp_path)

    assert [item.code for item in result.diagnostics] == ["timeout"]


async def test_runner_reports_launch_failure_and_bounded_streams(
    tmp_path: Path,
) -> None:
    # Shell form: a missing binary is started by the shell and exits non-zero
    # rather than failing at process spawn.
    missing = _handler(tmp_path, "definitely-not-a-real-hook-command")
    failed = await run_command_handler(missing, b"{}", cwd=tmp_path)
    empty = _handler(tmp_path, "   ")
    empty_result = await run_command_handler(empty, b"{}", cwd=tmp_path)
    code = "import sys; print('x'*100); print('y'*100, file=sys.stderr)"
    noisy = _handler(tmp_path, f"{sys.executable} -c {json.dumps(code)}")
    bounded = await run_command_handler(noisy, b"{}", cwd=tmp_path, max_output_bytes=10)

    assert [item.code for item in failed.diagnostics] == ["nonzero_exit"]
    assert [item.code for item in empty_result.diagnostics] == ["invalid_command"]
    assert {item.code for item in bounded.diagnostics} == {
        "stdout_truncated",
        "stderr_truncated",
    }
    assert bounded.plain_output == "x" * 10


def test_reducer_merges_session_context_and_common_fields(tmp_path: Path) -> None:
    invocation = _invocation(
        tmp_path,
        SessionStartEvent(
            event=HookEvent.SESSION_START, cause=SessionStartCause.STARTUP
        ),
    )
    results = [
        HandlerResult(
            handler_id="one",
            output=HookWireOutput.model_validate(
                {
                    "systemMessage": "notice",
                    "terminalSequence": "\x1b]9;done\x07",
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": "context one",
                    },
                }
            ),
        ),
        HandlerResult(
            handler_id="two",
            output=HookWireOutput.model_validate(
                {
                    "continue": False,
                    "stopReason": "stop",
                    "hookSpecificOutput": {
                        "hookEventName": "SessionStart",
                        "additionalContext": "context two",
                    },
                }
            ),
        ),
    ]

    decision = reduce_hook_results(invocation, results)
    assert isinstance(decision, SessionStartDecision)

    assert decision.context == ["context one", "context two"]
    assert decision.user_notices == ["notice"]
    assert decision.terminal_sequences == ["\x1b]9;done\x07"]
    assert decision.continue_processing is False
    assert decision.stop_reason == "stop"


def test_reducer_blocks_prompt_and_compaction(tmp_path: Path) -> None:
    prompt = reduce_hook_results(
        _invocation(
            tmp_path,
            UserPromptSubmitEvent(
                event=HookEvent.USER_PROMPT_SUBMIT,
                prompt="Deploy",
            ),
        ),
        [
            HandlerResult(
                handler_id="prompt-policy",
                plain_output="Use staging",
                output=HookWireOutput.model_validate(
                    {
                        "decision": "block",
                        "reason": "Production deploys require approval",
                        "hookSpecificOutput": {
                            "hookEventName": "UserPromptSubmit",
                            "additionalContext": "Check the release checklist",
                            "suppressOriginalPrompt": True,
                        },
                    }
                ),
            )
        ],
    )
    compact = reduce_hook_results(
        _invocation(
            tmp_path,
            PreCompactEvent(
                event=HookEvent.PRE_COMPACT,
                trigger=CompactTrigger.MANUAL,
            ),
        ),
        [
            HandlerResult(
                handler_id="compact-policy",
                output=HookWireOutput(
                    decision="block",
                    reason="Preserve the current context",
                ),
            )
        ],
    )

    assert isinstance(prompt, UserPromptSubmitDecision)
    assert prompt.continue_processing is False
    assert prompt.stop_reason == "Production deploys require approval"
    assert prompt.context == ["Use staging", "Check the release checklist"]
    assert prompt.suppress_original_prompt is True
    assert isinstance(compact, PreCompactDecision)
    assert compact.continue_processing is False
    assert compact.stop_reason == "Preserve the current context"


def test_reducer_keeps_prompt_suppression_across_handlers(tmp_path: Path) -> None:
    invocation = _invocation(
        tmp_path,
        UserPromptSubmitEvent(
            event=HookEvent.USER_PROMPT_SUBMIT,
            prompt="Deploy",
        ),
    )
    outputs = [
        {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "suppressOriginalPrompt": True,
            }
        },
        {
            "hookSpecificOutput": {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": "Use staging",
            }
        },
    ]

    decision = reduce_hook_results(
        invocation,
        [
            HandlerResult(
                handler_id=str(index),
                output=HookWireOutput.model_validate(output),
            )
            for index, output in enumerate(outputs)
        ],
    )

    assert isinstance(decision, UserPromptSubmitDecision)
    assert decision.suppress_original_prompt is True
    assert decision.context == ["Use staging"]


async def test_migrated_legacy_handler_remains_side_effect_only(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "legacy payload.json"
    script = (
        "import json,pathlib,sys;"
        "pathlib.Path(sys.argv[1]).write_text(json.dumps(json.load(sys.stdin)));"
        "print(json.dumps({'decision':'block','reason':'legacy'}));"
        "sys.exit(2)"
    )
    config = migrate_legacy_hooks(
        [
            {
                "command": [sys.executable, "-c", script, str(payload_path)],
                "events": ["session.start"],
            }
        ]
    )
    invocation = _invocation(
        tmp_path,
        UserPromptSubmitEvent(
            event=HookEvent.USER_PROMPT_SUBMIT,
            prompt="Continue",
        ),
    )

    decision = await HookEngine(HooksSnapshot.from_config(config)).run(
        invocation,
        transcript_path=_transcript_path(tmp_path),
    )

    assert isinstance(decision, UserPromptSubmitDecision)
    assert decision.continue_processing is True
    assert decision.context == []
    assert json.loads(payload_path.read_text()) == {
        "event": "session.start",
        "thread_id": "thread-1",
    }


async def test_migrated_dual_legacy_events_reconstruct_each_payload(
    tmp_path: Path,
) -> None:
    payload_dir = tmp_path / "payloads"
    payload_dir.mkdir()
    script = (
        "import json,pathlib,sys;"
        "data=json.load(sys.stdin);"
        "pathlib.Path(sys.argv[1], data['event']+'.json')"
        ".write_text(json.dumps(data))"
    )
    config = migrate_legacy_hooks(
        [
            {
                "command": [sys.executable, "-c", script, str(payload_dir)],
                "events": ["session.start", "user.prompt"],
            }
        ]
    )
    invocation = _invocation(
        tmp_path,
        UserPromptSubmitEvent(
            event=HookEvent.USER_PROMPT_SUBMIT,
            prompt="Continue",
        ),
    )

    decision = await HookEngine(HooksSnapshot.from_config(config)).run(
        invocation,
        transcript_path=_transcript_path(tmp_path),
    )

    assert isinstance(decision, UserPromptSubmitDecision)
    assert decision.continue_processing is True
    assert json.loads((payload_dir / "session.start.json").read_text()) == {
        "event": "session.start",
        "thread_id": "thread-1",
    }
    assert json.loads((payload_dir / "user.prompt.json").read_text()) == {
        "event": "user.prompt",
    }


def test_reducer_permission_precedence_is_deny_ask_allow(tmp_path: Path) -> None:
    invocation = _invocation(
        tmp_path,
        PreToolUseEvent(
            event=HookEvent.PRE_TOOL_USE,
            call=ToolCallData(id="call", name="Bash", args={}),
        ),
    )
    results = [
        HandlerResult(
            handler_id="allow",
            output=HookWireOutput.model_validate(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                    }
                }
            ),
        ),
        HandlerResult(
            handler_id="ask",
            output=HookWireOutput.model_validate(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "ask",
                    }
                }
            ),
        ),
        HandlerResult(
            handler_id="deny",
            output=HookWireOutput(decision="block", reason="no"),
        ),
    ]

    decision = reduce_hook_results(invocation, results)
    assert isinstance(decision, PermissionRequestDecision | PreToolUseDecision)

    assert decision.permission.behavior == "deny"
    assert decision.permission.reason == "no"


def test_reducer_covers_event_decision_shapes_and_loop_guards(tmp_path: Path) -> None:
    agent = AgentIdentity(id="agent-1", name="researcher")
    events_and_outputs = [
        (
            SessionEndEvent(event=HookEvent.SESSION_END, cause=SessionEndCause.OTHER),
            {},
        ),
        (
            NotificationEvent(
                event=HookEvent.NOTIFICATION,
                notification=DcodeNotification(type="agent_completed", message="Done"),
            ),
            {},
        ),
        (
            PermissionRequestEvent(
                event=HookEvent.PERMISSION_REQUEST,
                call=ToolCallData(id="call", name="Bash", args={}),
            ),
            {
                "hookSpecificOutput": {
                    "hookEventName": "PermissionRequest",
                    "decision": {"behavior": "deny", "message": "denied"},
                }
            },
        ),
        (
            PostToolUseEvent(
                event=HookEvent.POST_TOOL_USE,
                call=ToolCallData(id="call", name="Bash", args={}),
                result=ToolMessage(content="done", tool_call_id="call"),
            ),
            {
                "decision": "block",
                "reason": "feedback",
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": "context",
                },
            },
        ),
        (
            StopEvent(
                event=HookEvent.STOP,
                continuation_count=1,
                last_assistant_message="Done",
            ),
            {
                "hookSpecificOutput": {
                    "hookEventName": "Stop",
                    "additionalContext": "continue",
                }
            },
        ),
        (
            SubagentStartEvent(event=HookEvent.SUBAGENT_START, agent=agent),
            {
                "hookSpecificOutput": {
                    "hookEventName": "SubagentStart",
                    "additionalContext": "focus",
                }
            },
        ),
        (
            SubagentStopEvent(
                event=HookEvent.SUBAGENT_STOP,
                agent=agent,
                continuation_count=1,
                last_assistant_message="Done",
            ),
            {
                "hookSpecificOutput": {
                    "hookEventName": "SubagentStop",
                    "additionalContext": "continue",
                }
            },
        ),
    ]

    decisions = [
        reduce_hook_results(
            _invocation(tmp_path, event),
            [
                HandlerResult(
                    handler_id="handler",
                    output=HookWireOutput.model_validate(output),
                )
            ],
        )
        for event, output in events_and_outputs
    ]

    assert [decision.event for decision in decisions] == [
        HookEvent.SESSION_END,
        HookEvent.NOTIFICATION,
        HookEvent.PERMISSION_REQUEST,
        HookEvent.POST_TOOL_USE,
        HookEvent.STOP,
        HookEvent.SUBAGENT_START,
        HookEvent.SUBAGENT_STOP,
    ]
    permission = decisions[2]
    post_tool = decisions[3]
    stop = decisions[4]
    subagent_start = decisions[5]
    subagent_stop = decisions[6]
    assert isinstance(permission, PermissionRequestDecision)
    assert isinstance(post_tool, PostToolUseDecision)
    assert isinstance(stop, StopDecision)
    assert isinstance(subagent_start, SubagentStartDecision)
    assert isinstance(subagent_stop, SubagentStopDecision)
    assert permission.permission.behavior == "deny"
    assert post_tool.feedback == ["feedback"]
    assert post_tool.context == ["context"]
    assert stop.continue_loop is True
    assert stop.feedback == ["continue"]
    assert subagent_start.context == ["focus"]
    assert subagent_stop.context == ["continue"]
    assert subagent_stop.diagnostics == []


def test_reducer_guards_top_level_stop_blocks(tmp_path: Path) -> None:
    stop_invocation = _invocation(
        tmp_path,
        StopEvent(
            event=HookEvent.STOP,
            continuation_count=8,
            last_assistant_message="Done",
        ),
    )
    subagent_invocation = _invocation(
        tmp_path,
        SubagentStopEvent(
            event=HookEvent.SUBAGENT_STOP,
            agent=AgentIdentity(id="agent-1", name="researcher"),
            continuation_count=1,
            last_assistant_message="Done",
        ),
    )
    result = HandlerResult(
        handler_id="block",
        output=HookWireOutput(decision="block", reason="continue"),
    )

    stop = reduce_hook_results(stop_invocation, [result])
    subagent = reduce_hook_results(subagent_invocation, [result])

    assert isinstance(stop, StopDecision)
    assert isinstance(subagent, SubagentStopDecision)
    assert stop.continue_loop is False
    assert stop.diagnostics[0].code == "continuation_cap"
    assert subagent.context == []
    assert subagent.diagnostics[0].code == "continuation_guard"


def test_reducer_retains_fail_open_diagnostics(tmp_path: Path) -> None:
    invocation = _invocation(
        tmp_path,
        NotificationEvent(
            event=HookEvent.NOTIFICATION,
            notification=DcodeNotification(type="agent_completed", message="Done"),
        ),
    )

    decision = reduce_hook_results(
        invocation,
        [
            HandlerResult(
                handler_id="broken",
                diagnostics=(
                    HookDiagnostic(
                        code="timeout",
                        severity="warning",
                        message="timed out",
                    ),
                ),
            )
        ],
    )

    assert decision.continue_processing is True
    assert decision.diagnostics[0].code == "timeout"


def test_reducer_ignores_session_start_block(tmp_path: Path) -> None:
    invocation = _invocation(
        tmp_path,
        SessionStartEvent(
            event=HookEvent.SESSION_START, cause=SessionStartCause.STARTUP
        ),
    )

    decision = reduce_hook_results(
        invocation,
        [
            HandlerResult(
                handler_id="block",
                output=HookWireOutput(decision="block", reason="nope"),
            )
        ],
    )
    assert isinstance(decision, SessionStartDecision)

    assert decision.continue_processing is True
    assert decision.diagnostics[0].code == "unsupported_block"


def test_reducer_warns_on_unsupported_updated_input(tmp_path: Path) -> None:
    invocation = _invocation(
        tmp_path,
        PreToolUseEvent(
            event=HookEvent.PRE_TOOL_USE,
            call=ToolCallData(id="call", name="execute", args={}),
        ),
    )

    decision = reduce_hook_results(
        invocation,
        [
            HandlerResult(
                handler_id="mutate",
                output=HookWireOutput.model_validate(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "allow",
                            "updatedInput": {"command": "echo mutated"},
                        }
                    }
                ),
            )
        ],
    )
    assert isinstance(decision, PreToolUseDecision)

    assert decision.permission.behavior == "none"
    assert decision.diagnostics[0].code == "unsupported_field"
    assert decision.diagnostics[0].field == "updatedInput"


def test_reducer_honors_deny_even_with_updated_input(tmp_path: Path) -> None:
    invocation = _invocation(
        tmp_path,
        PreToolUseEvent(
            event=HookEvent.PRE_TOOL_USE,
            call=ToolCallData(id="call", name="execute", args={}),
        ),
    )

    decision = reduce_hook_results(
        invocation,
        [
            HandlerResult(
                handler_id="deny",
                output=HookWireOutput.model_validate(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": "blocked",
                            "updatedInput": {"command": "echo mutated"},
                        }
                    }
                ),
            )
        ],
    )
    assert isinstance(decision, PreToolUseDecision)

    assert decision.permission.behavior == "deny"
    assert decision.permission.reason == "blocked"
    assert decision.diagnostics[0].code == "unsupported_field"


def test_reducer_keeps_stop_sticky_and_retains_siblings(tmp_path: Path) -> None:
    invocation = _invocation(
        tmp_path,
        SessionStartEvent(
            event=HookEvent.SESSION_START,
            cause=SessionStartCause.STARTUP,
        ),
    )
    decision = reduce_hook_results(
        invocation,
        [
            HandlerResult(
                handler_id="first",
                output=HookWireOutput.model_validate(
                    {"continue": False, "stopReason": "first"}
                ),
            ),
            HandlerResult(
                handler_id="second",
                output=HookWireOutput.model_validate(
                    {
                        "continue": False,
                        "stopReason": "second",
                        "hookSpecificOutput": {
                            "hookEventName": "SessionStart",
                            "additionalContext": "later context",
                        },
                    }
                ),
            ),
            HandlerResult(
                handler_id="third",
                diagnostics=(
                    HookDiagnostic(
                        code="sibling_failed",
                        severity="warning",
                        message="sibling diagnostic",
                    ),
                ),
                plain_output="plain sibling",
            ),
        ],
    )

    assert isinstance(decision, SessionStartDecision)
    assert decision.continue_processing is False
    assert decision.stop_reason == "first"
    assert decision.context == ["later context", "plain sibling"]
    assert {item.code for item in decision.diagnostics} == {
        "additional_stop_reason",
        "sibling_failed",
    }


def test_reducer_same_rank_permission_is_first_wins(tmp_path: Path) -> None:
    invocation = _invocation(
        tmp_path,
        PreToolUseEvent(
            event=HookEvent.PRE_TOOL_USE,
            call=ToolCallData(id="call", name="execute", args={}),
        ),
    )
    results = [
        HandlerResult(
            handler_id=reason,
            output=HookWireOutput.model_validate(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "ask",
                        "permissionDecisionReason": reason,
                    }
                }
            ),
        )
        for reason in ("first", "second")
    ]

    decision = reduce_hook_results(invocation, results)

    assert isinstance(decision, PreToolUseDecision)
    assert decision.permission.behavior == "ask"
    assert decision.permission.reason == "first"


def test_permission_request_diagnoses_all_deferred_fields(tmp_path: Path) -> None:
    invocation = _invocation(
        tmp_path,
        PermissionRequestEvent(
            event=HookEvent.PERMISSION_REQUEST,
            call=ToolCallData(id="call", name="execute", args={}),
        ),
    )
    output = HookWireOutput.model_validate(
        {
            "hookSpecificOutput": {
                "hookEventName": "PermissionRequest",
                "decision": {
                    "behavior": "allow",
                    "updatedInput": {"command": "changed"},
                    "updatedPermissions": [
                        {
                            "type": "setMode",
                            "mode": "default",
                            "destination": "session",
                        }
                    ],
                },
            }
        }
    )

    decision = reduce_hook_results(
        invocation,
        [HandlerResult(handler_id="deferred", output=output)],
    )

    assert isinstance(decision, PermissionRequestDecision)
    assert decision.permission.behavior == "none"
    assert {item.field for item in decision.diagnostics} == {
        "updatedInput",
        "updatedPermissions",
    }


async def test_engine_runs_handlers_concurrently(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first_cmd = (
        "import json,pathlib,time; "
        "time.sleep(0.05); "
        f"pathlib.Path({str(first)!r}).write_text('first'); "
        "print(json.dumps({'continue': False, 'stopReason': 'stop'}))"
    )
    second_cmd = (
        "import pathlib,time; "
        "time.sleep(0.05); "
        f"pathlib.Path({str(second)!r}).write_text('second')"
    )
    snapshot = HooksSnapshot.from_config(
        _config(
            {
                "SessionStart": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": (
                                    f"{sys.executable} -c {json.dumps(first_cmd)}"
                                ),
                            },
                            {
                                "type": "command",
                                "command": (
                                    f"{sys.executable} -c {json.dumps(second_cmd)}"
                                ),
                            },
                        ]
                    }
                ]
            }
        )
    )
    invocation = _invocation(
        tmp_path,
        SessionStartEvent(
            event=HookEvent.SESSION_START, cause=SessionStartCause.STARTUP
        ),
    )

    decision = await HookEngine(snapshot).run(
        invocation,
        transcript_path=_transcript_path(tmp_path),
    )

    assert decision.continue_processing is False
    assert decision.stop_reason == "stop"
    assert first.read_text() == "first"
    assert second.read_text() == "second"


async def test_engine_uses_captured_snapshot(tmp_path: Path) -> None:
    original = _config(
        {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": f"{sys.executable} -c pass"}]}
            ]
        }
    )
    snapshot = HooksSnapshot.from_config(original)
    original.hooks[HookEvent.SESSION_START][0].hooks[0].command = "missing"
    invocation = _invocation(
        tmp_path,
        SessionStartEvent(
            event=HookEvent.SESSION_START, cause=SessionStartCause.STARTUP
        ),
    )

    decision = await HookEngine(snapshot).run(
        invocation,
        transcript_path=_transcript_path(tmp_path),
    )

    assert decision.diagnostics == []


def test_legacy_dispatcher_remains_public() -> None:
    assert callable(dispatch_hook)


def test_legacy_package_import_does_not_load_engine() -> None:
    code = (
        "import sys; import deepagents_code.hooks; "
        "raise SystemExit('deepagents_code.hooks.engine' in sys.modules)"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
    )

    assert result.returncode == 0
