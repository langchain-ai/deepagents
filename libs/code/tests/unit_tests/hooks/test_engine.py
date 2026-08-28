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
from deepagents_code.hooks.envelope import HookEnvelopeAdapter
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
    PostToolUseFailureDecision,
    PostToolUseFailureEvent,
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
    from deepagents_code.hooks.presenter import HookProgress
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


def test_snapshot_matches_notification_and_skips_tool_mismatch(tmp_path: Path) -> None:
    snapshot = HooksSnapshot.from_config(
        _config(
            {
                "Notification": [
                    {
                        "matcher": "permission_prompt",
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
            SessionEndEvent(event=HookEvent.SESSION_END, cause=SessionEndCause.CLEAR),
            {"hook_event_name": "SessionEnd", "reason": "clear"},
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
            NotificationEvent(
                event=HookEvent.NOTIFICATION,
                notification=DcodeNotification(
                    type=DcodeNotificationKind.COLD_CACHE_WARNING,
                    message="Prompt cache may be cold",
                ),
            ),
            {
                "hook_event_name": "Notification",
                "notification_type": "cold_cache_warning",
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
            PostToolUseEvent.from_tool_result(
                ToolMessage(content="done", tool_call_id="call-2"),
                call=ToolCallData(id="call-2", name="Bash", args={}),
            ),
            {"hook_event_name": "PostToolUse", "tool_use_id": "call-2"},
        ),
        (
            PostToolUseFailureEvent(
                event=HookEvent.POST_TOOL_USE_FAILURE,
                call=ToolCallData(id="call-3", name="Bash", args={}),
                error="Command exited with non-zero status code 42",
                duration_ms=15,
            ),
            {
                "hook_event_name": "PostToolUseFailure",
                "tool_use_id": "call-3",
                "error": "Command exited with non-zero status code 42",
                "is_interrupt": False,
                "duration_ms": 15,
            },
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
            PostToolUseEvent.from_tool_result(
                ToolMessage(content="done", tool_call_id="call"),
                call=ToolCallData(id="call", name="Bash", args={}),
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
            PostToolUseFailureEvent(
                event=HookEvent.POST_TOOL_USE_FAILURE,
                call=ToolCallData(id="call", name="Bash", args={}),
                error="failed",
            ),
            {
                "decision": "block",
                "reason": "failure feedback",
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUseFailure",
                    "additionalContext": "failure context",
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
        HookEvent.POST_TOOL_USE_FAILURE,
        HookEvent.STOP,
        HookEvent.SUBAGENT_START,
        HookEvent.SUBAGENT_STOP,
    ]
    permission = decisions[2]
    post_tool = decisions[3]
    post_tool_failure = decisions[4]
    stop = decisions[5]
    subagent_start = decisions[6]
    subagent_stop = decisions[7]
    assert isinstance(permission, PermissionRequestDecision)
    assert isinstance(post_tool, PostToolUseDecision)
    assert isinstance(post_tool_failure, PostToolUseFailureDecision)
    assert isinstance(stop, StopDecision)
    assert isinstance(subagent_start, SubagentStartDecision)
    assert isinstance(subagent_stop, SubagentStopDecision)
    assert permission.permission.behavior == "deny"
    assert post_tool.feedback == ["feedback"]
    assert post_tool.context == ["context"]
    assert post_tool_failure.feedback == ["failure feedback"]
    assert post_tool_failure.context == ["failure context"]
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


async def test_engine_reduces_in_config_order_when_completion_is_reversed(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first_cmd = (
        "import json,pathlib,time; "
        "time.sleep(0.08); "
        f"pathlib.Path({str(first)!r}).write_text('first'); "
        "print(json.dumps({'continue': False, 'stopReason': 'stop'}))"
    )
    second_cmd = (
        "import json,pathlib,time; "
        "time.sleep(0.01); "
        f"pathlib.Path({str(second)!r}).write_text('second'); "
        "print(json.dumps({'continue': False, 'stopReason': 'later'}))"
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
