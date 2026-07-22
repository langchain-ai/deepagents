"""Contract tests for hooks data models."""

from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from langchain_core.messages import ToolMessage
from pydantic import ValidationError

from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.models.adapters import (
    HOOK_DECISION_ADAPTER,
    HOOK_DOMAIN_EVENT_ADAPTER,
    HOOK_INVOCATION_REQUEST_ADAPTER,
    HOOK_INVOCATION_RESPONSE_ADAPTER,
    HOOK_WIRE_INPUT_ADAPTER,
    HOOK_WIRE_OUTPUT_ADAPTER,
    HOOKS_CONFIG_ADAPTER,
)
from deepagents_code.hooks.models.domain import (
    HookContext,
    HookEvent,
    HookInvocation,
    PermissionEffect,
    PostToolUseEvent,
    SessionStartCause,
    SessionStartDecision,
    SessionStartEvent,
    ToolCallData,
)
from deepagents_code.hooks.models.transport import (
    HookInvocationRequest,
    HookInvocationResponse,
)

_COMMON_WIRE_INPUT = {
    "session_id": "thread-1",
    "transcript_path": "/tmp/transcript.jsonl",
    "cwd": "/workspace",
}

_WIRE_INPUTS = [
    {
        **_COMMON_WIRE_INPUT,
        "hook_event_name": "SessionStart",
        "source": "startup",
        "model": "provider:model",
    },
    {
        **_COMMON_WIRE_INPUT,
        "hook_event_name": "UserPromptSubmit",
        "prompt": "Review this change",
    },
    {
        **_COMMON_WIRE_INPUT,
        "hook_event_name": "SessionEnd",
        "reason": "other",
    },
    {
        **_COMMON_WIRE_INPUT,
        "hook_event_name": "PermissionRequest",
        "tool_name": "Bash",
        "tool_input": {"command": "pwd"},
    },
    {
        **_COMMON_WIRE_INPUT,
        "hook_event_name": "Notification",
        "message": "Approval required",
        "notification_type": "permission_prompt",
    },
    {
        **_COMMON_WIRE_INPUT,
        "hook_event_name": "PreToolUse",
        "tool_name": "Write",
        "tool_input": {"file_path": "notes.txt", "content": "hello"},
        "tool_use_id": "call-1",
    },
    {
        **_COMMON_WIRE_INPUT,
        "hook_event_name": "PostToolUse",
        "tool_name": "Bash",
        "tool_input": {"command": "pwd"},
        "tool_response": {"stdout": "/workspace"},
        "tool_use_id": "call-2",
        "duration_ms": 12,
    },
    {
        **_COMMON_WIRE_INPUT,
        "hook_event_name": "PreCompact",
        "trigger": "manual",
        "custom_instructions": "Keep the implementation plan",
    },
    {
        **_COMMON_WIRE_INPUT,
        "hook_event_name": "Stop",
        "stop_hook_active": False,
        "last_assistant_message": "Done",
    },
    {
        **_COMMON_WIRE_INPUT,
        "hook_event_name": "SubagentStart",
        "agent_id": "agent-1",
        "agent_type": "researcher",
    },
    {
        **_COMMON_WIRE_INPUT,
        "hook_event_name": "SubagentStop",
        "stop_hook_active": False,
        "agent_id": "agent-1",
        "agent_type": "researcher",
        "agent_transcript_path": "/tmp/agent.jsonl",
        "last_assistant_message": "Found it",
    },
]


@pytest.mark.parametrize("payload", _WIRE_INPUTS)
def test_wire_inputs_round_trip_with_exact_keys(payload: dict[str, object]) -> None:
    parsed = HOOK_WIRE_INPUT_ADAPTER.validate_python(payload)

    assert (
        HOOK_WIRE_INPUT_ADAPTER.dump_python(
            parsed,
            mode="json",
            by_alias=True,
            exclude_none=True,
            exclude_defaults=True,
        )
        == payload
    )


def test_wire_input_ignores_unknown_fields() -> None:
    payload = {**_WIRE_INPUTS[0], "unknown": True}

    parsed = HOOK_WIRE_INPUT_ADAPTER.validate_python(payload)

    assert parsed.hook_event_name is HookEvent.SESSION_START
    assert (
        HOOK_WIRE_INPUT_ADAPTER.dump_python(
            parsed,
            mode="json",
            by_alias=True,
            exclude_none=True,
            exclude_defaults=True,
        )
        == _WIRE_INPUTS[0]
    )


_SPECIFIC_OUTPUTS = [
    {
        "hookEventName": "SessionStart",
        "additionalContext": "Use the project environment",
        "watchPaths": ["/workspace/src"],
    },
    {
        "hookEventName": "UserPromptSubmit",
        "additionalContext": "Apply the repository conventions",
        "suppressOriginalPrompt": True,
    },
    {
        "hookEventName": "PreToolUse",
        "permissionDecision": "deny",
        "permissionDecisionReason": "Protected path",
    },
    {
        "hookEventName": "PermissionRequest",
        "decision": {
            "behavior": "allow",
            "updatedPermissions": [
                {
                    "type": "addRules",
                    "rules": [{"toolName": "Bash", "ruleContent": "git status"}],
                    "behavior": "allow",
                    "destination": "session",
                }
            ],
        },
    },
    {
        "hookEventName": "PostToolUse",
        "additionalContext": "Check the formatter output",
        "updatedMCPToolOutput": {"content": "deferred"},
    },
    {
        "hookEventName": "Stop",
        "additionalContext": "Finish the remaining task",
    },
    {
        "hookEventName": "SubagentStart",
        "additionalContext": "Focus on tests",
    },
    {
        "hookEventName": "SubagentStop",
        "additionalContext": "Verify the subagent result",
    },
]


@pytest.mark.parametrize("specific_output", _SPECIFIC_OUTPUTS)
def test_wire_outputs_round_trip_with_aliases(
    specific_output: dict[str, object],
) -> None:
    payload = {
        "continue": False,
        "stopReason": "Stopped by hook",
        "hookSpecificOutput": specific_output,
    }

    parsed = HOOK_WIRE_OUTPUT_ADAPTER.validate_python(payload)

    assert (
        HOOK_WIRE_OUTPUT_ADAPTER.dump_python(
            parsed,
            mode="json",
            by_alias=True,
            exclude_none=True,
            exclude_defaults=True,
        )
        == payload
    )


def test_wire_output_retains_unknown_fields() -> None:
    payload = {"continue": True, "futureField": {"enabled": True}}

    parsed = HOOK_WIRE_OUTPUT_ADAPTER.validate_python(payload)

    assert parsed.model_extra == {"futureField": {"enabled": True}}
    assert HOOK_WIRE_OUTPUT_ADAPTER.dump_python(
        parsed,
        mode="json",
        by_alias=True,
        exclude_defaults=True,
    ) == {"futureField": {"enabled": True}}


def test_wire_output_does_not_apply_internal_field_names() -> None:
    parsed = HOOK_WIRE_OUTPUT_ADAPTER.validate_python({"continue_": False})

    assert parsed.continue_ is True
    assert parsed.model_extra == {"continue_": False}


def test_wire_specific_output_ignores_unknown_fields() -> None:
    parsed = HOOK_WIRE_OUTPUT_ADAPTER.validate_python(
        {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "decision": {"behavior": "deny"},
            }
        }
    )

    specific = parsed.hook_specific_output
    assert specific is not None
    assert specific.hook_event_name == "PreToolUse"
    assert specific.permission_decision == "deny"


def test_wire_specific_output_rejects_invalid_permission_decision() -> None:
    with pytest.raises(ValidationError):
        HOOK_WIRE_OUTPUT_ADAPTER.validate_python(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "not-a-decision",
                }
            }
        )


def test_domain_event_union_selects_event_model() -> None:
    event = HOOK_DOMAIN_EVENT_ADAPTER.validate_python(
        {
            "event": "Notification",
            "notification": {
                "type": "approval_required",
                "message": "Approval required",
            },
        }
    )

    assert event.event is HookEvent.NOTIFICATION


def test_post_tool_use_accepts_native_tool_message() -> None:
    result = ToolMessage(content="done", tool_call_id="call-1")

    event = PostToolUseEvent(
        event=HookEvent.POST_TOOL_USE,
        call=ToolCallData(id="call-1", name="write_file", args={}),
        result=result,
    )

    assert event.result is result


def test_domain_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        HookContext.model_validate(
            {
                "thread_id": "thread-1",
                "cwd": Path("/workspace"),
                "approval_mode": ApprovalMode.MANUAL,
                "unsupported": True,
            }
        )
    with pytest.raises(ValidationError):
        HookContext.model_validate(
            {
                "thread_id": "thread-1",
                "cwd": Path("/workspace"),
                "approval_mode": ApprovalMode.MANUAL,
                "transcript_path": "/tmp/transcript.jsonl",
            }
        )


def test_decision_union_selects_event_model() -> None:
    decision = HOOK_DECISION_ADAPTER.validate_python(
        {
            "event": "PreToolUse",
            "permission": {"behavior": "ask"},
            "context": ["Explain the operation"],
        }
    )

    assert decision.event is HookEvent.PRE_TOOL_USE
    assert decision.permission == PermissionEffect(behavior="ask")


def test_hooks_config_validates_event_keys_and_aliases() -> None:
    payload = {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "./check.sh",
                            "timeout": 30,
                            "statusMessage": "Checking command",
                        }
                    ],
                }
            ]
        }
    }

    config = HOOKS_CONFIG_ADAPTER.validate_python(payload)

    assert config.hooks[HookEvent.PRE_TOOL_USE][0].hooks[0].status_message == (
        "Checking command"
    )
    assert (
        HOOKS_CONFIG_ADAPTER.dump_python(
            config,
            mode="json",
            by_alias=True,
            exclude_none=True,
        )
        == payload
    )


def test_hooks_config_rejects_async_and_ignores_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="async"):
        HOOKS_CONFIG_ADAPTER.validate_python(
            {
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "Bash",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "./check.sh",
                                    "async": True,
                                }
                            ],
                        }
                    ]
                }
            }
        )

    config = HOOKS_CONFIG_ADAPTER.validate_python(
        {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Bash",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "./check.sh",
                                "futureHandlerField": "keep-parsing",
                            }
                        ],
                    }
                ]
            }
        }
    )
    handler = config.hooks[HookEvent.PRE_TOOL_USE][0].hooks[0]
    assert handler.command == "./check.sh"
    assert handler.timeout is None
    assert handler.async_ is None

    normalized = HOOKS_CONFIG_ADAPTER.validate_python(
        {
            "hooks": {
                "PreToolUse": [
                    {
                        "matcher": "Bash",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "./check.sh",
                                "async": False,
                            }
                        ],
                    }
                ]
            }
        }
    )
    assert normalized.hooks[HookEvent.PRE_TOOL_USE][0].hooks[0].async_ is None


def test_hooks_config_rejects_unsupported_handler_type() -> None:
    with pytest.raises(ValidationError):
        HOOKS_CONFIG_ADAPTER.validate_python(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "http",
                                    "command": "https://example.com",
                                }
                            ]
                        }
                    ]
                }
            }
        )


def test_transport_models_round_trip_typed_domain_payloads() -> None:
    invocation_id = uuid4()
    invocation = HookInvocation(
        context=HookContext(
            thread_id="thread-1",
            cwd=Path("/workspace"),
            approval_mode=ApprovalMode.MANUAL,
        ),
        event=SessionStartEvent(
            event=HookEvent.SESSION_START,
            cause=SessionStartCause.STARTUP,
        ),
    )
    request = HookInvocationRequest(
        protocol_version=1,
        invocation_id=invocation_id,
        snapshot_id="snapshot-1",
        run_id="run-1",
        invocation=invocation,
        deadline=datetime(2026, 7, 20, tzinfo=UTC),
    )
    response = HookInvocationResponse(
        protocol_version=1,
        invocation_id=invocation_id,
        snapshot_id="snapshot-1",
        decision=SessionStartDecision(
            event=HookEvent.SESSION_START,
            context=["Loaded project context"],
        ),
    )

    request_json = HOOK_INVOCATION_REQUEST_ADAPTER.dump_json(request)
    response_json = HOOK_INVOCATION_RESPONSE_ADAPTER.dump_json(response)

    assert HOOK_INVOCATION_REQUEST_ADAPTER.validate_json(request_json) == request
    assert HOOK_INVOCATION_RESPONSE_ADAPTER.validate_json(response_json) == response
    assert b"transcript_path" not in request_json
    assert b"agent_transcript_path" not in request_json
    assert UUID(str(response.invocation_id)) == invocation_id
