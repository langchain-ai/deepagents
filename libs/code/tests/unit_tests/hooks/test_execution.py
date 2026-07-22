"""Unit tests for Hooks v2 execution safety and policies."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import pytest

from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.capabilities import (
    ExitCodePolicy,
    PlainOutputPolicy,
    get_event_spec,
)
from deepagents_code.hooks.env import sanitize_hook_environ
from deepagents_code.hooks.models.config import HooksConfig
from deepagents_code.hooks.models.domain import (
    HookContext,
    HookEvent,
    HookInvocation,
    SessionStartCause,
    SessionStartDecision,
    SessionStartEvent,
    ToolCallData,
)
from deepagents_code.hooks.models.wire import HookWireOutput
from deepagents_code.hooks.reducer import MAX_STOP_CONTINUATIONS, reduce_hook_results
from deepagents_code.hooks.runner import HandlerResult, run_command_handler
from deepagents_code.hooks.snapshot import HooksSnapshot
from deepagents_code.hooks.tools import format_mcp_wire_name, to_wire_call
from deepagents_code.hooks.validate_terminal_sequence import validate_terminal_sequence

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.hooks.snapshot import HookHandler


def _invocation(tmp_path: Path) -> HookInvocation:
    return HookInvocation(
        context=HookContext(
            thread_id="thread",
            cwd=tmp_path,
            approval_mode=ApprovalMode.MANUAL,
        ),
        event=SessionStartEvent(
            event=HookEvent.SESSION_START,
            cause=SessionStartCause.STARTUP,
        ),
    )


def _handler(command: str, *, timeout: float) -> HookHandler:
    snapshot = HooksSnapshot.from_config(
        HooksConfig.model_validate(
            {
                "hooks": {
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
            }
        )
    )
    return snapshot.handlers[HookEvent.SESSION_START][0]


def test_terminal_sequence_allowlist() -> None:
    assert validate_terminal_sequence("\x1b]0;title\x07") == "\x1b]0;title\x07"
    assert validate_terminal_sequence("\x1b]9;hello\x07") == "\x1b]9;hello\x07"
    assert validate_terminal_sequence("\x07") == "\x07"
    assert validate_terminal_sequence("\x1b]8;;https://example.com\x07") is None
    assert validate_terminal_sequence("\x1b]9;line\nbreak\x07") is None
    assert validate_terminal_sequence("\x1b[31mred\x1b[0m") is None


def test_reducer_rejects_invalid_terminal_and_deferred_fields(
    tmp_path: Path,
) -> None:
    decision = reduce_hook_results(
        _invocation(tmp_path),
        [
            HandlerResult(
                handler_id="one",
                output=HookWireOutput.model_validate(
                    {
                        "terminalSequence": "\x1b[31mnope",
                        "customField": "x",
                        "hookSpecificOutput": {
                            "hookEventName": "SessionStart",
                            "additionalContext": "ok",
                            "sessionTitle": "Nope",
                            "reloadSkills": True,
                        },
                    }
                ),
            )
        ],
    )

    assert isinstance(decision, SessionStartDecision)
    assert decision.terminal_sequences == []
    assert decision.context == ["ok"]
    codes = {item.code for item in decision.diagnostics}
    assert "invalid_terminal_sequence" in codes
    assert "unsupported_field" in codes


def test_sanitized_env_strips_secrets_from_injected_source() -> None:
    env = sanitize_hook_environ(
        {
            "SAFE_PATH": "/tmp",
            "OPENAI_API_KEY": "placeholder",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost",
            "MY_TOKEN": "placeholder",
            "mixed_case_secret": "placeholder",
            "PYTHONPATH": "/opt/lib",
            "HOME": "/home/user",
        }
    )

    assert env["SAFE_PATH"] == "/tmp"
    assert env["OTEL_EXPORTER_OTLP_ENDPOINT"] == "http://localhost"
    assert env["PYTHONPATH"] == "/opt/lib"
    assert env["HOME"] == "/home/user"
    assert "OPENAI_API_KEY" not in env
    assert "MY_TOKEN" not in env
    assert "mixed_case_secret" not in env


def test_mcp_tool_mapping_requires_resolved_metadata() -> None:
    assert format_mcp_wire_name("github", "create_issue") == "mcp__github__create_issue"
    call = ToolCallData(
        id="1",
        name="create_issue",
        args={"title": "x"},
        mcp_server="github",
    )

    assert to_wire_call(call) == ("mcp__github__create_issue", {"title": "x"})
    assert to_wire_call(ToolCallData(id="2", name="github_create_issue", args={})) == (
        "github_create_issue",
        {},
    )


def test_exit_and_plain_output_policies_match_registry() -> None:
    assert (
        get_event_spec(HookEvent.PRE_TOOL_USE).exit_code_policy is ExitCodePolicy.DENY
    )
    assert (
        get_event_spec(HookEvent.POST_TOOL_USE).exit_code_policy
        is ExitCodePolicy.FEEDBACK
    )
    assert (
        get_event_spec(HookEvent.STOP).exit_code_policy is ExitCodePolicy.CONTINUE_LOOP
    )
    assert (
        get_event_spec(HookEvent.SESSION_START).plain_output_policy
        is PlainOutputPolicy.CONTEXT
    )
    assert MAX_STOP_CONTINUATIONS == 8


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-specific")
async def test_runner_kills_process_group_on_timeout(tmp_path: Path) -> None:
    script = tmp_path / "hook.sh"
    side_effect = tmp_path / "survived"
    script.write_text(
        f"#!/bin/sh\n(sleep 0.2; touch {side_effect}) &\nwait\n",
        encoding="utf-8",
    )
    script.chmod(0o755)

    result = await run_command_handler(
        _handler(str(script), timeout=0.05),
        b"{}",
        cwd=tmp_path,
        default_timeout=0.05,
    )

    assert [item.code for item in result.diagnostics] == ["timeout"]
    await asyncio.sleep(0.3)
    assert not side_effect.exists()


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-specific")
async def test_runner_kills_process_group_on_cancellation(tmp_path: Path) -> None:
    script = tmp_path / "cancel.sh"
    ready = tmp_path / "ready"
    side_effect = tmp_path / "survived"
    script.write_text(
        f"#!/bin/sh\ntouch {ready}\n(sleep 0.2; touch {side_effect}) &\nwait\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    task = asyncio.create_task(
        run_command_handler(
            _handler(str(script), timeout=30),
            b"{}",
            cwd=tmp_path,
            default_timeout=30,
        )
    )
    for _ in range(50):
        if ready.exists():
            break
        await asyncio.sleep(0.01)
    assert ready.exists()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.3)
    assert not side_effect.exists()


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-specific")
async def test_runner_kills_descendants_after_shell_exits(tmp_path: Path) -> None:
    script = tmp_path / "exited.sh"
    side_effect = tmp_path / "survived"
    script.write_text(
        f"#!/bin/sh\n(sleep 0.2; touch {side_effect}) &\nexit 0\n",
        encoding="utf-8",
    )
    script.chmod(0o755)

    result = await run_command_handler(
        _handler(str(script), timeout=0.05),
        b"{}",
        cwd=tmp_path,
        default_timeout=0.05,
    )

    assert [item.code for item in result.diagnostics] == ["timeout"]
    await asyncio.sleep(0.3)
    assert not side_effect.exists()
