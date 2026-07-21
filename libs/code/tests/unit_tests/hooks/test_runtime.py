"""Unit tests for Hooks v2 session runtime foundation (DCD-70)."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import ValidationError

from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.capabilities import (
    HOOK_EVENT_SPECS,
    assert_hook_event_registry_complete,
    get_event_spec,
)
from deepagents_code.hooks.env import is_secret_env_name, sanitize_hook_environ
from deepagents_code.hooks.loading import (
    canonical_hooks_bytes,
    compute_snapshot_id,
    load_hooks_config,
)
from deepagents_code.hooks.migration import migrate_legacy_hooks
from deepagents_code.hooks.models.config import HooksConfig
from deepagents_code.hooks.models.domain import (
    AgentIdentity,
    HookContext,
    HookEvent,
    HookInvocation,
    SessionStartCause,
    SessionStartDecision,
    SessionStartEvent,
    SubagentStartEvent,
    ToolCallData,
)
from deepagents_code.hooks.models.wire import HookWireOutput
from deepagents_code.hooks.reducer import MAX_STOP_CONTINUATIONS, reduce_hook_results
from deepagents_code.hooks.runner import HandlerResult, run_command_handler
from deepagents_code.hooks.runtime import HooksRuntime
from deepagents_code.hooks.snapshot import HooksSnapshot
from deepagents_code.hooks.terminal import validate_terminal_sequence
from deepagents_code.hooks.tools import format_mcp_wire_name, to_wire_call
from deepagents_code.hooks.transcript import TranscriptStore, redact_transcript_value

if TYPE_CHECKING:
    from pathlib import Path


def test_registry_covers_all_hook_events() -> None:
    assert_hook_event_registry_complete()
    assert set(HOOK_EVENT_SPECS) == set(HookEvent)
    assert get_event_spec(
        HookEvent.SESSION_END
    ).default_timeout_seconds == pytest.approx(600.0)
    assert get_event_spec(HookEvent.PERMISSION_REQUEST).matcher_field == "tool_name"


def test_load_hooks_config_precedence_and_snapshot_hash(tmp_path: Path) -> None:
    user_dir = tmp_path / "user"
    project_dir = tmp_path / "project"
    user_dir.mkdir()
    (project_dir / ".deepagents").mkdir(parents=True)
    (user_dir / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {"hooks": [{"type": "command", "command": "user-hook"}]}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    (project_dir / ".deepagents" / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {"hooks": [{"type": "command", "command": "project-hook"}]}
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_hooks_config(cwd=project_dir, config_dir=user_dir)
    groups = loaded.config.hooks[HookEvent.SESSION_START]

    assert [group.hooks[0].command for group in groups] == [
        "project-hook",
        "user-hook",
    ]
    assert loaded.snapshot_id == compute_snapshot_id(loaded.config)
    assert loaded.snapshot_id == compute_snapshot_id(
        HooksConfig.model_validate(
            {
                "hooks": {
                    "SessionStart": [
                        {"hooks": [{"type": "command", "command": "project-hook"}]},
                        {"hooks": [{"type": "command", "command": "user-hook"}]},
                    ]
                }
            }
        )
    )
    assert canonical_hooks_bytes(loaded.config).startswith(b'{"hooks":')


def test_legacy_migration_only_maps_exact_session_end_semantics(
    tmp_path: Path,
) -> None:
    migrated = migrate_legacy_hooks(
        [
            {"command": ["echo", "start"], "events": ["session.start"]},
            {"command": ["echo", "tool"], "events": ["tool.use"]},
            {"command": ["echo", "result"], "events": ["tool.result"]},
            {"command": ["echo", "end"], "events": ["session.end"]},
            {"command": ["echo", "perm"], "events": ["permission.request"]},
            {"command": ["echo", "compact"], "events": ["context.compact"]},
        ]
    )

    assert set(migrated.hooks) == {HookEvent.SESSION_END}
    assert HookEvent.PRE_TOOL_USE not in migrated.hooks
    assert HookEvent.POST_TOOL_USE not in migrated.hooks

    user_dir = tmp_path / "user"
    user_dir.mkdir()
    (user_dir / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": [
                    {"command": ["echo", "start"], "events": ["session.start"]},
                    {"command": ["echo", "tool"], "events": ["tool.use"]},
                    {"command": ["echo", "end"], "events": ["session.end"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    loaded = load_hooks_config(cwd=tmp_path, config_dir=user_dir)
    assert HookEvent.SESSION_START not in loaded.config.hooks
    assert HookEvent.SESSION_END in loaded.config.hooks
    assert HookEvent.PRE_TOOL_USE not in loaded.config.hooks
    assert any(item.code == "legacy_migrated" for item in loaded.diagnostics)


def test_async_command_config_is_rejected() -> None:
    with pytest.raises(ValidationError, match="async"):
        HooksConfig.model_validate(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "echo",
                                    "async": True,
                                }
                            ]
                        }
                    ]
                }
            }
        )


def test_snapshot_id_is_immutable_and_stable() -> None:
    config = HooksConfig.model_validate(
        {
            "hooks": {
                "PreToolUse": [
                    {"hooks": [{"type": "command", "command": "policy"}]},
                ]
            }
        }
    )
    first = HooksSnapshot.from_config(config)
    second = HooksSnapshot.from_config(config)

    assert first.snapshot_id == second.snapshot_id
    assert len(first.snapshot_id) == 64

    with_false_async = HooksConfig.model_validate(
        {
            "hooks": {
                "PreToolUse": [
                    {
                        "hooks": [
                            {
                                "type": "command",
                                "command": "policy",
                                "async": False,
                            }
                        ]
                    }
                ]
            }
        }
    )
    assert compute_snapshot_id(with_false_async) == first.snapshot_id
    assert with_false_async.hooks[HookEvent.PRE_TOOL_USE][0].hooks[0].async_ is None
    with pytest.raises(ValueError, match="does not match"):
        HooksSnapshot.from_config(config, snapshot_id="not-canonical")


def test_terminal_sequence_allowlist() -> None:
    assert validate_terminal_sequence("\x1b]0;title\x07") == "\x1b]0;title\x07"
    assert validate_terminal_sequence("\x1b]9;hello\x07") == "\x1b]9;hello\x07"
    assert validate_terminal_sequence("\x07") == "\x07"
    assert validate_terminal_sequence("\x1b]8;;https://example.com\x07") is None
    assert validate_terminal_sequence("\x1b[31mred\x1b[0m") is None
    assert validate_terminal_sequence("\x1b]0;line\nbreak\x07") is None
    assert validate_terminal_sequence("\x1b]0;hidden\x00value\x07") is None
    assert validate_terminal_sequence("\x1b]0;c1\x85value\x07") is None


def test_reducer_rejects_invalid_terminal_and_deferred_fields(tmp_path: Path) -> None:
    invocation = HookInvocation(
        context=HookContext(
            thread_id="t",
            cwd=tmp_path,
            approval_mode=ApprovalMode.MANUAL,
        ),
        event=SessionStartEvent(
            event=HookEvent.SESSION_START,
            cause=SessionStartCause.STARTUP,
        ),
    )
    decision = reduce_hook_results(
        invocation,
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


async def test_session_start_plain_stdout_becomes_context(tmp_path: Path) -> None:
    runtime = HooksRuntime.create(cwd=tmp_path, config_dir=tmp_path / "missing")
    code = "print('hello context')"
    snapshot = HooksSnapshot.from_config(
        HooksConfig.model_validate(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": (
                                        f"{sys.executable} -c {json.dumps(code)}"
                                    ),
                                }
                            ]
                        }
                    ]
                }
            }
        )
    )
    runtime = HooksRuntime(
        snapshot=snapshot,
        transcripts=runtime.transcripts,
        engine=runtime.engine.__class__(snapshot),
        cwd=tmp_path,
    )
    decision = await runtime.invoke(
        HookInvocation(
            context=HookContext(
                thread_id="thread-1",
                cwd=tmp_path,
                approval_mode=ApprovalMode.MANUAL,
            ),
            event=SessionStartEvent(
                event=HookEvent.SESSION_START,
                cause=SessionStartCause.STARTUP,
            ),
        )
    )
    assert isinstance(decision, SessionStartDecision)
    assert decision.context == ["hello context"]


def test_sanitized_env_strips_secrets_and_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAFE_PATH", "/tmp")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost")
    monkeypatch.setenv("MY_TOKEN", "secret")
    monkeypatch.setenv("PYTHONPATH", "/opt/lib")
    monkeypatch.setenv("HOME", "/home/user")

    env = sanitize_hook_environ()

    assert env["SAFE_PATH"] == "/tmp"
    assert env["PYTHONPATH"] == "/opt/lib"
    assert env["HOME"] == "/home/user"
    assert "OPENAI_API_KEY" not in env
    assert "OTEL_EXPORTER_OTLP_ENDPOINT" not in env
    assert "MY_TOKEN" not in env
    assert is_secret_env_name("ANTHROPIC_API_KEY")
    assert is_secret_env_name("openai_api_key")


def test_mcp_and_extended_native_tool_mapping() -> None:
    assert format_mcp_wire_name("github", "create_issue") == "mcp__github__create_issue"
    call = ToolCallData(
        id="1",
        name="create_issue",
        args={"title": "x"},
        mcp_server="github",
    )
    assert to_wire_call(call) == ("mcp__github__create_issue", {"title": "x"})
    assert to_wire_call(
        ToolCallData(
            id="2",
            name="github_execute",
            args={"program": "safe"},
            mcp_server="github",
        )
    ) == ("mcp__github__execute", {"program": "safe"})
    assert to_wire_call(
        ToolCallData(id="3", name="web_search", args={"query": "q"})
    ) == (
        "web_search",
        {"query": "q"},
    )


def test_transcript_store_permissions_atomicity_revision_redaction(
    tmp_path: Path,
) -> None:
    store = TranscriptStore(tmp_path / "transcripts", retention_revisions=2)
    store.append_messages(
        "thread-a",
        [
            HumanMessage(
                content="token OPENAI_API_KEY=sk-secret https://example.com?x=1"
            ),
            AIMessage(content="done"),
        ],
    )
    handle = store.materialize("thread-a")

    assert handle.path.is_file()
    assert handle.path.is_absolute()
    if os.name != "nt":
        mode = stat.S_IMODE(handle.path.stat().st_mode)
        assert mode == 0o600
    lines = handle.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert "sk-secret" not in first["content"]
    assert "[redacted]" in first["content"]
    assert first["sequence"] == 0
    assert handle.revision == store.revision("thread-a")
    assert "tool_calls" not in first
    assert set(first) <= {
        "schema_version",
        "sequence",
        "record_id",
        "timestamp",
        "thread_id",
        "agent_id",
        "role",
        "message_id",
        "content",
        "name",
    }

    previous = handle.path.read_text(encoding="utf-8")
    store.append_messages("thread-a", [HumanMessage(content="again")])
    second = store.materialize("thread-a")
    assert second.revision != handle.revision
    assert second.path.is_file()
    assert previous != second.path.read_text(encoding="utf-8")
    backups = list(handle.path.parent.glob(f"{handle.path.name}.bak-*"))
    assert backups
    assert backups[0].read_text(encoding="utf-8") == previous
    assert backups[0].name.endswith(handle.revision)

    agent = store.materialize("thread-a", agent_id="agent-1")
    assert agent.path == store.agent_path("thread-a", "agent-1")
    assert agent.path.is_absolute()
    assert agent.path.is_file()

    redacted = redact_transcript_value({"token": "placeholder"})
    assert redacted == {"token": "[redacted]"}


def test_transcript_paths_are_safe_unique_and_private(tmp_path: Path) -> None:
    root = tmp_path / "permissive"
    root.mkdir(mode=0o777)
    if os.name != "nt":
        root.chmod(0o777)
    store = TranscriptStore(root)

    identifiers = ["../escape", "a/b", "a\\b", "é", "e\u0301", "same"]
    paths = [store.thread_path(identifier) for identifier in identifiers]

    assert len(set(paths)) == len(identifiers)
    assert all(path.parent == store.root for path in paths)
    assert all(".." not in path.name and "/" not in path.name for path in paths)

    agent = store.materialize("../escape", agent_id="../../agent")
    assert agent.path.is_relative_to(store.root)
    assert agent.path.is_file()
    if os.name != "nt":
        assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
        assert stat.S_IMODE(agent.path.parent.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(agent.path.parent.stat().st_mode) == 0o700

    with pytest.raises(ValueError, match="nonnegative"):
        TranscriptStore(tmp_path / "invalid", retention_revisions=-1)


def test_transcript_redaction_covers_tokens_and_urls() -> None:
    bare_token = "sk-" + ("x" * 24)
    bearer = "Bearer " + ("y" * 24)
    url = "https://user:password@example.com/path?access_token=opaque#private-fragment"
    redacted = redact_transcript_value(f"{bare_token} {bearer} {url}")

    assert isinstance(redacted, str)
    assert bare_token not in redacted
    assert bearer not in redacted
    assert "user:password" not in redacted
    assert "opaque" not in redacted
    assert "private-fragment" not in redacted
    assert redacted.count("[redacted]") >= 2
    assert "%5Bredacted%5D" in redacted


def test_transcript_repairs_corrupt_existing_file_permissions(tmp_path: Path) -> None:
    root = tmp_path / "transcripts"
    initial = TranscriptStore(root)
    path = initial.thread_path("thread")
    path.write_text("{invalid json}\n", encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o644)

    reloaded = TranscriptStore(root)
    handle = reloaded.materialize("thread")

    assert handle.path.read_text(encoding="utf-8") == ""
    assert handle.revision == reloaded.revision("thread")
    if os.name != "nt":
        assert stat.S_IMODE(handle.path.stat().st_mode) == 0o600


def test_transcript_revision_is_deterministic_and_thread_safe(tmp_path: Path) -> None:
    messages = [
        HumanMessage(id="user-1", content="first"),
        AIMessage(id="assistant-1", content="second"),
    ]
    first = TranscriptStore(tmp_path / "first")
    second = TranscriptStore(tmp_path / "second")
    first.append_messages("thread", messages)
    second.append_messages("thread", messages)
    first_handle = first.materialize("thread")
    second_handle = second.materialize("thread")

    assert first_handle.revision == second_handle.revision
    assert first_handle.path.read_bytes() == second_handle.path.read_bytes()

    concurrent = TranscriptStore(tmp_path / "concurrent")

    def append(index: int) -> None:
        concurrent.append_messages(
            "thread",
            [HumanMessage(id=f"message-{index}", content=str(index))],
        )
        concurrent.materialize("thread")

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(append, range(40)))

    handle = concurrent.materialize("thread")
    records = [
        json.loads(line)
        for line in handle.path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 40
    assert [record["sequence"] for record in records] == list(range(40))
    assert len({record["message_id"] for record in records}) == 40
    assert handle.revision == concurrent.revision("thread")


async def test_runtime_materializes_paths_and_invokes(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "hooks.json").write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": (
                                        f"{sys.executable} -c "
                                        + json.dumps(
                                            "import json,sys; "
                                            "payload=json.load(sys.stdin); "
                                            "path=payload['transcript_path']; "
                                            "open(path).read(); "
                                            "print(json.dumps({"
                                            "'systemMessage':'ok',"
                                            "'hookSpecificOutput':{"
                                            "'hookEventName':'SessionStart',"
                                            "'additionalContext':'from-hook'"
                                            "}}))"
                                        )
                                    ),
                                }
                            ]
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    runtime = HooksRuntime.create(cwd=tmp_path, config_dir=config_dir)
    runtime.append_messages("thread-1", [HumanMessage(content="hi")])

    invocation = HookInvocation(
        context=HookContext(
            thread_id="thread-1",
            cwd=tmp_path,
            approval_mode=ApprovalMode.MANUAL,
        ),
        event=SessionStartEvent(
            event=HookEvent.SESSION_START,
            cause=SessionStartCause.STARTUP,
        ),
    )
    prepared = runtime.prepare_invocation(invocation)
    decision = await runtime.invoke(invocation)

    assert isinstance(decision, SessionStartDecision)
    assert decision.user_notices == ["ok"]
    assert decision.context == ["from-hook"]
    assert runtime.snapshot_id
    assert prepared.invocation is invocation
    assert prepared.transcript_path == runtime.transcripts.thread_path("thread-1")
    assert prepared.transcript_path.is_file()
    assert "transcript_path" not in invocation.context.model_fields_set

    subagent = HookInvocation(
        context=invocation.context,
        event=SubagentStartEvent(
            event=HookEvent.SUBAGENT_START,
            agent=AgentIdentity(id="../agent", name="researcher"),
        ),
    )
    prepared_subagent = runtime.prepare_invocation(subagent)
    assert prepared_subagent.agent_transcript_path is not None
    assert prepared_subagent.agent_transcript_path.is_file()
    assert prepared_subagent.agent_transcript_path.is_relative_to(
        runtime.transcripts.root
    )


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-specific")
async def test_runner_kills_process_group_on_timeout(tmp_path: Path) -> None:
    # Spawn a child that starts its own grandchild; killing only the shell would
    # leave the grandchild. start_new_session + killpg must reap both.
    script = tmp_path / "hook.sh"
    grandchild_pid = tmp_path / "grandchild.pid"
    script.write_text(
        f"#!/bin/sh\nsleep 30 &\necho $! > {grandchild_pid}\nwait\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    snapshot = HooksSnapshot.from_config(
        HooksConfig.model_validate(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": str(script),
                                    "timeout": 0.05,
                                }
                            ]
                        }
                    ]
                }
            }
        )
    )
    handler = snapshot.handlers[HookEvent.SESSION_START][0]

    result = await run_command_handler(
        handler,
        b"{}",
        cwd=tmp_path,
        default_timeout=0.05,
    )

    assert [item.code for item in result.diagnostics] == ["timeout"]
    if grandchild_pid.is_file():
        pid = int(grandchild_pid.read_text(encoding="utf-8").strip())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_stop_continuation_cap_constant() -> None:
    assert MAX_STOP_CONTINUATIONS == 8


def test_exit_code_policies_match_capability_registry() -> None:
    from deepagents_code.hooks.capabilities import ExitCodePolicy

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
        get_event_spec(HookEvent.SESSION_START).exit_code_policy
        is ExitCodePolicy.DIAGNOSE
    )
    assert (
        get_event_spec(HookEvent.SESSION_START).plain_output_policy.value == "context"
    )


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-specific")
async def test_runner_kills_process_group_on_cancellation(tmp_path: Path) -> None:
    script = tmp_path / "cancel.sh"
    grandchild_pid = tmp_path / "grandchild.pid"
    script.write_text(
        f"#!/bin/sh\nsleep 30 &\necho $! > {grandchild_pid}\nwait\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    snapshot = HooksSnapshot.from_config(
        HooksConfig.model_validate(
            {
                "hooks": {
                    "SessionStart": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": str(script),
                                    "timeout": 30,
                                }
                            ]
                        }
                    ]
                }
            }
        )
    )
    handler = snapshot.handlers[HookEvent.SESSION_START][0]
    task = asyncio.create_task(
        run_command_handler(handler, b"{}", cwd=tmp_path, default_timeout=30)
    )
    for _ in range(50):
        if grandchild_pid.is_file():
            break
        await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    if grandchild_pid.is_file():
        pid = int(grandchild_pid.read_text(encoding="utf-8").strip())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
