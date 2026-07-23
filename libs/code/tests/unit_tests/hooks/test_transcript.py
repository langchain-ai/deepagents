"""Unit tests for Hooks v2 transcripts and session runtime."""

from __future__ import annotations

import json
import os
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deepagents_code.approval_mode import ApprovalMode
from deepagents_code.hooks.models.domain import (
    AgentIdentity,
    HookContext,
    HookEvent,
    HookInvocation,
    SessionStartCause,
    SessionStartDecision,
    SessionStartEvent,
    SubagentStopEvent,
)
from deepagents_code.hooks.runtime import HooksRuntime
from deepagents_code.hooks.transcript import TranscriptStore, redact_transcript_value

if TYPE_CHECKING:
    from pathlib import Path


def test_transcript_store_permissions_atomicity_revision_redaction(
    tmp_path: Path,
) -> None:
    store = TranscriptStore(tmp_path / "transcripts", retention_revisions=2)
    store.append_messages(
        "thread-a",
        [
            HumanMessage(
                content=(
                    "token OPENAI_API_KEY=placeholder "
                    "https://example.com?access_token=opaque"
                )
            ),
            AIMessage(content="done"),
        ],
    )
    handle = store.materialize("thread-a")

    assert handle.path.is_file()
    assert handle.path.is_absolute()
    if os.name != "nt":
        assert stat.S_IMODE(handle.path.stat().st_mode) == 0o600
    lines = handle.path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert "placeholder" not in first["content"]
    assert "opaque" not in first["content"]
    assert "[redacted]" in first["content"]
    assert first["sequence"] == 0
    assert handle.revision == store.revision("thread-a")
    assert "tool_calls" not in first

    previous = handle.path.read_text(encoding="utf-8")
    store.append_messages("thread-a", [HumanMessage(content="again")])
    second = store.materialize("thread-a")

    assert second.revision != handle.revision
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
    url = "https://user:password@example.com/path?access_token=opaque#fragment"
    webhook_secret = "T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
    webhook = f"https://hooks.slack.com/services/{webhook_secret}"
    redacted = redact_transcript_value(f"{bare_token} {bearer} {url} {webhook}")

    assert isinstance(redacted, str)
    assert bare_token not in redacted
    assert bearer not in redacted
    assert "user:password" not in redacted
    assert webhook_secret not in redacted
    assert "opaque" not in redacted
    assert "fragment" not in redacted
    assert redacted.count("[redacted]") >= 2
    assert "%5Bredacted%5D" in redacted
    assert "https://hooks.slack.com/[redacted]" in redacted


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


def test_runtime_stores_transcripts_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    config_dir = tmp_path / "config"
    workspace.mkdir()

    runtime = HooksRuntime.create(cwd=workspace, config_dir=config_dir)

    assert runtime.transcripts.root == (config_dir / "transcripts").resolve()
    assert not (workspace / ".deepagents").exists()


async def test_runtime_materializes_paths_and_invokes(tmp_path: Path) -> None:
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    command = (
        "import json,sys; "
        "payload=json.load(sys.stdin); "
        "open(payload['transcript_path']).read(); "
        "print(json.dumps({"
        "'systemMessage':'ok',"
        "'hookSpecificOutput':{"
        "'hookEventName':'SessionStart',"
        "'additionalContext':'from-hook'"
        "}}))"
    )
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
                                        f"{sys.executable} -c {json.dumps(command)}"
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

    decision = await runtime.invoke(invocation)
    prepared = runtime.prepare_invocation(invocation)

    assert isinstance(decision, SessionStartDecision)
    assert decision.user_notices == ["ok"]
    assert decision.context == ["from-hook"]
    assert runtime.snapshot_id
    assert prepared.transcript_path == runtime.transcripts.thread_path("thread-1")
    assert prepared.transcript_path.is_file()
    assert "transcript_path" not in invocation.context.model_fields_set

    agent = AgentIdentity(id="agent-1", name="researcher")
    prepared_subagent = runtime.prepare_invocation(
        HookInvocation(
            context=invocation.context,
            event=SubagentStopEvent(
                event=HookEvent.SUBAGENT_STOP,
                agent=agent,
                continuation_count=0,
                last_assistant_message="done",
            ),
        )
    )
    assert prepared_subagent.agent_transcript_path is not None
    assert prepared_subagent.agent_transcript_path.is_file()
    assert prepared_subagent.agent_transcript_path.is_relative_to(
        runtime.transcripts.root
    )
