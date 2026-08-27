"""Unit tests for Hooks v2 transcripts and session runtime."""

from __future__ import annotations

import json
import os
import stat
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from deepagents_code._constants import LOCAL_CONTEXT_MESSAGE_SOURCE
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
from deepagents_code.hooks.transcript import (
    SUBAGENT_TRANSCRIPT_ID_METADATA_KEY,
    TranscriptRecorder,
    TranscriptStore,
    redact_transcript_value,
)

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


def test_transcript_deduplicates_stable_message_identity(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path / "transcripts")
    message = HumanMessage(id="user-1", content="hello")

    store.append_messages("thread", [message, message])
    store.append_messages("thread", [message])

    records = store.materialize("thread").path.read_text(encoding="utf-8").splitlines()
    assert len(records) == 1


def test_materialize_merges_records_written_by_another_store(tmp_path: Path) -> None:
    """Two stores sharing a transcript must not drop each other's records."""
    root = tmp_path / "transcripts"
    first = TranscriptStore(root)
    first.append_messages("thread", [HumanMessage(id="shared-1", content="shared")])
    first.materialize("thread")

    second = TranscriptStore(root)
    second.append_messages("thread", [HumanMessage(id="other-1", content="other")])
    second.materialize("thread")

    first.append_messages("thread", [HumanMessage(id="mine-1", content="mine")])
    handle = first.materialize("thread")

    records = [
        json.loads(line)
        for line in handle.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["message_id"] for record in records] == [
        "shared-1",
        "mine-1",
        "other-1",
    ]
    assert [record["sequence"] for record in records] == [0, 1, 2]


def test_stream_recorder_collects_completed_main_and_identified_subagent(
    tmp_path: Path,
) -> None:
    runtime = HooksRuntime.create(
        cwd=tmp_path,
        config_dir=tmp_path / "config",
        transcript_root=tmp_path / "transcripts",
    )
    recorder = TranscriptRecorder(runtime, "thread")
    recorder.record(AIMessageChunk(id="main-1", content="hel"), {}, main_agent=True)
    recorder.record(
        AIMessageChunk(id="main-1", content="lo", chunk_position="last"),
        {},
        main_agent=True,
    )
    recorder.record(
        AIMessage(id="sub-1", content="research"),
        {SUBAGENT_TRANSCRIPT_ID_METADATA_KEY: "agent-1"},
        main_agent=False,
    )
    recorder.record(AIMessage(id="unstable", content="skip"), {}, main_agent=False)
    for source in (
        "summarization",
        "auto_mode_classifier",
        LOCAL_CONTEXT_MESSAGE_SOURCE,
    ):
        recorder.record(
            AIMessage(id=source, content=f"hidden {source}"),
            {"lc_source": source},
            main_agent=True,
        )
    recorder.record(
        HumanMessage(
            id="context-message",
            content="hidden message metadata",
            additional_kwargs={"lc_source": LOCAL_CONTEXT_MESSAGE_SOURCE},
        ),
        {},
        main_agent=True,
    )
    chunk_without_metadata = Mock(spec=AIMessageChunk)
    chunk_without_metadata.id = "mock-chunk"
    chunk_without_metadata.content = "ignored"
    recorder.record(chunk_without_metadata, {}, main_agent=True)  # ty: ignore[invalid-argument-type]

    main = runtime.transcripts.materialize("thread").path.read_text()
    agent = runtime.transcripts.materialize(
        "thread", agent_id="agent-1"
    ).path.read_text()

    assert '"content":"hello"' in main
    assert '"content":"research"' in agent
    assert all(
        value not in main + agent
        for value in (
            "skip",
            "hidden summarization",
            "hidden auto",
            "hidden local",
            "hidden message",
        )
    )


def _recorder_runtime(tmp_path: Path) -> tuple[HooksRuntime, TranscriptRecorder]:
    runtime = HooksRuntime.create(
        cwd=tmp_path,
        config_dir=tmp_path / "config",
        transcript_root=tmp_path / "transcripts",
    )
    return runtime, TranscriptRecorder(runtime, "thread")


def _read_main(runtime: HooksRuntime) -> str:
    return runtime.transcripts.materialize("thread").path.read_text()


def _read_agent(runtime: HooksRuntime, agent_id: str) -> str:
    return runtime.transcripts.materialize("thread", agent_id=agent_id).path.read_text()


def test_attempt_scope_stages_until_complete(tmp_path: Path) -> None:
    runtime, recorder = _recorder_runtime(tmp_path)
    recorder.start_attempt(agent_id=None, call_id="call-1", attempt=1)
    recorder.record(AIMessage(id="staged-1", content="staged one"), {}, main_agent=True)
    recorder.record(AIMessage(id="staged-2", content="staged two"), {}, main_agent=True)

    assert "staged one" not in _read_main(runtime)
    assert "staged two" not in _read_main(runtime)

    recorder.complete_attempt(agent_id=None, call_id="call-1", attempt=1)
    main = _read_main(runtime)

    assert '"content":"staged one"' in main
    assert '"content":"staged two"' in main


def test_attempt_retry_discard_including_last_chunk(tmp_path: Path) -> None:
    runtime, recorder = _recorder_runtime(tmp_path)
    recorder.start_attempt(agent_id=None, call_id="call-1", attempt=1)
    recorder.record(AIMessageChunk(id="c1", content="hel"), {}, main_agent=True)
    recorder.record(
        AIMessageChunk(id="c1", content="lo", chunk_position="last"),
        {},
        main_agent=True,
    )
    recorder.record(AIMessage(id="final-1", content="whole"), {}, main_agent=True)

    recorder.discard_attempt(agent_id=None, call_id="call-1", attempt=1)

    main = _read_main(runtime)
    assert "hello" not in main
    assert "whole" not in main
    assert not recorder._attempts
    assert not recorder._chunks

    recorder.start_attempt(agent_id=None, call_id="call-1", attempt=2)
    recorder.record(AIMessage(id="retry-1", content="retry ok"), {}, main_agent=True)
    recorder.complete_attempt(agent_id=None, call_id="call-1", attempt=2)

    assert '"content":"retry ok"' in _read_main(runtime)


def test_attempt_discard_cleans_partial_chunks(tmp_path: Path) -> None:
    runtime, recorder = _recorder_runtime(tmp_path)
    recorder.start_attempt(agent_id=None, call_id="call-1", attempt=1)
    recorder.record(AIMessageChunk(id="c1", content="dangling"), {}, main_agent=True)

    assert recorder._chunks

    recorder.discard_attempt(agent_id=None, call_id="call-1", attempt=1)

    assert not recorder._chunks
    assert "dangling" not in _read_main(runtime)

    recorder.record(AIMessage(id="after", content="after"), {}, main_agent=True)
    main = _read_main(runtime)
    assert "dangling" not in main
    assert '"content":"after"' in main


def test_attempt_lifecycle_duplicate_and_mismatch_are_idempotent(
    tmp_path: Path,
) -> None:
    runtime, recorder = _recorder_runtime(tmp_path)

    recorder.complete_attempt(agent_id=None, call_id="call-1", attempt=1)
    recorder.discard_attempt(agent_id=None, call_id="call-1", attempt=1)

    recorder.start_attempt(agent_id=None, call_id="call-1", attempt=1)
    recorder.record(AIMessage(id="m1", content="kept"), {}, main_agent=True)

    recorder.complete_attempt(agent_id=None, call_id="other-call", attempt=1)
    recorder.complete_attempt(agent_id=None, call_id="call-1", attempt=2)
    recorder.discard_attempt(agent_id="agent-1", call_id="call-1", attempt=1)
    assert recorder._attempts[None].staged
    assert "kept" not in _read_main(runtime)

    recorder.complete_attempt(agent_id=None, call_id="call-1", attempt=1)
    assert '"content":"kept"' in _read_main(runtime)

    recorder.complete_attempt(agent_id=None, call_id="call-1", attempt=1)
    recorder.discard_attempt(agent_id=None, call_id="call-1", attempt=1)
    assert not recorder._attempts


def test_attempt_scopes_are_isolated_per_agent(tmp_path: Path) -> None:
    runtime, recorder = _recorder_runtime(tmp_path)
    recorder.start_attempt(agent_id=None, call_id="call-m", attempt=1)
    recorder.start_attempt(agent_id="agent-1", call_id="call-a", attempt=1)
    recorder.record(AIMessage(id="m1", content="main staged"), {}, main_agent=True)
    recorder.record(
        AIMessage(id="a1", content="agent staged"),
        {SUBAGENT_TRANSCRIPT_ID_METADATA_KEY: "agent-1"},
        main_agent=False,
    )

    recorder.discard_attempt(agent_id="agent-1", call_id="call-a", attempt=1)

    agent = _read_agent(runtime, "agent-1")
    assert "agent staged" not in agent
    assert "main staged" not in _read_main(runtime)

    recorder.complete_attempt(agent_id=None, call_id="call-m", attempt=1)

    assert '"content":"main staged"' in _read_main(runtime)
    assert "agent staged" not in _read_agent(runtime, "agent-1")


def test_start_attempt_replaces_scope_and_drop_uncommitted_clears_all(
    tmp_path: Path,
) -> None:
    runtime, recorder = _recorder_runtime(tmp_path)
    recorder.start_attempt(agent_id=None, call_id="call-1", attempt=1)
    recorder.record(AIMessage(id="old", content="old staged"), {}, main_agent=True)
    recorder.record(AIMessageChunk(id="c1", content="old chunk"), {}, main_agent=True)

    recorder.start_attempt(agent_id=None, call_id="call-1", attempt=2)
    assert recorder._attempts[None].attempt == 2

    recorder.complete_attempt(agent_id=None, call_id="call-1", attempt=1)
    assert "old staged" not in _read_main(runtime)

    recorder.record(AIMessage(id="new", content="new staged"), {}, main_agent=True)
    recorder.drop_uncommitted()

    assert not recorder._attempts
    assert not recorder._chunks
    main = _read_main(runtime)
    assert "old staged" not in main
    assert "new staged" not in main

    recorder.record(AIMessage(id="later", content="later"), {}, main_agent=True)
    assert '"content":"later"' in _read_main(runtime)


def test_destructive_scope_replace_and_drop_are_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Losing staged records must never be silent.

    Both paths make the on-screen conversation and the persisted transcript
    diverge, and neither raises. A count in the log is the only way an operator
    can tell that a lifecycle event went missing.
    """
    _runtime, recorder = _recorder_runtime(tmp_path)
    recorder.start_attempt(agent_id=None, call_id="call-1", attempt=0)
    recorder.record(AIMessage(id="a", content="staged"), {}, main_agent=True)

    with caplog.at_level("WARNING"):
        # A start for a different attempt, with no discard first.
        recorder.start_attempt(agent_id=None, call_id="call-1", attempt=1)
    assert "1 staged record(s)" in caplog.text

    recorder.record(AIMessage(id="b", content="staged too"), {}, main_agent=True)
    caplog.clear()
    with caplog.at_level("WARNING"):
        recorder.drop_uncommitted()
    assert "Dropping 1 staged transcript record(s)" in caplog.text


def test_drop_uncommitted_is_quiet_when_nothing_was_staged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Teardown on a clean run is a no-op, so it must not warn."""
    _runtime, recorder = _recorder_runtime(tmp_path)
    recorder.start_attempt(agent_id=None, call_id="call-1", attempt=0)
    recorder.record(AIMessage(id="a", content="committed"), {}, main_agent=True)
    recorder.complete_attempt(agent_id=None, call_id="call-1", attempt=0)

    with caplog.at_level("WARNING"):
        recorder.drop_uncommitted()

    assert "Dropping" not in caplog.text


def test_attempt_last_chunk_not_materialized_until_complete(tmp_path: Path) -> None:
    runtime, recorder = _recorder_runtime(tmp_path)
    recorder.start_attempt(agent_id=None, call_id="call-1", attempt=1)
    recorder.record(AIMessageChunk(id="c1", content="hel"), {}, main_agent=True)
    recorder.record(
        AIMessageChunk(id="c1", content="lo", chunk_position="last"),
        {},
        main_agent=True,
    )

    assert not recorder._chunks
    assert "hello" not in _read_main(runtime)

    recorder.complete_attempt(agent_id=None, call_id="call-1", attempt=1)
    assert '"content":"hello"' in _read_main(runtime)


def test_attempt_records_without_lifecycle_keep_direct_append(
    tmp_path: Path,
) -> None:
    runtime, recorder = _recorder_runtime(tmp_path)
    recorder.record(AIMessage(id="direct", content="direct"), {}, main_agent=True)
    recorder.record(
        AIMessage(id="sub-direct", content="sub direct"),
        {SUBAGENT_TRANSCRIPT_ID_METADATA_KEY: "agent-1"},
        main_agent=False,
    )

    assert '"content":"direct"' in _read_main(runtime)
    assert '"content":"sub direct"' in _read_agent(runtime, "agent-1")
    assert not recorder._attempts
    assert not recorder._chunks


def test_checkpoint_append_hides_local_context(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path / "transcripts")
    store.append_messages(
        "thread",
        [
            HumanMessage(
                id="local-context",
                content="hidden context",
                additional_kwargs={"lc_source": LOCAL_CONTEXT_MESSAGE_SOURCE},
            ),
            HumanMessage(id="user", content="visible input"),
        ],
    )

    transcript = store.materialize("thread").path.read_text()

    assert "hidden context" not in transcript
    assert "visible input" in transcript


def test_runtime_stores_transcripts_outside_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    config_dir = tmp_path / "config"
    global_dir = tmp_path / "global-deepagents"
    workspace.mkdir()
    monkeypatch.setattr(
        "deepagents_code.hooks.runtime.DEFAULT_CONFIG_DIR",
        global_dir,
    )

    runtime = HooksRuntime.create(cwd=workspace, config_dir=config_dir)

    assert runtime.transcripts.root == (global_dir / "transcripts").resolve()
    assert not (workspace / ".deepagents").exists()
    assert not (config_dir / "transcripts").exists()


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
    runtime = HooksRuntime.create(
        cwd=tmp_path,
        config_dir=config_dir,
        transcript_root=tmp_path / "transcripts",
    )
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
