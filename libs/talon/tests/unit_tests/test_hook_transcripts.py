from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import HumanMessage

from deepagents_code.hooks.transcript import TranscriptStore
from deepagents_talon.hook_transcripts import ConversationTranscripts, erase_hook_transcripts
from deepagents_talon.hooks import TalonHooks
from tests.archive_helpers import make_runtime, make_saver
from tests.unit_tests.test_archive import OTHER, WHATSAPP, _save

if TYPE_CHECKING:
    from pathlib import Path


def _projections(home: Path, session: str) -> tuple[TalonHooks, list[Path]]:
    (home / "hooks.json").write_text('{"hooks": {}}')
    parent = TalonHooks.load(home, home, session)
    assert parent is not None
    adapters = [parent, parent.child("worker"), parent.child("worker")]
    paths = []
    for adapter in adapters:
        identity = adapter.context["thread_id"]
        adapter.record({"messages": [HumanMessage("first", id="first")]})
        path = adapter.runtime.transcripts.materialize(identity).path
        adapter.record({"messages": [HumanMessage("second", id="second")]})
        adapter.runtime.transcripts.materialize(identity)
        paths.extend([path, *path.parent.glob(f"{path.name}.bak-*")])
    assert len(set(paths)) == 6
    return parent, paths


@pytest.mark.parametrize("clear", [False, True])
async def test_erasure_survives_restart_and_removed_hooks(tmp_path, *, clear):
    database = tmp_path / "history.sqlite"
    parent, paths = _projections(tmp_path, "owned")
    _, foreign = _projections(tmp_path, "foreign")
    _, retained = _projections(tmp_path, "keep")
    legacy = TranscriptStore(tmp_path / "transcripts")
    legacy.append_messages("owned", [HumanMessage("legacy", id="old")])
    old = legacy.materialize("owned").path
    legacy.append_messages("owned", [HumanMessage("revision", id="new")])
    legacy.materialize("owned")
    paths.extend([old, *old.parent.glob(f"{old.name}.bak-*")])
    async with make_saver(database) as saver:
        await _save(saver, "owned", "owned")
        await _save(saver, "keep", "keep")
        await _save(saver, "foreign", "foreign", scope=OTHER)
    (tmp_path / "hooks.json").unlink()
    async with make_saver(database) as saver:
        runtime = make_runtime(saver, tmp_path)
        if clear:
            await runtime.clear_history("whatsapp", "chat")
        else:
            clone = saver.with_allowlist([])
            await clone.delete_conversations(WHATSAPP, ["owned"], current_session="active")
        assert all(not path.exists() for path in paths)
        assert all(path.exists() for path in foreign)
        assert all(path.exists() != clear for path in retained)
        child = parent.child("late worker")
        with pytest.raises(OSError, match="deleted"):
            child.record({"messages": [HumanMessage("late")]})
        with pytest.raises(OSError, match="deleted"):
            parent.runtime.transcripts.materialize("owned")
        with pytest.raises(OSError, match="deleted"):
            ConversationTranscripts(tmp_path, "owned")
        assert await saver.archive.sessions(WHATSAPP) == ([] if clear else ["keep"])


async def test_cleanup_failure_retains_ownership_for_restart_retry(tmp_path, monkeypatch):
    _, paths = _projections(tmp_path, "owned")
    database = tmp_path / "history.sqlite"
    async with make_saver(database) as saver:
        make_runtime(saver, tmp_path)
        await _save(saver, "owned", "owned")
        with monkeypatch.context() as patch:

            def fail(_path):
                msg = "unlink failed"
                raise OSError(msg)

            patch.setattr("deepagents_talon.hook_transcripts._remove", fail)
            with pytest.raises(OSError, match="unlink failed"):
                await saver.delete_conversations(WHATSAPP, ["owned"], current_session="active")
        assert await saver.archive.sessions(WHATSAPP) == ["owned"]
    async with make_saver(database) as saver:
        make_runtime(saver, tmp_path)
        await saver.clear_history(WHATSAPP)
        assert all(not path.exists() for path in paths)
        assert await saver.archive.sessions(WHATSAPP) == []


async def test_inflight_materialization_finishes_before_erasure(tmp_path, monkeypatch):
    store = ConversationTranscripts(tmp_path, "owned")
    store.append_messages("child", [HumanMessage("secret")])
    entered = threading.Event()
    release = threading.Event()
    materialize = TranscriptStore.materialize

    def delayed(self, thread_id: str, *, agent_id: str | None = None):
        entered.set()
        assert release.wait(5)
        return materialize(self, thread_id, agent_id=agent_id)

    monkeypatch.setattr(TranscriptStore, "materialize", delayed)
    writing = asyncio.create_task(asyncio.to_thread(store.materialize, "child"))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        deleting = asyncio.create_task(asyncio.to_thread(erase_hook_transcripts, tmp_path, "owned"))
        await asyncio.sleep(0)
        release.set()
        await writing
        await deleting
        assert not store.thread_path("child").exists()
        with pytest.raises(OSError, match="deleted"):
            store.materialize("child")
    finally:
        release.set()
        await writing


def test_cleanup_does_not_follow_symlinks(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "keep"
    secret.write_text("unrelated")
    store = ConversationTranscripts(tmp_path, "owned")
    (store.root / "link").symlink_to(outside, target_is_directory=True)
    erase_hook_transcripts(tmp_path, "owned")
    assert secret.read_text() == "unrelated"
    other = ConversationTranscripts(tmp_path, "other")
    other.root.rmdir()
    other.root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError, match="Unsafe"):
        erase_hook_transcripts(tmp_path, "other")
    assert secret.read_text() == "unrelated"


def test_agent_projections_and_backup_only_artifacts_are_erased(tmp_path):
    store = ConversationTranscripts(tmp_path, "../../conversation")
    store.append_messages("child", [HumanMessage("first", id="one")], agent_id="worker")
    path = store.materialize("child", agent_id="worker").path
    store.append_messages("child", [HumanMessage("second", id="two")], agent_id="worker")
    store.materialize("child", agent_id="worker")
    backups = list(path.parent.glob(f"{path.name}.bak-*"))
    assert backups
    path.unlink()
    erase_hook_transcripts(tmp_path, "../../conversation")
    assert not path.exists()
    assert all(not backup.exists() for backup in backups)
