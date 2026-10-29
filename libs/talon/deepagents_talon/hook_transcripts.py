"""Conversation-owned hook projections and retryable local erasure."""

from __future__ import annotations

import hashlib
import shutil
import threading
from typing import TYPE_CHECKING

from deepagents_code.hooks.transcript import TranscriptStore

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from langchain_core.messages import BaseMessage

    from deepagents_code.hooks.transcript import TranscriptHandle

# Hook loading runs in a worker thread; recording and erasure run on the loop too.
TRANSCRIPT_LOCK = threading.RLock()


def _directory(path: Path) -> None:
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        msg = f"Unsafe hook transcript directory: {path}"
        raise OSError(msg)
    path.mkdir(mode=0o700, parents=True, exist_ok=True)


def _owner_root(home: Path, session: str) -> Path:
    root = home / "transcripts"
    _directory(root)
    owners = root / "conversations"
    _directory(owners)
    owner = owners / hashlib.sha256(session.encode()).hexdigest()
    _directory(owner)
    return owner


def _remove(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        path.unlink(missing_ok=True)
    else:
        shutil.rmtree(path)


def erase_hook_transcripts(home: Path, session: str) -> None:
    """Revoke writers before erasing projections; retain the marker for retries."""
    with TRANSCRIPT_LOCK:
        owner = _owner_root(home, session)
        marker = owner / ".deleted"
        if not marker.exists() and not marker.is_symlink():
            marker.touch(mode=0o600, exist_ok=False)
        for path in owner.iterdir():
            if path.name != ".deleted":
                _remove(path)
        # Parent projections written before conversation ownership was introduced.
        legacy = TranscriptStore(home / "transcripts").thread_path(session)
        _remove(legacy)
        for backup in legacy.parent.glob(f"{legacy.name}.bak-*"):
            _remove(backup)
        _remove(legacy.with_suffix(""))


class ConversationTranscripts(TranscriptStore):
    """Keep isolated child IDs inside their durable parent's erasure boundary."""

    def __init__(self, home: Path, session: str) -> None:
        """Open a parent-owned store, refusing previously erased sessions."""
        with TRANSCRIPT_LOCK:
            super().__init__(_owner_root(home, session))
            self._check(self.root)

    def _check(self, path: Path) -> None:
        marker = self.root / ".deleted"
        if marker.exists() or marker.is_symlink():
            msg = "Hook transcript conversation has been deleted"
            raise OSError(msg)
        for candidate in (self.root.parent.parent, self.root.parent, self.root):
            if candidate.is_symlink():
                msg = "Unsafe hook transcript directory"
                raise OSError(msg)
        candidate = self.root
        for part in path.relative_to(self.root).parts:
            candidate /= part
            if candidate.is_symlink():
                msg = "Unsafe hook transcript path"
                raise OSError(msg)

    def append_messages(
        self, thread_id: str, messages: Sequence[BaseMessage], *, agent_id: str | None = None
    ) -> None:
        """Reject buffered writes from revoked parent or child adapters."""
        with TRANSCRIPT_LOCK:
            path = self.agent_path(thread_id, agent_id) if agent_id else self.thread_path(thread_id)
            self._check(path)
            super().append_messages(thread_id, messages, agent_id=agent_id)

    def materialize(self, thread_id: str, *, agent_id: str | None = None) -> TranscriptHandle:
        """Serialize projection writes with erasure, including background workers."""
        with TRANSCRIPT_LOCK:
            path = self.agent_path(thread_id, agent_id) if agent_id else self.thread_path(thread_id)
            self._check(path)
            return super().materialize(thread_id, agent_id=agent_id)
