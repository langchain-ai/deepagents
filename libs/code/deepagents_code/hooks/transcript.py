"""Client-owned conversation transcript projections for Hooks v2.

Materializes versioned per-thread and per-subagent JSONL files that hook
commands can read via `transcript_path` / `agent_transcript_path`.

Lag semantics:
    The on-disk JSONL may lag behind live checkpoint/UI state. Callers that need
    the just-finished assistant turn must prefer `last_assistant_message` on
    Stop/SubagentStop. `materialize()` flushes pending records immediately before
    returning a path so hooks see a consistent snapshot of what the store has
    accepted so far, not a live tail of the server checkpoint.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
import threading
import unicodedata
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

if os.name != "nt":
    import fcntl
else:
    import msvcrt

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_chunk_to_message,
)
from pydantic import BaseModel, ConfigDict

from deepagents_code.config_manifest import _is_secret_env
from deepagents_code.json_types import JSON_VALUE_ADAPTER, JsonValue

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence
    from typing import Protocol

    class _TranscriptRuntime(Protocol):
        def append_messages(
            self,
            thread_id: str,
            messages: Sequence[BaseMessage],
            *,
            agent_id: str | None = None,
        ) -> None: ...


logger = logging.getLogger(__name__)

SUBAGENT_TRANSCRIPT_ID_METADATA_KEY = "dcode_subagent_id"
_INTERNAL_STREAM_SOURCES = frozenset({"summarization", "auto_mode_classifier"})

TRANSCRIPT_SCHEMA_VERSION = 1
DEFAULT_RETENTION_REVISIONS = 20
_FILE_MODE = 0o600
_DIR_MODE = 0o700
# Credential-style assignments. Bare names like PASSWORD= are matched via the
# trailing keyword alternatives when preceded by an underscore-separated prefix
# (for example OPENAI_API_KEY=), matching the repository secret-name policy.
_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b([A-Z][A-Z0-9_]*(?:API[_-]?KEY|KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)"
    r"[A-Z0-9_]*)\s*=\s*([^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\b(Bearer)\s+[A-Za-z0-9._~+/=-]{8,}")
_PREFIXED_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:"
    r"sk-(?:ant-)?|sk_(?:live|test)_|pk_(?:live|test)_|gh[pousr]_|"
    r"github_pat_|glpat-|xox[baprs]-|hf_|npm_|AIza|AKIA"
    r")[A-Za-z0-9._-]{8,}"
)
_JWT_RE = re.compile(
    r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{6,}\."
    r"[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}"
)
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_SAFE_PREFIX_RE = re.compile(r"[^a-z0-9]+")
_SAFE_PREFIX_LENGTH = 32
_EMPTY_REVISION = hashlib.sha256(b"").hexdigest()


class TranscriptRecord(BaseModel):
    """One JSONL record in a materialized transcript projection."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = TRANSCRIPT_SCHEMA_VERSION
    """Transcript schema version used to interpret this record."""

    sequence: int
    """Zero-based position of this record within its transcript."""

    record_id: str
    """Message identifier, or a deterministic role-and-sequence fallback."""

    timestamp: str | None = None
    """Source message timestamp when one is available."""

    thread_id: str
    """Conversation thread that owns this record."""

    agent_id: str | None = None
    """Subagent scope for an agent transcript, otherwise `None`."""

    role: Literal["user", "assistant", "tool", "system"]
    """Normalized conversation role for the projected message."""

    message_id: str | None = None
    """Original LangChain message identifier when one is available."""

    content: JsonValue
    """Redacted JSON-compatible message content."""

    name: str | None = None
    """Tool or message name when one is available."""


@dataclass(frozen=True, slots=True)
class TranscriptHandle:
    """Identity of a materialized transcript file."""

    path: Path
    revision: str
    thread_id: str
    agent_id: str | None = None


@dataclass
class _TranscriptBuffer:
    records: list[TranscriptRecord] = field(default_factory=list)
    record_ids: set[str] = field(default_factory=set)
    dirty: bool = False
    revision: str = _EMPTY_REVISION


class TranscriptStore:
    """Append-only JSONL transcript projections owned by the client process."""

    def __init__(
        self,
        root: Path,
        *,
        retention_revisions: int = DEFAULT_RETENTION_REVISIONS,
    ) -> None:
        """Create a store rooted at `root`.

        Args:
            root: Directory that will contain per-thread transcript files.
            retention_revisions: Maximum prior `.bak-*` revisions retained per
                transcript after each rewrite.

        Raises:
            ValueError: If `retention_revisions` is negative.
        """
        if retention_revisions < 0:
            msg = "retention_revisions must be nonnegative"
            raise ValueError(msg)
        self.root = root.expanduser().resolve()
        self.retention_revisions = retention_revisions
        self._buffers: dict[tuple[str, str | None], _TranscriptBuffer] = {}
        self._lock = threading.RLock()
        _ensure_private_directories(self.root, self.root)

    def thread_path(self, thread_id: str) -> Path:
        """Return the materialized path for a thread transcript.

        Args:
            thread_id: Conversation thread identifier.

        Returns:
            Absolute JSONL path for the thread.
        """
        return self.root / f"{_safe_component(thread_id)}.jsonl"

    def agent_path(self, thread_id: str, agent_id: str) -> Path:
        """Return the materialized path for a subagent transcript.

        Args:
            thread_id: Parent conversation thread identifier.
            agent_id: Subagent identifier.

        Returns:
            Absolute JSONL path nested under the thread.
        """
        return (
            self.root
            / _safe_component(thread_id)
            / "agents"
            / f"{_safe_component(agent_id)}.jsonl"
        )

    def append_messages(
        self,
        thread_id: str,
        messages: Sequence[BaseMessage],
        *,
        agent_id: str | None = None,
    ) -> None:
        """Append redacted message projections to the in-memory buffer.

        Args:
            thread_id: Conversation thread identifier.
            messages: LangChain messages to project.
            agent_id: Optional subagent scope.
        """
        with self._lock:
            buffer = self._buffer(thread_id, agent_id)
            for message in messages:
                record = _record_from_message(
                    message,
                    thread_id=thread_id,
                    agent_id=agent_id,
                    sequence=len(buffer.records),
                )
                if record is None:
                    continue
                if (
                    record.message_id is not None
                    and record.record_id in buffer.record_ids
                ):
                    continue
                buffer.records.append(record)
                buffer.record_ids.add(record.record_id)
                buffer.dirty = True

    def materialize(
        self,
        thread_id: str,
        *,
        agent_id: str | None = None,
    ) -> TranscriptHandle:
        """Flush pending records and return the client-readable path.

        Materialization rewrites the whole per-thread JSONL file, so it runs
        under a cross-process advisory lock and re-reads the on-disk records
        first, merging in anything another dcode process appended since this
        process last loaded the file.

        Args:
            thread_id: Conversation thread identifier.
            agent_id: Optional subagent scope.

        Returns:
            Handle with path and content revision identity.
        """
        with self._lock:
            buffer = self._buffer(thread_id, agent_id)
            path = (
                self.agent_path(thread_id, agent_id)
                if agent_id is not None
                else self.thread_path(thread_id)
            )
            _ensure_private_directories(self.root, path.parent)
            if path.is_file() and os.name != "nt":
                path.chmod(_FILE_MODE)
            with _file_lock(path.with_suffix(path.suffix + ".lock")):
                _merge_disk_records(path, buffer)
                if buffer.dirty or not path.is_file():
                    revision = _write_transcript(
                        self.root,
                        path,
                        buffer.records,
                        self.retention_revisions,
                    )
                    buffer.revision = revision
                    buffer.dirty = False
            return TranscriptHandle(
                path=path,
                revision=buffer.revision,
                thread_id=thread_id,
                agent_id=agent_id,
            )

    def revision(self, thread_id: str, *, agent_id: str | None = None) -> str:
        """Return the current revision id without forcing a flush.

        Args:
            thread_id: Conversation thread identifier.
            agent_id: Optional subagent scope.

        Returns:
            Content revision string for the buffered projection.
        """
        with self._lock:
            buffer = self._buffer(thread_id, agent_id)
            if buffer.dirty:
                return _revision_for_records(buffer.records)
            return buffer.revision

    def _buffer(self, thread_id: str, agent_id: str | None) -> _TranscriptBuffer:
        key = (thread_id, agent_id)
        buffer = self._buffers.get(key)
        if buffer is None:
            buffer = _TranscriptBuffer()
            path = (
                self.agent_path(thread_id, agent_id)
                if agent_id is not None
                else self.thread_path(thread_id)
            )
            if path.is_file():
                buffer.records, valid = _read_transcript(path)
                buffer.record_ids = {record.record_id for record in buffer.records}
                buffer.revision = _revision_for_records(buffer.records)
                buffer.dirty = not valid
            self._buffers[key] = buffer
        return buffer


@dataclass(slots=True)
class TranscriptRecorder:
    """Collect completed stream messages into a Hooks transcript runtime."""

    runtime: _TranscriptRuntime
    thread_id: str
    _chunks: dict[tuple[str | None, str], BaseMessageChunk] = field(
        default_factory=dict
    )

    def record(
        self,
        message: object,
        metadata: Mapping[str, object] | None,
        *,
        main_agent: bool,
    ) -> None:
        """Record one streamed message when its transcript identity is stable.

        Args:
            message: Streamed LangChain message or chunk.
            metadata: Stream metadata carrying optional subagent identity.
            main_agent: Whether the message belongs to the root graph.
        """
        if (
            metadata is not None
            and metadata.get("lc_source") in _INTERNAL_STREAM_SOURCES
        ):
            return
        agent_id = None if main_agent else _stream_agent_id(metadata)
        if not main_agent and agent_id is None:
            return
        if isinstance(message, BaseMessageChunk):
            key = (agent_id, message.id or type(message).__name__)
            previous = self._chunks.get(key)
            combined = message if previous is None else previous + message
            self._chunks[key] = combined
            if getattr(message, "chunk_position", None) != "last":
                return
            self._chunks.pop(key, None)
            self._append(message_chunk_to_message(combined), agent_id=agent_id)
            return
        if isinstance(message, BaseMessage):
            self._append(message, agent_id=agent_id)

    def append(self, messages: Sequence[BaseMessage]) -> None:
        """Append checkpoint or input messages to the root transcript."""
        append_messages = getattr(self.runtime, "append_messages", None)
        if callable(append_messages):
            append_messages(self.thread_id, messages)

    def _append(self, message: BaseMessage, *, agent_id: str | None) -> None:
        append_messages = getattr(self.runtime, "append_messages", None)
        if not callable(append_messages):
            return
        try:
            append_messages(self.thread_id, [message], agent_id=agent_id)
        except (TypeError, ValueError):
            logger.warning(
                "Skipping invalid streamed transcript message",
                exc_info=True,
            )


def _stream_agent_id(metadata: Mapping[str, object] | None) -> str | None:
    if metadata is None:
        return None
    value = metadata.get(SUBAGENT_TRANSCRIPT_ID_METADATA_KEY)
    return value if isinstance(value, str) and value else None


def _record_from_message(
    message: BaseMessage,
    *,
    thread_id: str,
    agent_id: str | None,
    sequence: int,
) -> TranscriptRecord | None:
    if isinstance(message, HumanMessage):
        role: Literal["user", "assistant", "tool", "system"] = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, ToolMessage):
        role = "tool"
    elif isinstance(message, SystemMessage):
        role = "system"
    else:
        return None

    raw = message.model_dump(mode="json")
    content = JSON_VALUE_ADAPTER.validate_python(
        redact_transcript_value(raw.get("content"))
    )
    message_id = message.id if isinstance(message.id, str) else None
    name = getattr(message, "name", None)
    tool_name = name if isinstance(name, str) else None
    record_id = message_id or f"{role}:{sequence}"
    return TranscriptRecord(
        sequence=sequence,
        record_id=record_id,
        thread_id=thread_id,
        agent_id=agent_id,
        role=role,
        message_id=message_id,
        content=content,
        name=tool_name,
    )


def redact_transcript_value(value: object) -> JsonValue:
    """Redact secret-like strings inside transcript content.

    Args:
        value: Arbitrary message content.

    Returns:
        JSON-compatible content with URLs/credentials scrubbed.
    """
    if isinstance(value, str):
        return _redact_text(value)
    if isinstance(value, list):
        return [redact_transcript_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): (
                "[redacted]"
                if _is_secret_env(str(key))
                else redact_transcript_value(item)
            )
            for key, item in value.items()
        }
    return JSON_VALUE_ADAPTER.validate_python(value)


def _redact_text(text: str) -> str:
    redacted = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}=[redacted]",
        text,
    )
    redacted = _BEARER_RE.sub(lambda match: f"{match.group(1)} [redacted]", redacted)
    redacted = _PREFIXED_TOKEN_RE.sub("[redacted]", redacted)
    redacted = _JWT_RE.sub("[redacted]", redacted)
    return _URL_RE.sub(lambda match: _redact_url(match.group(0)), redacted)


def _redact_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname or ""
        port = parsed.port
    except ValueError:
        return "[redacted URL]"
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    netloc = f"{hostname}:{port}" if port is not None else hostname
    path = "/[redacted]" if parsed.path else ""
    query_items = parse_qsl(parsed.query, keep_blank_values=True)
    query = urlencode([(key, "[redacted]") for key, _value in query_items])
    fragment = "[redacted]" if parsed.fragment else ""
    return urlunsplit((parsed.scheme, netloc, path, query, fragment))


@contextmanager
def _file_lock(lock_path: Path) -> Iterator[None]:
    """Hold an advisory cross-process lock on `lock_path`.

    Materialization rewrites the whole transcript file, so concurrent dcode
    processes resuming the same thread must serialize their read-merge-write
    cycle on something stronger than the in-process `threading.RLock`.
    """
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, _FILE_MODE)
    try:
        if os.name != "nt":
            fcntl.flock(fd, fcntl.LOCK_EX)
        else:
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        try:
            yield
        finally:
            if os.name != "nt":
                fcntl.flock(fd, fcntl.LOCK_UN)
            else:
                with suppress(OSError):
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    finally:
        os.close(fd)


def _merge_disk_records(path: Path, buffer: _TranscriptBuffer) -> None:
    """Fold on-disk records missing from `buffer` back into it.

    Another process may have materialized the shared transcript since this
    process last loaded it; re-reading under the file lock prevents the next
    rewrite from silently dropping that process's records. Buffer records win
    ordering ties; disk-only records keep their relative order appended after.
    """
    if not path.is_file():
        return
    disk_records, valid = _read_transcript(path)
    if not valid:
        return
    buffer_ids = {record.record_id for record in buffer.records}
    if all(record.record_id in buffer_ids for record in disk_records):
        return
    merged = list(buffer.records)
    merged_ids = set(buffer_ids)
    for record in disk_records:
        if record.record_id in merged_ids:
            continue
        merged.append(record.model_copy(update={"sequence": len(merged)}))
        merged_ids.add(record.record_id)
    buffer.records = merged
    buffer.record_ids = merged_ids
    buffer.dirty = True


def _write_transcript(
    root: Path,
    path: Path,
    records: Sequence[TranscriptRecord],
    retention_revisions: int,
) -> str:
    _ensure_private_directories(root, path.parent)
    payload = "".join(
        record.model_dump_json(exclude_none=True) + "\n" for record in records
    )
    revision = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    fd, raw_tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    tmp_path = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            tmp_path.chmod(_FILE_MODE)
        # Copy the previous revision aside first, then atomically replace the
        # live path so concurrent readers never observe a missing file.
        if path.exists():
            prior_payload = path.read_bytes()
            prior_revision = hashlib.sha256(prior_payload).hexdigest()
            backup = path.with_suffix(path.suffix + f".bak-{prior_revision}")
            _write_backup(backup, prior_payload)
            _prune_backups(path, retention_revisions)
        tmp_path.replace(path)
        if os.name != "nt":
            path.chmod(_FILE_MODE)
    except OSError:
        logger.warning("Failed to materialize transcript at %s", path, exc_info=True)
        with suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise
    return revision


def _write_backup(path: Path, payload: bytes) -> None:
    fd, raw_tmp = tempfile.mkstemp(dir=path.parent, suffix=".bak.tmp")
    tmp_path = Path(raw_tmp)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.name != "nt":
            tmp_path.chmod(_FILE_MODE)
        tmp_path.replace(path)
    except OSError:
        with suppress(OSError):
            tmp_path.unlink(missing_ok=True)
        raise


def _safe_component(identifier: str) -> str:
    normalized = unicodedata.normalize("NFKD", identifier)
    readable = normalized.encode("ascii", errors="ignore").decode("ascii").lower()
    prefix = _SAFE_PREFIX_RE.sub("-", readable).strip("-")[:_SAFE_PREFIX_LENGTH]
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()
    return f"{prefix or 'id'}--{digest}"


def _ensure_private_directories(root: Path, target: Path) -> None:
    root.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    target.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
    if os.name == "nt":
        return
    root.chmod(_DIR_MODE)
    relative = target.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        current.chmod(_DIR_MODE)


def _prune_backups(path: Path, retention_revisions: int) -> None:
    pattern = f"{path.name}.bak-*"
    backups = sorted(
        path.parent.glob(pattern),
        key=lambda item: (item.stat().st_mtime_ns, item.name),
    )
    excess = len(backups) - retention_revisions
    for stale in backups[: max(0, excess)]:
        with suppress(OSError):
            stale.unlink()


def _read_transcript(path: Path) -> tuple[list[TranscriptRecord], bool]:
    records: list[TranscriptRecord] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(TranscriptRecord.model_validate_json(line))
    except (OSError, ValueError):
        logger.warning("Could not read transcript at %s", path, exc_info=True)
        return [], False
    return records, True


def _revision_for_records(records: Sequence[TranscriptRecord]) -> str:
    payload = "".join(
        record.model_dump_json(exclude_none=True) + "\n" for record in records
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
