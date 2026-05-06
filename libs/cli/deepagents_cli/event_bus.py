"""Alpha external event ingress for the Textual CLI app."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from deepagents_cli.command_registry import BypassTier

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

ExternalEventKind = Literal["command", "prompt", "signal"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ExternalEvent:
    """A transport-independent event delivered from outside the TUI."""

    kind: ExternalEventKind
    payload: str
    source: str
    bypass: BypassTier = BypassTier.QUEUED
    correlation_id: str | None = None


class EventSource(Protocol):
    """Source of external events for the CLI app."""

    async def start(
        self,
        sink: Callable[[ExternalEvent], Awaitable[None]],
    ) -> None:
        """Start forwarding events to `sink`.

        Args:
            sink: Async callback that receives parsed external events.
        """

    async def stop(self) -> None:
        """Stop forwarding events and release transport resources."""


class UnixSocketEventSource:
    """Line-delimited JSON event source over a local Unix domain socket."""

    def __init__(self, path: Path | None = None) -> None:
        """Create a Unix-socket event source.

        Args:
            path: Socket path. When omitted, a per-process path under the
                runtime or temp directory is used.
        """
        self.path = path or default_unix_socket_path()
        self._server: asyncio.AbstractServer | None = None
        self._sink: Callable[[ExternalEvent], Awaitable[None]] | None = None

    async def start(
        self,
        sink: Callable[[ExternalEvent], Awaitable[None]],
    ) -> None:
        """Start listening for newline-delimited JSON events."""
        self._sink = sink
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with contextlib.suppress(FileNotFoundError):
            _unlink_existing_socket(self.path)
        self._server = await asyncio.start_unix_server(
            self._handle_client,
            path=str(self.path),
        )
        self.path.chmod(0o600)

    async def stop(self) -> None:
        """Close the listener and remove the socket path."""
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            await server.wait_closed()
        with contextlib.suppress(FileNotFoundError, FileExistsError):
            _unlink_existing_socket(self.path)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while line := await reader.readline():
                event = decode_external_event(line, source=f"unix:{self.path}")
                if self._sink is not None:
                    await self._sink(event)
                writer.write(b'{"ok":true}\n')
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()


def default_unix_socket_path() -> Path:
    """Return the default per-process Unix socket path."""
    root = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(root) if root else Path(tempfile.gettempdir())
    return base / "deepagents" / f"events-{os.getpid()}.sock"


def _unlink_existing_socket(path: Path) -> None:
    """Remove a stale Unix socket without touching other filesystem entries.

    Args:
        path: Candidate socket path to remove.

    Raises:
        FileExistsError: If `path` exists but is not a Unix socket.
    """
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISSOCK(info.st_mode):
        msg = f"Refusing to remove non-socket external event path: {path}"
        raise FileExistsError(msg)
    path.unlink()


def decode_external_event(data: bytes, *, source: str) -> ExternalEvent:
    """Decode one newline-delimited JSON external event.

    Args:
        data: Raw JSON line.
        source: Transport-specific source label attached to the event.

    Returns:
        Parsed external event.

    Raises:
        TypeError: If the envelope is not a JSON object.
        ValueError: If the event envelope is invalid.
    """
    try:
        raw = json.loads(data)
    except json.JSONDecodeError as exc:
        msg = "External event must be valid JSON"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = "External event must be a JSON object"
        raise TypeError(msg)

    kind = raw.get("kind")
    if kind not in {"command", "prompt", "signal"}:
        msg = "External event kind must be command, prompt, or signal"
        raise ValueError(msg)

    payload = raw.get("payload")
    if not isinstance(payload, str) or not payload.strip():
        msg = "External event payload must be a non-empty string"
        raise ValueError(msg)

    bypass = raw.get("bypass", BypassTier.QUEUED.value)
    try:
        bypass_tier = BypassTier(bypass)
    except ValueError as exc:
        msg = "External event bypass must be a valid bypass tier"
        raise ValueError(msg) from exc

    correlation_id = raw.get("correlation_id")
    if correlation_id is not None and not isinstance(correlation_id, str):
        msg = "External event correlation_id must be a string when present"
        raise ValueError(msg)

    return ExternalEvent(
        kind=kind,
        payload=payload,
        source=source,
        bypass=bypass_tier,
        correlation_id=correlation_id,
    )
