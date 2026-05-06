"""External event ingress for the Textual CLI app.

Exposes a small `EventSource` protocol plus a Unix-domain-socket implementation
that lets local processes push commands, prompts, and signals into a running
CLI session over a newline-delimited JSON wire protocol.

!!! warning "Experimental"

    The wire format and configuration env vars may change without semver
    guarantees while this surface stabilizes.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, get_args

from deepagents_cli.command_registry import BypassTier

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

ExternalEventKind = Literal["command", "prompt", "signal"]
"""Top-level event kinds carried by the wire protocol."""

ExternalSignal = Literal["interrupt", "force-clear"]
"""Closed vocabulary of `kind="signal"` payloads accepted by the listener."""

EventSink = "Callable[[ExternalEvent], Awaitable[None]]"
"""Type alias for the async callback that receives parsed events."""

_VALID_KINDS: frozenset[str] = frozenset(get_args(ExternalEventKind))
_VALID_SIGNALS: frozenset[str] = frozenset(get_args(ExternalSignal))

_ACK = b'{"ok":true}\n'
"""Wire-level acknowledgement returned for an accepted event."""

_MAX_LINE_BYTES = 64 * 1024
"""Per-line read limit; an oversized line is rejected with a NACK."""

_CLIENT_IDLE_TIMEOUT_SECONDS = 60.0
"""Maximum idle time on a client connection before the server closes it."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ExternalEvent:
    """A transport-independent event delivered from outside the TUI."""

    kind: ExternalEventKind
    payload: str
    source: str
    bypass: BypassTier = BypassTier.QUEUED
    correlation_id: str | None = None

    def __post_init__(self) -> None:
        """Validate invariants for direct construction.

        Raises:
            ValueError: If `kind` is not a known kind, the payload is empty
                or whitespace-only, or the kind is `"signal"` but the payload
                is not a recognized signal name.
        """
        if self.kind not in _VALID_KINDS:
            msg = f"Unknown external event kind: {self.kind!r}"
            raise ValueError(msg)
        if not self.payload or not self.payload.strip():
            msg = "External event payload must be a non-empty string"
            raise ValueError(msg)
        if self.kind == "signal" and self.payload.strip().lower() not in _VALID_SIGNALS:
            msg = (
                f"Unknown external signal: {self.payload!r}; "
                f"expected one of {sorted(_VALID_SIGNALS)}"
            )
            raise ValueError(msg)


class EventSource(Protocol):
    """Source of external events for the CLI app.

    Implementations must be safe to `stop()` even when `start()` failed
    partway through; the app always invokes `stop()` from a `finally` block.
    """

    async def start(
        self,
        sink: Callable[[ExternalEvent], Awaitable[None]],
    ) -> None:
        """Start forwarding events to `sink`.

        Args:
            sink: Async callback that receives parsed external events.
        """

    async def serve_forever(self) -> None:
        """Park until the source is cancelled or its transport dies.

        Implementations should re-raise `CancelledError` and propagate
        unexpected transport-layer failures so the lifecycle owner can
        notice and surface them.
        """

    async def stop(self) -> None:
        """Stop forwarding events and release transport resources."""


class UnixSocketEventSource:
    """Line-delimited JSON event source over a local Unix domain socket.

    The listener creates its parent directory with mode `0o700` and binds the
    socket inside it under a transient `umask(0o077)` so the socket inherits
    `0o600` from the moment of `bind()`. Stale sockets at the configured path
    are removed on start, but only after a `stat` confirms the path is a
    socket — a regular file or directory at that path is left untouched and
    causes start to fail loudly.
    """

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
        """Start listening for newline-delimited JSON events.

        Args:
            sink: Async callback invoked with each decoded event.

        Raises:
            RuntimeError: If `start()` has already been called on this
                instance without a subsequent `stop()`.
            FileExistsError: If the socket path is occupied by a non-socket
                filesystem entry.
            OSError: If the socket cannot be bound (e.g. permission denied,
                path too long).
        """  # noqa: DOC502  # FileExistsError/OSError propagate from helpers
        if self._server is not None:
            msg = "UnixSocketEventSource is already started"
            raise RuntimeError(msg)

        self._sink = sink
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with contextlib.suppress(FileNotFoundError):
            _unlink_existing_socket(self.path)

        previous_umask = os.umask(0o077)
        try:
            self._server = await asyncio.start_unix_server(
                self._handle_client,
                path=str(self.path),
                limit=_MAX_LINE_BYTES,
            )
        finally:
            os.umask(previous_umask)

        # Defense in depth: even if `start_unix_server` somehow ignored the
        # umask (different libc, mocked socket layer), force-tighten the mode.
        with contextlib.suppress(OSError):
            self.path.chmod(0o600)
        logger.debug("External event listener bound at %s", self.path)

    async def serve_forever(self) -> None:
        """Park until the underlying server is cancelled or fails.

        `asyncio.start_unix_server` already begins accepting connections,
        so this delegates to the server's own `serve_forever`. A fatal
        error inside the accept loop propagates here, letting the
        lifecycle owner notice and react.

        Raises:
            RuntimeError: If invoked before `start()`.
        """
        if self._server is None:
            msg = "UnixSocketEventSource.serve_forever called before start()"
            raise RuntimeError(msg)
        await self._server.serve_forever()

    async def stop(self) -> None:
        """Close the listener and remove the socket path.

        Idempotent: safe to call after a failed or never-started `start()`.
        """
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            with contextlib.suppress(Exception):
                await server.wait_closed()
        try:
            _unlink_existing_socket(self.path)
        except FileNotFoundError:
            pass
        except FileExistsError as exc:
            logger.warning(
                "Leaving non-socket entry at %s during shutdown: %s",
                self.path,
                exc,
            )
        except OSError as exc:
            logger.warning("Failed to unlink event socket %s: %s", self.path, exc)

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        r"""Read newline-delimited JSON envelopes from one client.

        Each accepted line yields an event to the configured sink and is
        acked with `{"ok":true}\n` (plus `correlation_id` when present).
        Rejected lines (oversized, malformed, sink unconfigured, sink
        raised) are answered with `{"ok":false,"error":...}\n` and the
        loop continues so a single bad caller cannot kill the connection.
        """
        peer = writer.get_extra_info("peername") or "<unknown>"
        logger.debug("External event client connected: %s", peer)
        try:
            while True:
                try:
                    line = await asyncio.wait_for(
                        reader.readline(),
                        timeout=_CLIENT_IDLE_TIMEOUT_SECONDS,
                    )
                except TimeoutError:
                    logger.debug("Closing idle external event client %s", peer)
                    break
                except (ValueError, asyncio.LimitOverrunError) as exc:
                    logger.warning("External event line exceeded read limit: %s", exc)
                    await _write_nack(writer, "line exceeds read limit", None)
                    break

                if not line:
                    break

                await self._handle_one_line(line, writer)
        except (BrokenPipeError, ConnectionResetError) as exc:
            logger.debug("External event client %s disconnected: %s", peer, exc)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
            logger.debug("External event client closed: %s", peer)

    async def _handle_one_line(
        self,
        line: bytes,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Decode and dispatch one envelope, replying with ACK or NACK."""
        correlation_id: str | None = None
        try:
            event = decode_external_event(line, source=f"unix:{self.path}")
        except (ValueError, TypeError) as exc:
            logger.warning("Rejected malformed external event: %s", exc)
            with contextlib.suppress(ValueError, TypeError, json.JSONDecodeError):
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    candidate = parsed.get("correlation_id")
                    if isinstance(candidate, str):
                        correlation_id = candidate
            await _write_nack(writer, str(exc), correlation_id)
            return

        correlation_id = event.correlation_id

        if self._sink is None:
            logger.warning("External event arrived before sink was set; dropping")
            await _write_nack(writer, "listener not ready", correlation_id)
            return

        try:
            await self._sink(event)
        except Exception as exc:
            logger.exception("External event sink raised")
            await _write_nack(writer, f"sink failed: {exc}", correlation_id)
            return

        await _write_ack(writer, correlation_id)


async def _write_ack(
    writer: asyncio.StreamWriter,
    correlation_id: str | None,
) -> None:
    """Write the success acknowledgement frame, echoing `correlation_id`."""
    if correlation_id is None:
        writer.write(_ACK)
    else:
        body = json.dumps({"ok": True, "correlation_id": correlation_id})
        writer.write(body.encode("utf-8") + b"\n")
    with contextlib.suppress(BrokenPipeError, ConnectionResetError):
        await writer.drain()


async def _write_nack(
    writer: asyncio.StreamWriter,
    error: str,
    correlation_id: str | None,
) -> None:
    """Write a failure response frame; never raises on a closed socket."""
    body: dict[str, object] = {"ok": False, "error": error}
    if correlation_id is not None:
        body["correlation_id"] = correlation_id
    try:
        writer.write(json.dumps(body).encode("utf-8") + b"\n")
        await writer.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass


def default_unix_socket_path() -> Path:
    """Return the default per-process Unix socket path.

    Prefers `XDG_RUNTIME_DIR` (per-user, tmpfs-backed, auto-cleaned on
    logout) and falls back to the system temp dir when the runtime
    directory is unset.
    """
    root = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(root) if root else Path(tempfile.gettempdir())
    return base / "deepagents" / f"events-{os.getpid()}.sock"


def _unlink_existing_socket(path: Path) -> None:
    """Remove a stale Unix socket without touching other filesystem entries.

    Args:
        path: Candidate socket path to remove.

    Raises:
        FileNotFoundError: If `path` does not exist.
        FileExistsError: If `path` exists but is not a Unix socket.
        OSError: If the entry exists but cannot be removed.
    """  # noqa: DOC502  # FileNotFoundError/OSError propagate from stat/unlink
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
        ValueError: If any envelope field is missing, of the wrong type, or
            otherwise invalid.
    """
    try:
        raw = json.loads(data)
    except json.JSONDecodeError as exc:
        msg = f"External event must be valid JSON: {exc.msg}"
        raise ValueError(msg) from exc
    if not isinstance(raw, dict):
        msg = "External event must be a JSON object"
        raise TypeError(msg)

    kind = raw.get("kind")
    if kind not in _VALID_KINDS:
        msg = f"External event kind must be one of {sorted(_VALID_KINDS)}; got {kind!r}"
        raise ValueError(msg)

    payload = raw.get("payload")
    if not isinstance(payload, str) or not payload.strip():
        msg = "External event payload must be a non-empty string"
        raise ValueError(msg)

    bypass = raw.get("bypass", BypassTier.QUEUED.value)
    try:
        bypass_tier = BypassTier(bypass)
    except ValueError as exc:
        msg = f"External event bypass must be a valid bypass tier: {exc}"
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
