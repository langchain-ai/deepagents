"""Message local deepagents-code (`dcode`) sessions over their event sockets.

!!! warning "Experimental"

    dcode's external event socket is itself experimental, opt-in on the dcode side
    (`DEEPAGENTS_CODE_EXTERNAL_EVENT_SOCKET=1`), and off by default here. Its wire
    format may change without notice.

The socket only acknowledges that dcode queued a message; it carries no reply. Talon
therefore reports delivery, not an answer, and the operator reads any response in the
dcode session itself.

Talon reaches only sockets that the same operating-system user owns, in private
directories, and checks the peer's user ID after connecting. As with the Claude Code
bridge, the operating-system user is the trust boundary. Each message is a normal
queued prompt wrapped in a `<talon-relayed-message>` envelope: Talon never sends slash
commands, signals, or queue-bypass requests, and dcode does not enforce the envelope,
so Talon's own approval prompt is the gate before a message reaches the session.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import stat
import tempfile
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from langchain_core.tools import BaseTool, tool

from deepagents_talon.background import _IN_SUBAGENT
from deepagents_talon.claude_sessions import peer_uid
from deepagents_talon.codex_sessions import relay
from deepagents_talon.tool_approvals import APPROVAL_OPERATOR

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

ENABLE_ENV = "DEEPAGENTS_TALON_DCODE_SESSIONS"
"""Set to `1` to give the operator's agent the dcode session tools."""

SEND_TOOL = "message_dcode_session"
"""Tool that always requires operator approval, whatever `tools.json` says."""

_MAX_MESSAGE_BYTES = 8_000
_MAX_ACK_BYTES = 64 * 1024
_IO_SECONDS = 10.0
_MAX_SENDS_PER_MINUTE = 20
_MAX_SENDS_PER_SESSION = 5
_MARKER = "Talon request ID: "
_SOCKET_NAME = re.compile(r"^events-([1-9][0-9]{0,9})\.sock$")
_PID = re.compile(r"^[1-9][0-9]{0,9}$")


def enabled(env: Mapping[str, str]) -> bool:
    """Return whether the operator enabled the experimental dcode bridge.

    Args:
        env: Process environment to read.
    """
    return env.get(ENABLE_ENV, "").lower() in {"1", "true", "yes"}


@dataclass(frozen=True, slots=True)
class DcodeSession:
    """One running dcode session with a verified event socket."""

    pid: int
    socket: Path

    def listing(self) -> dict[str, str]:
        """Return what the agent needs to pick a target, without the socket path."""
        return {"session": str(self.pid)}


def socket_dirs(env: Mapping[str, str]) -> list[Path]:
    """Return the directories where dcode binds its default event sockets.

    Mirrors dcode's `default_unix_socket_path`, which prefers `XDG_RUNTIME_DIR`
    and falls back to the system temp directory.

    Args:
        env: Process environment, for `XDG_RUNTIME_DIR`.
    """
    dirs = [Path(tempfile.gettempdir()) / "deepagents"]
    if runtime_dir := env.get("XDG_RUNTIME_DIR"):
        dirs.insert(0, Path(runtime_dir) / "deepagents")
    return dirs


def _owned(path: Path, kind: int) -> bool:
    """Return whether `path` itself, not a symlink target, is ours and of `kind`."""
    try:
        info = path.lstat()
    except OSError:
        return False
    return stat.S_IFMT(info.st_mode) == kind and info.st_uid == os.getuid()


def _private_dir(path: Path) -> bool:
    return _owned(path, stat.S_IFDIR) and path.lstat().st_mode & 0o077 == 0


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError, PermissionError, OverflowError:
        return False
    return True


def discover(dirs: list[Path]) -> list[DcodeSession]:
    """Find live dcode sessions whose sockets we own in private directories.

    Args:
        dirs: Candidate socket directories.
    """
    found: dict[int, DcodeSession] = {}
    for directory in dirs:
        if not _private_dir(directory):
            continue
        with suppress(OSError):
            for entry in sorted(directory.iterdir()):
                match = _SOCKET_NAME.fullmatch(entry.name)
                if match is None:
                    continue
                pid = int(match[1])
                if pid not in found and _owned(entry, stat.S_IFSOCK) and _alive(pid):
                    found[pid] = DcodeSession(pid, entry)
    return list(found.values())


def prompt_line(message: str, correlation_id: str) -> bytes:
    """Build the one queued-prompt event Talon ever sends.

    Args:
        message: Plain text approved by the operator.
        correlation_id: ID dcode echoes in its acknowledgement.
    """
    payload = relay(message, f"{_MARKER}{correlation_id}")
    event = {"kind": "prompt", "payload": payload, "correlation_id": correlation_id}
    return (json.dumps(event) + "\n").encode()


def _acknowledged(line: bytes, correlation_id: str) -> bool:
    try:
        ack = json.loads(line)
    except ValueError:
        return False
    return (
        isinstance(ack, dict)
        and ack.get("ok") is True
        and ack.get("correlation_id") == correlation_id
    )


def _denied() -> str | None:
    if _IN_SUBAGENT.get():
        return "dcode sessions are unavailable to subagents."
    if APPROVAL_OPERATOR.get() is not True:
        return "Only the paired operator can reach dcode sessions, in their own turn."
    return None


class DcodeSessions:
    """List and message the operator's running dcode sessions."""

    def __init__(
        self,
        env: Mapping[str, str],
        *,
        dirs: list[Path] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Prepare the opt-in bridge.

        Args:
            env: Process environment, for socket locations.
            dirs: Socket directories, injectable for tests.
            clock: Monotonic clock, injectable for tests.
        """
        self._dirs = dirs if dirs is not None else socket_dirs(env)
        self._clock = clock
        self._sends: deque[tuple[float, int]] = deque()
        self.tools = self._tools()

    def sessions(self) -> list[DcodeSession]:
        """List running dcode sessions with a verified event socket."""
        return discover(self._dirs)

    async def send(self, target: DcodeSession, message: str) -> dict[str, str]:
        """Queue one approved message as a prompt in a dcode session.

        Args:
            target: Session returned by `sessions`.
            message: Plain-text message to send.

        Raises:
            RuntimeError: The socket was not ours or dcode did not accept the message.
        """  # raised by `_deliver`
        if not message.strip() or len(message.encode()) > _MAX_MESSAGE_BYTES:
            return {"error": f"Message must be 1-{_MAX_MESSAGE_BYTES} UTF-8 bytes."}
        if not self._allow(target.pid):
            return {"error": "Too many dcode messages just now; wait a minute."}
        correlation_id = str(uuid4())
        await self._deliver(target, prompt_line(message, correlation_id), correlation_id)
        return {"status": "delivered", "session": str(target.pid), "request_id": correlation_id}

    async def _deliver(self, target: DcodeSession, line: bytes, correlation_id: str) -> None:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(str(target.socket), limit=_MAX_ACK_BYTES),
            timeout=_IO_SECONDS,
        )
        try:
            sock = writer.get_extra_info("socket")
            if sock is None or peer_uid(sock) != os.getuid():
                msg = "The dcode session socket is not owned by this user."
                raise RuntimeError(msg)
            writer.write(line)
            await asyncio.wait_for(writer.drain(), timeout=_IO_SECONDS)
            ack = await asyncio.wait_for(reader.readline(), timeout=_IO_SECONDS)
            if not _acknowledged(ack, correlation_id):
                msg = "The dcode session did not accept the message."
                raise RuntimeError(msg)
        finally:
            writer.close()
            with suppress(OSError):
                await writer.wait_closed()

    def _allow(self, pid: int) -> bool:
        cutoff = self._clock() - 60
        while self._sends and self._sends[0][0] <= cutoff:
            self._sends.popleft()
        if (
            len(self._sends) >= _MAX_SENDS_PER_MINUTE
            or sum(target == pid for _, target in self._sends) >= _MAX_SENDS_PER_SESSION
        ):
            return False
        self._sends.append((self._clock(), pid))
        return True

    def _resolve(self, wanted: str) -> DcodeSession | str:
        if not _PID.fullmatch(wanted.strip()):
            return "Pass a session from list_dcode_sessions."
        pid = int(wanted.strip())
        match = next((session for session in self.sessions() if session.pid == pid), None)
        return match or "No running dcode session has that id; call list_dcode_sessions."

    def _tools(self) -> list[BaseTool]:
        @tool
        def list_dcode_sessions() -> list[dict[str, str]] | str:
            """List the operator's running dcode sessions that accept messages."""
            if reason := _denied():
                return reason
            return [session.listing() for session in self.sessions()]

        @tool(SEND_TOOL)
        async def message_dcode_session(session: str, message: str) -> dict[str, str] | str:
            """Send an approved message to a running dcode session as a queued prompt.

            dcode confirms delivery but sends no reply back; the operator reads the
            answer in that session. Do not poll or wait for a reply.

            Args:
                session: Session id from `list_dcode_sessions`.
                message: Plain text to send; keep it under 8000 UTF-8 bytes.
            """
            if reason := _denied():
                return reason
            target = self._resolve(session)
            if isinstance(target, str):
                return target
            try:
                return await self.send(target, message)
            except (RuntimeError, TimeoutError, OSError) as error:
                detail = str(error) if isinstance(error, RuntimeError) else type(error).__name__
                return f"Could not message that dcode session: {detail}"

        return [list_dcode_sessions, message_dcode_session]
