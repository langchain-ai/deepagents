"""Async helpers for targeting `play.sh`-launched player sessions.

Single source of truth for the iTerm2 session-discovery contract.
Both the `vibe-players` CLI and the `vibe-control` web server import
these helpers; `play.sh` is the producer side of the same contract
(tagging new sessions with `user.vibe_player` + matching name).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from uuid import uuid4

import iterm2

SESSION_PREFIX = "vibe-player-"
SOCKET_VARIABLE = "user.vibe_event_socket"
SOCKET_TIMEOUT_SECS = 2.0

logger = logging.getLogger(__name__)

# `/quit` is a QUEUED slash command; give the CLI a beat to tear down
# the Textual app before piping `deepagents -y` back in.
RESET_QUIT_GRACE_SECS = 1.0

# Gap between typing a slash command and sending Enter. Without it,
# iTerm2 flushes the whole buffer at once and Textual's `Input` submits
# before the trailing chars are ingested, truncating the command.
SUBMIT_GRACE_SECS = 0.05


async def _submit_slash(session: iterm2.Session, command: str) -> None:
    """Type a slash command into a Textual CLI and press Enter.

    Enter in raw terminal mode is CR (`\\r`), not LF (`\\n`); Textual's
    input widget won't treat `\\n` as submit. Splitting the text and the
    Enter into two sends also lets the event loop ingest the full
    command before the submit arrives.
    """
    await session.async_send_text(command)
    await asyncio.sleep(SUBMIT_GRACE_SECS)
    await session.async_send_text("\r")


async def matching_sessions(
    ports: list[str] | None,
) -> list[tuple[str, iterm2.Session]]:
    """Return `(port, session)` pairs for vibe-player sessions.

    Opens a fresh iTerm2 connection each call; short-lived requests
    don't benefit from pooling and a shared connection would have to
    survive the FastAPI reload cycle.

    Args:
        ports: Only return sessions whose port is in this list. `None`
            returns every player session across all iTerm2 windows.

    Returns:
        List of `(port, session)` pairs; empty if iTerm2 is not running
        or no player sessions match.
    """
    connection = await iterm2.Connection.async_create()
    app = await iterm2.async_get_app(connection)
    if app is None:
        return []
    out: list[tuple[str, iterm2.Session]] = []
    for window in app.windows:
        for tab in window.tabs:
            for session in tab.sessions:
                port = await session.async_get_variable("user.vibe_player")
                if not port:
                    name = await session.async_get_variable("session.name") or ""
                    if not name.startswith(SESSION_PREFIX):
                        continue
                    port = name[len(SESSION_PREFIX) :]
                port = str(port)
                if ports is None or port in ports:
                    out.append((port, session))
    return out


async def list_players() -> list[str]:
    """Return the ports of every active player session."""
    return [port for port, _ in await matching_sessions(None)]


async def _event_socket_for_session(session: iterm2.Session) -> Path | None:
    """Return the external-event socket path advertised by a player session."""
    raw = await session.async_get_variable(SOCKET_VARIABLE)
    if not raw:
        return None
    return Path(str(raw))


async def _send_force_clear(socket_path: Path) -> None:
    """Ask a running CLI to force-clear via its external event socket.

    Args:
        socket_path: Unix-domain socket path exported by the player CLI.

    Raises:
        OSError: If the socket cannot be reached.
        TimeoutError: If the CLI does not acknowledge the event in time.
        RuntimeError: If the CLI returns a negative acknowledgement.
    """
    correlation_id = f"vibe-clear-{uuid4().hex}"
    envelope = {
        "kind": "signal",
        "payload": "force-clear",
        "correlation_id": correlation_id,
    }
    reader, writer = await asyncio.wait_for(
        asyncio.open_unix_connection(str(socket_path)),
        timeout=SOCKET_TIMEOUT_SECS,
    )
    try:
        writer.write(json.dumps(envelope).encode("utf-8") + b"\n")
        await asyncio.wait_for(writer.drain(), timeout=SOCKET_TIMEOUT_SECS)
        line = await asyncio.wait_for(reader.readline(), timeout=SOCKET_TIMEOUT_SECS)
    finally:
        writer.close()
        await writer.wait_closed()

    if not line:
        msg = f"External event socket {socket_path} closed without an ACK"
        raise RuntimeError(msg)
    response = json.loads(line)
    if not isinstance(response, dict):
        msg = f"External event socket {socket_path} returned a non-object ACK"
        raise RuntimeError(msg)
    if response.get("ok") is not True:
        error = response.get("error", "unknown error")
        msg = f"External event socket {socket_path} rejected force-clear: {error}"
        raise RuntimeError(msg)


async def clear_players(ports: list[str] | None) -> list[str]:
    """Send a socket `force-clear` signal to the targeted player CLIs.

    Args:
        ports: Ports to target. `None` targets every active session.

    Returns:
        List of ports that were actually cleared.
    """
    cleared: list[str] = []
    for port, session in await matching_sessions(ports):
        socket_path = await _event_socket_for_session(session)
        if socket_path is None:
            logger.warning("Player %s has no %s variable", port, SOCKET_VARIABLE)
            continue
        try:
            await _send_force_clear(socket_path)
        except (OSError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to clear player %s via external event socket %s: %s",
                port,
                socket_path,
                exc,
            )
            continue
        cleared.append(port)
    return cleared


async def reset_players(ports: list[str] | None) -> list[str]:
    """Quit and relaunch the targeted CLIs back to the splash screen.

    Uses `/quit` (an always-bypass slash command) so a mid-turn CLI
    still exits cleanly, then re-invokes `deepagents -y` from the CLI's
    existing cwd — which still has `VIBE_*` env vars intact.

    Args:
        ports: Ports to target. `None` targets every active session.

    Returns:
        List of ports that were actually reset.
    """
    reset: list[str] = []
    for port, session in await matching_sessions(ports):
        await _submit_slash(session, "/quit")
        await asyncio.sleep(RESET_QUIT_GRACE_SECS)
        await session.async_send_text("deepagents -y\n")
        reset.append(port)
    return reset
