"""Async helpers for targeting `play.sh`-launched player sessions.

Single source of truth for the iTerm2 session-discovery contract. The
`vibe-control` web server imports these helpers; `play.sh` is the producer side
of the same contract (tagging new sessions with `user.vibe_player` + matching
name).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import iterm2

from control_server import deepagents_config
from control_server.event_socket import send_socket_event

SESSION_PREFIX = "vibe-player-"
SOCKET_VARIABLE = "user.vibe_event_socket"

logger = logging.getLogger(__name__)

_connection: iterm2.Connection | None = None
_connection_lock: asyncio.Lock | None = None


async def _get_connection() -> iterm2.Connection:
    """Return a shared iTerm2 API connection for the control server.

    `Connection.async_create()` authenticates by spawning an iTerm2 helper
    subprocess. The web control page polls `/api/players`, so creating a new
    connection per request can exhaust the process file-descriptor limit.
    """
    global _connection, _connection_lock  # noqa: PLW0603
    if _connection is not None:
        return _connection
    if _connection_lock is None:
        _connection_lock = asyncio.Lock()
    async with _connection_lock:
        if _connection is None:
            _connection = await iterm2.Connection.async_create()
        return _connection


def _drop_connection() -> None:
    """Forget the shared iTerm2 connection after a transport failure."""
    global _connection  # noqa: PLW0603
    _connection = None


async def matching_sessions(
    ports: list[str] | None,
) -> list[tuple[str, iterm2.Session]]:
    """Return `(port, session)` pairs for vibe-player sessions.

    Args:
        ports: Only return sessions whose port is in this list. `None`
            returns every player session across all iTerm2 windows.

    Returns:
        List of `(port, session)` pairs; empty if iTerm2 is not running
        or no player sessions match.
    """
    connection = await _get_connection()
    try:
        app = await iterm2.async_get_app(connection)
    except Exception:
        _drop_connection()
        connection = await _get_connection()
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
    deepagents_config.clear_recent_model()
    await _send_socket_event(
        socket_path,
        kind="signal",
        payload="force-clear",
        correlation_prefix="vibe-clear",
    )


async def _send_socket_event(
    socket_path: Path,
    *,
    kind: str,
    payload: str,
    correlation_prefix: str,
) -> None:
    """Send one JSON-lines external event to a player CLI socket.

    Args:
        socket_path: Unix-domain socket path exported by the player CLI.
        kind: External event kind accepted by the CLI event bus.
        payload: External event payload.
        correlation_prefix: Prefix used for the ACK correlation id.

    Raises:
        OSError: If the socket cannot be reached.
        TimeoutError: If the CLI does not acknowledge the event in time.
        RuntimeError: If the CLI returns a negative acknowledgement.
        json.JSONDecodeError: If the CLI returns malformed JSON.
    """
    await send_socket_event(
        socket_path,
        kind=kind,
        payload=payload,
        correlation_prefix=correlation_prefix,
    )


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


async def send_prompt_to_players(ports: list[str] | None, prompt: str) -> list[str]:
    """Inject the round prompt into targeted player CLIs.

    Args:
        ports: Ports to target. `None` targets every active session.
        prompt: Creative prompt selected by the controller.

    Returns:
        List of ports that received the prompt.
    """
    sent: list[str] = []
    message = f"/skill:web-vibe Prompt: {prompt}"
    for port, session in await matching_sessions(ports):
        socket_path = await _event_socket_for_session(session)
        if socket_path is None:
            logger.warning("Player %s has no %s variable", port, SOCKET_VARIABLE)
            continue
        try:
            await _send_socket_event(
                socket_path,
                kind="command",
                payload=message,
                correlation_prefix="vibe-prompt",
            )
        except (OSError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to send prompt to player %s via external event socket %s: %s",
                port,
                socket_path,
                exc,
            )
            continue
        sent.append(port)
    return sent


async def times_up_players(ports: list[str] | None) -> list[str]:
    """Send a `times-up` signal to targeted player CLIs.

    Args:
        ports: Ports to target. `None` targets every active session.

    Returns:
        List of ports that received the signal.
    """
    sent: list[str] = []
    for port, session in await matching_sessions(ports):
        socket_path = await _event_socket_for_session(session)
        if socket_path is None:
            logger.warning("Player %s has no %s variable", port, SOCKET_VARIABLE)
            continue
        try:
            await _send_socket_event(
                socket_path,
                kind="signal",
                payload="times-up",
                correlation_prefix="vibe-times-up",
            )
        except (OSError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to send times-up to player %s via external event socket %s: %s",
                port,
                socket_path,
                exc,
            )
            continue
        sent.append(port)
    return sent


async def players_ready(ports: list[str] | None) -> list[str]:
    """Notify targeted player CLIs that both players are ready to start.

    Args:
        ports: Ports to target. `None` targets every active session.

    Returns:
        List of ports that received the signal.
    """
    sent: list[str] = []
    for port, session in await matching_sessions(ports):
        socket_path = await _event_socket_for_session(session)
        if socket_path is None:
            logger.warning("Player %s has no %s variable", port, SOCKET_VARIABLE)
            continue
        try:
            await _send_socket_event(
                socket_path,
                kind="signal",
                payload="players-ready",
                correlation_prefix="vibe-control-players-ready",
            )
        except (OSError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to send players-ready to player %s via external event "
                "socket %s: %s",
                port,
                socket_path,
                exc,
            )
            continue
        sent.append(port)
    return sent
