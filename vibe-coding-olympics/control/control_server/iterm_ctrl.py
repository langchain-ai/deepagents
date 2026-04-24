"""Async helpers for targeting `play.sh`-launched player sessions.

Mirrors the reusable parts of `../control.py` so the web UI does not
shell out. The CLI script remains authoritative — keep both in sync if
session tagging conventions ever change.
"""

from __future__ import annotations

import asyncio

import iterm2

SESSION_PREFIX = "vibe-player-"

# `/quit` is a QUEUED slash command; give the CLI a beat to tear down
# the Textual app before piping `deepagents` back in.
RESET_QUIT_GRACE_SECS = 1.0


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
                    port = name[len(SESSION_PREFIX):]
                port = str(port)
                if ports is None or port in ports:
                    out.append((port, session))
    return out


async def list_players() -> list[str]:
    """Return the ports of every active player session."""
    return [port for port, _ in await matching_sessions(None)]


async def clear_players(ports: list[str] | None) -> list[str]:
    """Send `/clear` to the targeted player sessions.

    Args:
        ports: Ports to target. `None` targets every active session.

    Returns:
        List of ports that were actually cleared.
    """
    cleared: list[str] = []
    for port, session in await matching_sessions(ports):
        await session.async_send_text("/clear\n")
        cleared.append(port)
    return cleared


async def reset_players(ports: list[str] | None) -> list[str]:
    """Quit and relaunch the targeted CLIs back to the splash screen.

    Uses `/quit` (an always-bypass slash command) so a mid-turn CLI
    still exits cleanly, then re-invokes `deepagents` from the CLI's
    existing cwd — which still has `VIBE_*` env vars intact.

    Args:
        ports: Ports to target. `None` targets every active session.

    Returns:
        List of ports that were actually reset.
    """
    reset: list[str] = []
    for port, session in await matching_sessions(ports):
        await session.async_send_text("/quit\n")
        await asyncio.sleep(RESET_QUIT_GRACE_SECS)
        await session.async_send_text("deepagents\n")
        reset.append(port)
    return reset
