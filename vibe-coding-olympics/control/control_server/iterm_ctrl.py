"""Async helpers for targeting `play.sh`-launched player sessions.

Single source of truth for the iTerm2 session-discovery contract.
Both the `vibe-players` CLI and the `vibe-control` web server import
these helpers; `play.sh` is the producer side of the same contract
(tagging new sessions with `user.vibe_player` + matching name).
"""

from __future__ import annotations

import asyncio

import iterm2

SESSION_PREFIX = "vibe-player-"

# `/quit` is a QUEUED slash command; give the CLI a beat to tear down
# the Textual app before piping `deepagents` back in.
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
        await _submit_slash(session, "/clear")
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
        await _submit_slash(session, "/quit")
        await asyncio.sleep(RESET_QUIT_GRACE_SECS)
        await session.async_send_text("deepagents\n")
        reset.append(port)
    return reset
