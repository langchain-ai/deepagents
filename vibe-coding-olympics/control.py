#!/usr/bin/env python3
"""Dispatch commands to running Vibe Coding Olympics player CLIs.

Each `play.sh` invocation tags its primary iTerm2 session with the name
`vibe-player-<port>` (and sets `user.vibe_player=<port>`). This script
discovers those sessions via the iTerm2 Python API and sends keystrokes or
text to them.

Usage:
    ./control.py list
    ./control.py clear (--port PORT | --all)
    ./control.py reset (--port PORT | --all)

Run via `uv run` so the project's `iterm2` dep is resolved:
    uv run --project . control.py list
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Awaitable, Callable

import iterm2

SESSION_PREFIX = "vibe-player-"

# Seconds to wait after `/quit` before relaunching `deepagents`. The CLI's
# `/quit` is a QUEUED slash command — it can take a beat to tear down the
# Textual app and return to the shell prompt. One second is comfortable
# without feeling laggy.
RESET_QUIT_GRACE_SECS = 1.0


async def _matching_sessions(
    connection: iterm2.Connection,
    ports: list[str] | None,
) -> list[tuple[str, iterm2.Session]]:
    """Find iTerm2 sessions tagged as vibe players.

    Args:
        connection: Active iTerm2 API connection.
        ports: Only return sessions whose port is in this list. `None`
            returns every player session across all windows.

    Returns:
        `(port, session)` pairs, one per matching session.
    """
    app = await iterm2.async_get_app(connection)
    if app is None:
        return []
    out: list[tuple[str, iterm2.Session]] = []
    for window in app.windows:
        for tab in window.tabs:
            for session in tab.sessions:
                port = await session.async_get_variable("user.vibe_player")
                if not port:
                    # Fall back to session name in case the user-var was
                    # never set (e.g., old play.sh run).
                    name = await session.async_get_variable("session.name") or ""
                    if not name.startswith(SESSION_PREFIX):
                        continue
                    port = name[len(SESSION_PREFIX) :]
                port = str(port)
                if ports is None or port in ports:
                    out.append((port, session))
    return out


def _resolve_targets(args: argparse.Namespace) -> list[str] | None:
    """Return the port list to target, or `None` for all."""
    if args.all:
        return None
    return [args.port]


async def _cmd_list(connection: iterm2.Connection, _args: argparse.Namespace) -> int:
    matches = await _matching_sessions(connection, None)
    if not matches:
        print("No active player sessions found.")
        return 0
    for port, _ in matches:
        print(f"vibe-player-{port}")
    return 0


async def _cmd_clear(connection: iterm2.Connection, args: argparse.Namespace) -> int:
    ports = _resolve_targets(args)
    matches = await _matching_sessions(connection, ports)
    if not matches:
        print("No matching player sessions.", file=sys.stderr)
        return 1
    for port, session in matches:
        await session.async_send_text("/clear\n")
        print(f"cleared vibe-player-{port}")
    return 0


async def _cmd_reset(connection: iterm2.Connection, args: argparse.Namespace) -> int:
    ports = _resolve_targets(args)
    matches = await _matching_sessions(connection, ports)
    if not matches:
        print("No matching player sessions.", file=sys.stderr)
        return 1
    for port, session in matches:
        # `/quit` is an ALWAYS-bypass slash command, so it exits cleanly
        # even if the CLI is mid-turn. After the CLI exits we're back at
        # the shell prompt in $VIBE_DIR with VIBE_* env vars intact, so
        # `deepagents` (bare, no args) drops straight into the splash.
        await session.async_send_text("/quit\n")
        await asyncio.sleep(RESET_QUIT_GRACE_SECS)
        await session.async_send_text("deepagents\n")
        print(f"reset vibe-player-{port}")
    return 0


HANDLERS: dict[str, Callable[[iterm2.Connection, argparse.Namespace], Awaitable[int]]] = {
    "list": _cmd_list,
    "clear": _cmd_clear,
    "reset": _cmd_reset,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dispatch commands to running Vibe Olympics player CLIs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List active player sessions")

    for name, help_text in (
        ("clear", "Send /clear to player CLI(s)"),
        ("reset", "Quit and relaunch player CLI(s) back to the splash screen"),
    ):
        p = sub.add_parser(name, help=help_text)
        target = p.add_mutually_exclusive_group(required=True)
        target.add_argument("--port", help="Target a single player by port")
        target.add_argument("--all", action="store_true", help="Target every player")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    handler = HANDLERS[args.cmd]
    exit_code = 0

    async def runner(connection: iterm2.Connection) -> None:
        nonlocal exit_code
        exit_code = await handler(connection, args)

    iterm2.run_until_complete(runner)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
