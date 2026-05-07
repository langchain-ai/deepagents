"""CLI for dispatching commands to running Vibe Olympics player CLIs.

Thin wrapper over `iterm_ctrl`; the iTerm2 session-discovery contract
lives there so both this CLI and the web server in `app.py` agree on
what counts as a player session.

Usage:
    vibe-players list
    vibe-players prompt PROMPT (--port PORT | --all)
    vibe-players times-up (--port PORT | --all)
    vibe-players clear (--port PORT | --all)
    vibe-players reset (--port PORT | --all)
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from control_server import iterm_ctrl


def _resolve_targets(args: argparse.Namespace) -> list[str] | None:
    """Return the port list to target, or `None` for all."""
    if args.all:
        return None
    return [args.port]


async def _cmd_list(_args: argparse.Namespace) -> int:
    """Print every discovered player session, one per line."""
    ports = await iterm_ctrl.list_players()
    if not ports:
        print("No active player sessions found.")
        return 0
    for port in ports:
        print(f"{iterm_ctrl.SESSION_PREFIX}{port}")
    return 0


async def _cmd_clear(args: argparse.Namespace) -> int:
    """Send a socket `force-clear` signal to matching sessions."""
    cleared = await iterm_ctrl.clear_players(_resolve_targets(args))
    if not cleared:
        print("No matching player sessions.", file=sys.stderr)
        return 1
    for port in cleared:
        print(f"cleared {iterm_ctrl.SESSION_PREFIX}{port}")
    return 0


async def _cmd_prompt(args: argparse.Namespace) -> int:
    """Inject a prompt into matching player CLIs."""
    sent = await iterm_ctrl.send_prompt_to_players(
        _resolve_targets(args), args.prompt
    )
    if not sent:
        print("No matching player sessions.", file=sys.stderr)
        return 1
    for port in sent:
        print(f"sent prompt to {iterm_ctrl.SESSION_PREFIX}{port}")
    return 0


async def _cmd_times_up(args: argparse.Namespace) -> int:
    """Send a times-up signal to matching player CLIs."""
    sent = await iterm_ctrl.times_up_players(_resolve_targets(args))
    if not sent:
        print("No matching player sessions.", file=sys.stderr)
        return 1
    for port in sent:
        print(f"sent times-up to {iterm_ctrl.SESSION_PREFIX}{port}")
    return 0


async def _cmd_reset(args: argparse.Namespace) -> int:
    """Quit and relaunch matching player CLIs back to the splash screen."""
    reset = await iterm_ctrl.reset_players(_resolve_targets(args))
    if not reset:
        print("No matching player sessions.", file=sys.stderr)
        return 1
    for port in reset:
        print(f"reset {iterm_ctrl.SESSION_PREFIX}{port}")
    return 0


_HANDLERS = {
    "list": _cmd_list,
    "prompt": _cmd_prompt,
    "times-up": _cmd_times_up,
    "clear": _cmd_clear,
    "reset": _cmd_reset,
}


def _parse_args() -> argparse.Namespace:
    """Build the argparse tree shared by all subcommands."""
    parser = argparse.ArgumentParser(
        prog="vibe-players",
        description="Dispatch commands to running Vibe Olympics player CLIs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List active player sessions")

    p = sub.add_parser("prompt", help="Send a prompt to player CLI(s)")
    p.add_argument("prompt", help="Prompt to inject into the player CLI")
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument("--port", help="Target a single player by port")
    target.add_argument("--all", action="store_true", help="Target every player")

    for name, help_text in (
        ("times-up", "Send a times-up signal to player CLI(s)"),
        ("clear", "Send a socket force-clear signal to player CLI(s)"),
        ("reset", "Quit and relaunch player CLI(s) back to the splash screen"),
    ):
        p = sub.add_parser(name, help=help_text)
        target = p.add_mutually_exclusive_group(required=True)
        target.add_argument("--port", help="Target a single player by port")
        target.add_argument("--all", action="store_true", help="Target every player")

    return parser.parse_args()


def main() -> None:
    """Entry point for the `vibe-players` console script."""
    args = _parse_args()
    handler = _HANDLERS[args.cmd]
    sys.exit(asyncio.run(handler(args)))


if __name__ == "__main__":
    main()
