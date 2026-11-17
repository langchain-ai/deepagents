"""Print the MXC policy MXCSandbox will enforce for a folder, and optionally exercise it.

Usage::

    uv run python examples/show_policy.py [ROOT_DIR] [--network] [--ui] [--clipboard LEVEL] [--exercise]

With no argument the current directory is used. ``--exercise`` runs a handful
of commands through the sandbox (no API key needed) so you can see what an
agent would get back for allowed and blocked actions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from langchain_mxc import MXCExecutorNotFoundError, MXCSandbox

HOME = Path.home()

POSIX_COMMANDS = [
    ("allowed: run the toolchain", "python --version && git --version"),
    ("allowed: write inside the project", "echo sandboxed > .mxc-demo && cat .mxc-demo && rm .mxc-demo"),
    ("allowed: write to the scratch $HOME", 'echo scratch > "$HOME/note.txt" && cat "$HOME/note.txt"'),
    ("blocked: write outside the project", f"touch {HOME / 'mxc-should-not-exist.txt'}"),
    ("blocked: read credentials", f"ls {HOME / '.ssh'}"),
    ("network (off by default)", "curl -sS --max-time 5 https://example.com -o /dev/null && echo reached network"),
]

WINDOWS_COMMANDS = [
    ("allowed: run Python", "python --version"),
    ("git (on Windows it needs the window system, so it only starts with --ui)", "git --version"),
    ("allowed: write inside the project", "echo sandboxed> .mxc-demo && type .mxc-demo && del .mxc-demo"),
    ("allowed: write to the scratch %USERPROFILE%", 'echo scratch> "%USERPROFILE%\\note.txt" && type "%USERPROFILE%\\note.txt"'),
    ("blocked: write outside the project", f'echo x> "{HOME / "mxc-should-not-exist.txt"}"'),
    ("blocked: read credentials", f'dir "{HOME / ".ssh"}"'),
    ("network (off by default)", "curl.exe -sS --max-time 5 http://example.com -o NUL && echo reached network"),
]

EXERCISE_COMMANDS = WINDOWS_COMMANDS if sys.platform == "win32" else POSIX_COMMANDS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root_dir", nargs="?", default=os.getcwd(), help="project folder the agent may write to")
    parser.add_argument("--network", action="store_true", help="allow outbound network")
    parser.add_argument("--ui", action="store_true", help="let commands create windows")
    parser.add_argument("--clipboard", default="none", choices=["none", "read", "write", "all"], help="clipboard access for commands")
    parser.add_argument("--exercise", action="store_true", help="run sample commands through the sandbox")
    args = parser.parse_args()

    try:
        sandbox = MXCSandbox(root_dir=args.root_dir, allow_network=args.network, allow_ui=args.ui, clipboard=args.clipboard, timeout=30)
    except MXCExecutorNotFoundError as exc:
        print(exc)
        return 2

    policy = sandbox.policy
    print(f"sandbox id:  {sandbox.id}")
    print(f"executor:    {sandbox.executor_path}")
    print(f"root_dir:    {policy.root_dir}")
    print(f"scratch dir: {sandbox.scratch_dir}  (HOME/TMPDIR inside the sandbox)")
    print(f"network:     {'allowed' if policy.allow_network else 'denied'}")
    ui = "allowed" if policy.allow_ui else "disabled"
    injection = "on" if policy.allow_input_injection else "off"
    print(f"ui:          {ui}, clipboard={policy.clipboard}, input injection={injection}")
    print("\nread-write paths:")
    for path in policy.read_write_paths:
        print(f"  {path}")
    print("\nread-only paths (toolchain + system):")
    for path in policy.read_only_paths:
        print(f"  {path}")
    print("\ndenied paths (credential folders that exist on this host):")
    for path in policy.denied_paths or ["  (none found)"]:
        print(f"  {path}")
    print("\nrequest sent to the executor for `echo hi` (10 s timeout):")
    print(json.dumps(sandbox.build_request("echo hi", 10), indent=2))

    if not args.exercise:
        print("\nRe-run with --exercise to run sample commands through the sandbox.")
        return 0

    print("\n=== exercising the policy ===")
    for label, command in EXERCISE_COMMANDS:
        print(f"\n--- {label}\n$ {command}")
        response = sandbox.execute(command)
        for line in response.output.splitlines():
            print(f"    {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
