#!/usr/bin/env bash
# Launch a fresh player round in a new iTerm2 window.
#
# Layout:
#   top pane    — deepagents CLI (optionally pre-loaded with web-vibe + prompt)
#   bottom pane — tail -f of the Vite log, so server activity is visible
#
# Requires (one-time per laptop):
#   - iTerm2 installed
#   - Prefs → General → Magic → "Enable Python API" is ON
#   - `pip install iterm2` on the python3 that runs this script
#
# Usage:
#   ./play.sh PROMPT [PORT]
#
#   PROMPT  required; the creative brief for the round. Queued as the first
#           message via /skill:web-vibe.
#   PORT    optional; defaults to 3001.

set -euo pipefail

if [ $# -lt 1 ] || [ -z "${1:-}" ]; then
  echo "usage: $0 PROMPT [PORT]" >&2
  echo "  PROMPT is required -- e.g. \"a website for a taco truck\"" >&2
  exit 64
fi

PROMPT="$1"
PORT="${2:-3001}"

DIR=$(mktemp -d -t "vibe-player-${PORT}-XXXX")
LOG="/tmp/vite.log"
: > "$LOG"   # ensure tail -f has a file to open even on first-ever run

# Expose the repo's .deepagents/ (skills, configs) to the fresh round dir so
# the CLI's project-level skill lookup finds web-vibe. The CLI's project
# detection (`find_project_root`) walks up looking for `.git/`, so we also
# drop an empty `.git/` marker — that's the signal that anchors the project
# and activates the `<project>/.deepagents/skills/` lookup.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DEEPAGENTS="$(cd "$SCRIPT_DIR/.." && pwd)/.deepagents"
if [ -d "$REPO_DEEPAGENTS" ]; then
  ln -s "$REPO_DEEPAGENTS" "$DIR/.deepagents"
  mkdir -p "$DIR/.git"
else
  echo "warning: $REPO_DEEPAGENTS not found; web-vibe skill will be missing" >&2
fi

echo "Player dir: $DIR"
echo "Port:       $PORT"
echo "Log:        $LOG"

export VIBE_PORT="$PORT" VIBE_DIR="$DIR" VIBE_PROMPT="$PROMPT" VIBE_LOG="$LOG"

# Bring iTerm2 to the foreground before the Python API connects.
open -a iTerm

# Resolve the python that has `iterm2` installed. Prefer this project's venv
# (populated by `uv sync`); fall back to bare python3 for users who installed
# iterm2 globally.
if command -v uv >/dev/null 2>&1 && [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
  PY_RUN=(uv run --project "$SCRIPT_DIR" python3)
else
  PY_RUN=(python3)
fi

"${PY_RUN[@]}" - <<'PY'
import os
import shlex

import iterm2

PORT = os.environ["VIBE_PORT"]
DIR = os.environ["VIBE_DIR"]
PROMPT = os.environ["VIBE_PROMPT"].strip()
LOG = os.environ["VIBE_LOG"]

if not PROMPT:
    raise SystemExit("VIBE_PROMPT is empty; play.sh must pass a non-empty prompt")


async def main(connection):
    # async_get_app wires up Session.delegate etc. Without this, split_pane
    # and other session-delegate operations fail with AssertionError.
    await iterm2.async_get_app(connection)

    window = await iterm2.Window.async_create(connection)
    if window is None or not window.tabs:
        raise RuntimeError("iterm2 did not return a window with tabs")
    tab = window.tabs[0]
    sessions = tab.sessions
    if not sessions:
        raise RuntimeError("new iterm2 tab has no sessions")
    top = sessions[0]

    # Prime env + cwd in the CLI pane so the agent's shell-outs see VIBE_* vars.
    await top.async_send_text(
        "export "
        f"VIBE_PORT={shlex.quote(PORT)} "
        f"VIBE_DIR={shlex.quote(DIR)} "
        f"VIBE_LOG={shlex.quote(LOG)}\n"
    )
    await top.async_send_text(f"cd {shlex.quote(DIR)}\n")

    cli_cmd = f"deepagents -y --skill web-vibe -m {shlex.quote(PROMPT)}"
    await top.async_send_text(cli_cmd + "\n")

    # Bottom pane follows the Vite log so a crash is visible without alt-tab.
    bottom = await top.async_split_pane(vertical=False)
    await bottom.async_send_text(f"tail -f {shlex.quote(LOG)}\n")


iterm2.run_until_complete(main)
PY
