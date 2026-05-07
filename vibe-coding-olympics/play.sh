#!/usr/bin/env bash
# Launch a fresh player round in a new iTerm2 window.
#
# Layout:
#   tab 1 — starts the Vite dev server once, then launches Deep Agents CLI
#           waiting for the controller or auto-invoking web-vibe with PROMPT
#   tab 2 — tail -f of the Vite log, so server activity is visible
#
# Requires (one-time per laptop):
#   - iTerm2 installed
#   - Prefs → General → Magic → "Enable Python API" is ON
#   - `pip install iterm2` on the python3 that runs this script
#
# Usage:
#   ./play.sh [PORT]
#   ./play.sh --prompt PROMPT [PORT]
#
#   Default behavior launches the player CLI and waits for the controller to
#   inject the prompt after both players are ready.
#   --prompt  creative brief for standalone smoke tests. Queued as the first
#             message via /skill:web-vibe.
#   PORT    optional; defaults to 3001.

set -euo pipefail

WAIT_FOR_CONTROLLER=1
PROMPT=""
PORT="3001"

if [ "${1:-}" = "--prompt" ]; then
  WAIT_FOR_CONTROLLER=0
  if [ -z "${2:-}" ]; then
    echo "usage: $0 [PORT] | $0 --prompt PROMPT [PORT]" >&2
    exit 64
  fi
  PROMPT="$2"
  PORT="${3:-3001}"
elif [ $# -ge 1 ] && [ -n "${1:-}" ]; then
  PROMPT="$1"
  if [[ "$PROMPT" =~ ^[0-9]+$ ]]; then
    PORT="$PROMPT"
    PROMPT=""
  else
    WAIT_FOR_CONTROLLER=0
    PORT="${2:-3001}"
  fi
fi

DIR=$(mktemp -d -t "vibe-player-${PORT}-XXXX")
LOG="/tmp/vite-${PORT}.log"
EVENT_SOCKET="/tmp/deepagents-vibe-${PORT}.sock"
HOOKS_FILE="$DIR/hooks.json"
: > "$LOG"   # ensure tail -f has a file to open even on first-ever run

# Expose the repo's .deepagents/ (skills, configs) to the fresh round dir so
# the CLI's project-level skill lookup finds web-vibe. The CLI's project
# detection (`find_project_root`) walks up looking for `.git/`, so we also
# drop an empty `.git/` marker — that's the signal that anchors the project
# and activates the `<project>/.deepagents/skills/` lookup.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTROL_DIR="$SCRIPT_DIR/control"
CONTROL_API="${VIBE_CONTROL_API:-http://localhost:8766}"
REPO_DEEPAGENTS="$(cd "$SCRIPT_DIR/.." && pwd)/.deepagents"
if [ -d "$REPO_DEEPAGENTS" ]; then
  ln -s "$REPO_DEEPAGENTS" "$DIR/.deepagents"
  mkdir -p "$DIR/.git"
else
  echo "warning: $REPO_DEEPAGENTS not found; web-vibe skill will be missing" >&2
fi

python3 - "$HOOKS_FILE" "$CONTROL_DIR" <<'PY'
import json
import sys
from pathlib import Path

hooks_file = Path(sys.argv[1])
control_dir = sys.argv[2]
hooks_file.write_text(
    json.dumps(
        {
            "hooks": [
                {
                    "events": ["competition.player.ready", "user.name.set"],
                    "command": [
                        "uv",
                        "run",
                        "--project",
                        control_dir,
                        "vibe-player-hook",
                    ],
                }
            ]
        }
    )
)
PY

echo "Player dir: $DIR"
echo "Port:       $PORT"
echo "Log:        $LOG"
echo "Socket:     $EVENT_SOCKET"
echo "Hooks:      $HOOKS_FILE"
echo "Control:    $CONTROL_API"
if [ "$WAIT_FOR_CONTROLLER" -eq 1 ]; then
  echo "Mode:       waiting for controller prompt"
fi

export VIBE_PORT="$PORT" VIBE_DIR="$DIR" VIBE_PROMPT="$PROMPT" VIBE_LOG="$LOG"
export VIBE_EVENT_SOCKET="$EVENT_SOCKET"
export DEEPAGENTS_CLI_HOOKS_PATH="$HOOKS_FILE" VIBE_CONTROL_API="$CONTROL_API"
export VIBE_WAIT_FOR_CONTROLLER="$WAIT_FOR_CONTROLLER"

# Free the port up front so the poller below can't race a zombie Vite from a
# previous round. Without this, the poller's first curl would succeed against
# the old server and `open` Firefox on the stale page — which then gets
# killed moments later when start-server.sh tears the zombie down.
# LISTEN-only to avoid SIGKILLing clients (e.g. a still-open browser tab)
# that hold an established connection to the port.
if command -v lsof >/dev/null 2>&1; then
  stale_pids="$(lsof -ti ":$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "$stale_pids" ]; then
    echo "Freeing port $PORT from previous round (pids: $stale_pids)"
    # shellcheck disable=SC2086
    kill -9 $stale_pids 2>/dev/null || true
    sleep 0.3
  fi
fi

# Poll for the Vite server in the background, open the browser as soon as
# it's reachable. Fire-and-forget — waits up to ~60s, then gives up quietly
# so a failed round doesn't leave a zombie poller around.
(
  for _ in $(seq 1 60); do
    if curl -fs -o /dev/null -m 1 "http://localhost:$PORT"; then
      open "http://localhost:$PORT"
      exit 0
    fi
    sleep 1
  done
) >/dev/null 2>&1 &
disown

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
EVENT_SOCKET = os.environ["VIBE_EVENT_SOCKET"]
WAIT_FOR_CONTROLLER = os.environ["VIBE_WAIT_FOR_CONTROLLER"] == "1"
if WAIT_FOR_CONTROLLER:
    STARTUP_SUBHEADER = (
        "Welcome to LangChain Interrupt 2026\n\nWaiting for the prompt..."
    )
elif PROMPT:
    STARTUP_SUBHEADER = "Welcome to LangChain Interrupt 2026\n\nReady to vibecode!"
else:
    raise SystemExit("VIBE_PROMPT is empty; pass a non-empty --prompt value")


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

    # Tag this session so vibe-players / vibe-control can find it later by
    # port. Tag is both the session name (shown in iTerm2's title bar) and a
    # user-variable so the shell's auto-title can't silently clobber it.
    tag = f"vibe-player-{PORT}"
    await top.async_set_name(tag)
    await top.async_set_variable("user.vibe_player", PORT)
    await top.async_set_variable("user.vibe_event_socket", EVENT_SOCKET)

    # Prime env + cwd in the CLI pane so the agent's shell-outs see VIBE_* vars.
    await top.async_send_text(
        "export "
        f"VIBE_PORT={shlex.quote(PORT)} "
        f"VIBE_DIR={shlex.quote(DIR)} "
        f"VIBE_LOG={shlex.quote(LOG)} "
        f"VIBE_EVENT_SOCKET={shlex.quote(EVENT_SOCKET)} "
        f"VIBE_CONTROL_API={shlex.quote(os.environ['VIBE_CONTROL_API'])} "
        "DEEPAGENTS_CLI_HOOKS_PATH="
        f"{shlex.quote(os.environ['DEEPAGENTS_CLI_HOOKS_PATH'])} "
        "DEEPAGENTS_CLI_HIDE_SPLASH_VERSION=1 "
        "DEEPAGENTS_CLI_HIDE_GIT_BRANCH=1 "
        "DEEPAGENTS_CLI_HIDE_CWD=1 "
        "DEEPAGENTS_CLI_HIDE_LANGSMITH_TRACING=1 "
        "DEEPAGENTS_CLI_DEBUG_ONBOARDING=1 "
        f"DEEPAGENTS_CLI_THEME={shlex.quote('langchain dark')} "
        "DEEPAGENTS_CLI_HIDE_NEW_THREAD_MESSAGE=1 "
        "DEEPAGENTS_CLI_HIDE_SPLASH_TIPS=1 "
        "DEEPAGENTS_CLI_HIDE_STARTUP_COMMAND_TEXT=1 "
        "DEEPAGENTS_CLI_EXTERNAL_EVENT_SOCKET=1 "
        f"DEEPAGENTS_CLI_EXTERNAL_EVENT_SOCKET_PATH={shlex.quote(EVENT_SOCKET)} "
        f"DEEPAGENTS_CLI_COMPETITION_WAIT_FOR_START={int(WAIT_FOR_CONTROLLER)} "
        "DEEPAGENTS_CLI_DANGEROUSLY_OVERRIDE_STARTUP_SUBHEADER="
        f"{shlex.quote(STARTUP_SUBHEADER)}\n"
    )
    await top.async_send_text(f"cd {shlex.quote(DIR)}\n")

    # Start the browser preview once at player launch. Later controller rounds
    # reuse this server and browser tab; force-clear only resets the CLI state.
    startup_cmd = 'bash "$VIBE_DIR/.deepagents/skills/web-vibe/start-server.sh"'
    if WAIT_FOR_CONTROLLER:
        deepagents_cmd = "deepagents -y"
    else:
        deepagents_cmd = f"deepagents -y --skill web-vibe -m {shlex.quote(PROMPT)}"
    cli_cmd = f"{startup_cmd} && {deepagents_cmd}"
    await top.async_send_text(cli_cmd + "\n")

    # Second tab follows the Vite log so a crash is visible by switching tabs.
    log_tab = await window.async_create_tab()
    if log_tab is None or not log_tab.sessions:
        raise RuntimeError("iterm2 did not return a log tab with sessions")
    await log_tab.sessions[0].async_send_text(f"tail -f {shlex.quote(LOG)}\n")

    # Refocus tab 1 — creating the log tab leaves it active otherwise.
    await tab.async_activate()


iterm2.run_until_complete(main)
PY
