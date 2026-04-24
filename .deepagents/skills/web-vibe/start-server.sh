#!/usr/bin/env bash
# Idempotent dev-server startup for the web-vibe skill.
#
# Ensures a Vite project exists in $VIBE_DIR (default /tmp/vibe-round) and a
# dev server is serving it on $VIBE_PORT (default 5173). The script chdir's
# into $VIBE_DIR so the agent gets a consistent project root even if its own
# cwd drifts between turns.
#
# Idempotency: re-running is a no-op ONLY when the port is already healthy
# AND the process holding it has a cwd matching $VIBE_DIR. If /tmp was wiped
# between rounds and a zombie Vite is still bound, we kill + restart.
#
# Scaffold strategy (in priority order):
#   1. cwd has package.json → reuse
#   2. $VIBE_TEMPLATE exists → cp -a into cwd (fast, offline)
#   3. Fall back to `npm create vite` + `npm install` (slow, needs network)
#
# Env:
#   VIBE_DIR         Round project dir (default /tmp/vibe-round)
#   VIBE_PORT        Port to serve on (default 5173)
#   VIBE_LOG         Log file path (default /tmp/vite.log)
#   VIBE_TEMPLATE    Pre-built template dir (default ~/.vibe-template)
#   VIBE_NPM_TIMEOUT Seconds before npm commands are killed (default 45)
#
# Exits 0 with server URL on stdout; non-zero with log tail on stderr.

set -euo pipefail

ROUND_DIR="${VIBE_DIR:-/tmp/vibe-round}"
PORT="${VIBE_PORT:-5173}"
LOG_FILE="${VIBE_LOG:-/tmp/vite.log}"
TEMPLATE="${VIBE_TEMPLATE:-$HOME/.vibe-template}"
NPM_TIMEOUT="${VIBE_NPM_TIMEOUT:-45}"
URL="http://localhost:$PORT"

log() { printf '[web-vibe] %s\n' "$*" >&2; }
die() {
  log "ERROR: $*"
  [ -f "$LOG_FILE" ] && tail -40 "$LOG_FILE" >&2 || true
  exit 1
}

# Hard dependencies. Fail loud rather than silently degrade.
command -v lsof >/dev/null 2>&1 || die "lsof not installed (required for port management)"
command -v curl >/dev/null 2>&1 || die "curl not installed (required for health checks)"
command -v npm  >/dev/null 2>&1 || die "npm not installed"

# `timeout` varies by platform; accept either GNU or coreutils name.
TIMEOUT_CMD="$(command -v timeout || command -v gtimeout || true)"
run_npm() {
  if [ -n "$TIMEOUT_CMD" ]; then
    "$TIMEOUT_CMD" "$NPM_TIMEOUT" "$@"
  else
    "$@"
  fi
}

# Enter the round dir; script operates from here and so does Vite.
mkdir -p "$ROUND_DIR"
cd "$ROUND_DIR"
CWD="$(pwd -P)"

healthy() { curl -fs -o /dev/null -m 1 "$URL" 2>/dev/null; }

# NOTE: all port-to-PID lookups below restrict to TCP LISTEN sockets
# (`-sTCP:LISTEN`). `lsof -ti :PORT` without that flag also returns PIDs of
# *clients* with an established connection to the port — e.g. a browser
# talking to Vite/HMR. Returning those PIDs would make kill_port SIGKILL the
# browser, and make server_cwd_matches fingerprint the browser's cwd instead
# of the server's.
server_cwd_matches() {
  local pid scwd
  pid="$(lsof -ti ":$PORT" -sTCP:LISTEN 2>/dev/null | head -1)"
  [ -n "$pid" ] || return 1
  scwd="$(lsof -p "$pid" 2>/dev/null | awk '$4=="cwd"{print $NF; exit}')"
  [ -n "$scwd" ] || return 1
  [ "$scwd" = "$CWD" ]
}

wait_for_port_free() {
  local deadline=$((SECONDS + 5))
  while [ $SECONDS -lt "$deadline" ]; do
    lsof -ti ":$PORT" -sTCP:LISTEN >/dev/null 2>&1 || return 0
    sleep 0.1
  done
  return 1
}

wait_for_health() {
  local deadline=$((SECONDS + 12))
  while [ $SECONDS -lt "$deadline" ]; do
    if healthy; then return 0; fi
    sleep 0.2
  done
  return 1
}

kill_port() {
  local pids
  pids="$(lsof -ti ":$PORT" -sTCP:LISTEN 2>/dev/null || true)"
  [ -n "$pids" ] || return 0
  log "freeing port $PORT (killing pids: $pids)"
  # shellcheck disable=SC2086
  kill -9 $pids 2>/dev/null || true
  wait_for_port_free || die "port $PORT still held after kill"
  sleep 0.3   # kernel socket-cleanup buffer; Node bind() isn't always SO_REUSEADDR
}

# 1. Already healthy AND serving from this cwd → no-op.
if healthy; then
  if server_cwd_matches; then
    log "server already healthy on port $PORT ($CWD)"
    echo "$URL"
    exit 0
  fi
  log "port $PORT serving from a different cwd — tearing down"
fi

# 2. Free the port unconditionally before starting fresh.
kill_port

# 3. Scaffold: reuse > template copy > network scaffold.
if [ ! -f package.json ]; then
  if [ -d "$TEMPLATE" ] && [ -f "$TEMPLATE/package.json" ]; then
    log "copying template from $TEMPLATE (offline)"
    cp -a "$TEMPLATE"/. .
  else
    log "no template at $TEMPLATE — scaffolding via npm (slow path, needs network)"
    : > "$LOG_FILE"
    if ! run_npm npm create vite@latest . -- --template vanilla <<< "y" >>"$LOG_FILE" 2>&1; then
      die "npm create vite failed or exceeded ${NPM_TIMEOUT}s"
    fi
    if ! run_npm npm install >>"$LOG_FILE" 2>&1; then
      die "npm install failed or exceeded ${NPM_TIMEOUT}s"
    fi
  fi
fi

# 4. Start Vite via the project's declared dev script (uses local binary, no npx resolution).
log "starting Vite on port $PORT from $CWD"
: > "$LOG_FILE"
nohup npm run dev -- --port "$PORT" --host >>"$LOG_FILE" 2>&1 &
disown

if wait_for_health; then
  log "Vite ready on $URL"
  echo "$URL"
  exit 0
fi

die "Vite did not become healthy on port $PORT within 12s"
