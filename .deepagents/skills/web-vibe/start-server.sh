#!/usr/bin/env bash
# Idempotent dev-server startup for the web-vibe skill.
#
# Ensures a Vite project exists in the current working directory and a dev
# server is serving it on $VIBE_PORT (default 5173). Safe to re-run — if
# the port is already healthy, it exits immediately without side effects.
#
# Usage:
#   bash /skills/web-vibe/start-server.sh
#
# Env:
#   VIBE_PORT  Port to serve on (default: 5173)
#   VIBE_LOG   Log file path (default: /tmp/vite.log)
#
# Exits 0 with the server URL printed to stdout on success.
# Exits 1 with tail of log on stderr on failure.

set -euo pipefail

PORT="${VIBE_PORT:-5173}"
LOG_FILE="${VIBE_LOG:-/tmp/vite.log}"
URL="http://localhost:$PORT"

log() { printf '[web-vibe] %s\n' "$*" >&2; }

healthy() {
  curl -fsS -o /dev/null -m 1 "$URL"
}

wait_for_health() {
  # ~10s total: 50 * 0.2s
  for _ in $(seq 1 50); do
    if healthy; then return 0; fi
    sleep 0.2
  done
  return 1
}

# 1. Already healthy → no-op.
if healthy; then
  log "server already healthy on port $PORT"
  echo "$URL"
  exit 0
fi

# 2. Port occupied by a dead/unhealthy process → free it.
if command -v lsof >/dev/null 2>&1 && lsof -ti ":$PORT" >/dev/null 2>&1; then
  log "port $PORT occupied but unhealthy — killing stale process"
  lsof -ti ":$PORT" | xargs kill -9 2>/dev/null || true
  sleep 0.3
fi

# 3. Scaffold Vite project if this directory has no package.json.
if [ ! -f package.json ]; then
  log "no package.json — scaffolding Vite vanilla template in $(pwd)"
  : > "$LOG_FILE"
  npm create vite@latest . -- --template vanilla <<< "y" >>"$LOG_FILE" 2>&1
  npm install >>"$LOG_FILE" 2>&1
fi

# 4. Start Vite fully detached.
log "starting Vite on port $PORT"
nohup npx vite --port "$PORT" --host >>"$LOG_FILE" 2>&1 & disown

if wait_for_health; then
  log "Vite ready on $URL"
  echo "$URL"
  exit 0
fi

# 5. Fallback: live-server (no build step, full-page reload).
log "Vite did not become healthy — falling back to live-server"
tail -20 "$LOG_FILE" >&2 || true

nohup npx -y live-server --port="$PORT" --no-browser >>"$LOG_FILE" 2>&1 & disown

if wait_for_health; then
  log "live-server ready on $URL"
  echo "$URL"
  exit 0
fi

log "ERROR: no server came up on port $PORT"
tail -40 "$LOG_FILE" >&2 || true
exit 1
