#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-3001}"
PROMPT="${2:-}"

DIR=$(mktemp -d -t vibe-player-XXXX)
echo "Player dir: $DIR"
echo "Port: $PORT"

cd "$DIR"

if [ -n "$PROMPT" ]; then
  VIBE_PORT="$PORT" deepagents -m "/skill:web-vibe $PROMPT"
else
  VIBE_PORT="$PORT" deepagents
fi
