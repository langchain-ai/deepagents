#!/usr/bin/env bash
# Event-ops script — NOT invoked by the agent.
#
# Builds a pre-scaffolded Vite project at $VIBE_TEMPLATE (default
# ~/.vibe-template/) once per player laptop, so start-server.sh can
# bootstrap a round in <1s with no network dependency.
#
# Run this ONCE per laptop during event setup. Re-run to rebuild after
# Vite/Node upgrades.
#
# Env:
#   VIBE_TEMPLATE   Target dir (default: ~/.vibe-template)

set -euo pipefail

TEMPLATE="${VIBE_TEMPLATE:-$HOME/.vibe-template}"

echo "[setup] building Vite template at $TEMPLATE"

# Clean previous template so we don't merge stale state.
rm -rf "$TEMPLATE"
mkdir -p "$TEMPLATE"
cd "$TEMPLATE"

npm create vite@latest . -- --template vanilla <<< "y"
npm install

# Touch a marker so start-server.sh can recognize a valid template.
cat > .vibe-template <<EOF
built-at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
node: $(node --version 2>/dev/null || echo unknown)
npm: $(npm --version 2>/dev/null || echo unknown)
EOF

echo "[setup] template ready at $TEMPLATE"
echo "[setup] start-server.sh will 'cp -a \"\$TEMPLATE\"/. .' into each round dir"
