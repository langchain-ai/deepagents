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

cat > index.html <<'EOF'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vibe Round</title>
  </head>
  <body>
    <main id="app" aria-label="Ready screen">
      <h1>Ready to vibe code!</h1>
      <p>LangChain x Interrupt 2026</p>
    </main>
    <script type="module" src="/src/main.js"></script>
  </body>
</html>
EOF

cat > src/main.js <<'EOF'
import "./style.css";
EOF

cat > src/style.css <<'EOF'
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  min-height: 100dvh;
  display: grid;
  place-items: center;
  background: #050505;
  color: #f5f5f5;
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

#app {
  text-align: center;
}

h1 {
  margin: 0;
  font-size: 3rem;
  font-weight: 700;
}

p {
  margin: 0.75rem 0 0;
  color: #b5b5b5;
  font-size: 1.25rem;
}
EOF

: > src/counter.js

# Touch a marker so start-server.sh can recognize a valid template.
cat > .vibe-template <<EOF
built-at: $(date -u +%Y-%m-%dT%H:%M:%SZ)
node: $(node --version 2>/dev/null || echo unknown)
npm: $(npm --version 2>/dev/null || echo unknown)
EOF

echo "[setup] template ready at $TEMPLATE"
echo "[setup] start-server.sh will 'cp -a \"\$TEMPLATE\"/. .' into each round dir"
