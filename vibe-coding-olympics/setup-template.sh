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
    <main id="app" aria-label="Site placeholder">
      <h1>site will load here</h1>
    </main>
    <script type="module" src="/src/main.js"></script>
  </body>
</html>
EOF

cat > src/main.js <<'EOF'
import "./style.css";
EOF

cat > src/style.css <<'EOF'
:root {
  color-scheme: dark;
  --paper: #ffffff;
  --blue-a: #868cfe;
  --blue-b: #82c9fe;
  --pink-a: #ed92ff;
  --pink-b: #d5c3f7;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  min-height: 100dvh;
  overflow: hidden;
  background:
    linear-gradient(180deg, var(--blue-a) 0%, var(--blue-b) 20%, var(--paper) 50%),
    linear-gradient(180deg, var(--pink-a) 0%, var(--pink-b) 20%, var(--paper) 50%);
  background-position: left top, right top;
  background-size: 50% 100%, 50% 100%;
  background-repeat: no-repeat;
  color: #000000;
  font-family: "Aeonik Mono", "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}

#app {
  min-height: 100vh;
  min-height: 100dvh;
  display: grid;
  justify-items: center;
  align-content: start;
  padding: clamp(0.75rem, 2.2vh, 1.5rem) 1rem 0;
  text-align: center;
}

h1 {
  margin: 0;
  padding: 0.95rem 1.35rem;
  border: 2px solid #000000;
  background:
    linear-gradient(90deg, rgba(216, 239, 255, 0.94), rgba(245, 216, 255, 0.94)),
    #ffffff;
  font-size: clamp(1.2rem, 2.6vw, 2.7rem);
  font-weight: 700;
  letter-spacing: 0;
  line-height: 1;
}

@media (max-width: 640px) {
  h1 {
    font-size: clamp(1.2rem, 8vw, 2.4rem);
  }
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
