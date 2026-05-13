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
      <section class="top-band" aria-labelledby="ready-title">
        <div class="tag">LangChain x Interrupt 2026</div>
        <h1 id="ready-title">Site will be built here...</h1>
        <p>Awaiting the next live prompt</p>
        <div class="timer">00:00</div>
      </section>
      <section class="board" aria-hidden="true">
        <div class="panel panel-a">
          <div class="panel-head"></div>
        </div>
        <div class="panel panel-b">
          <div class="panel-head"></div>
        </div>
        <div class="strip strip-left"></div>
        <div class="strip strip-right"></div>
        <div class="arrow arrow-left"></div>
        <div class="arrow arrow-right"></div>
      </section>
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
  --ink: #000000;
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
    linear-gradient(90deg, transparent calc(50% - 1px), var(--ink) calc(50% - 1px), var(--ink) calc(50% + 1px), transparent calc(50% + 1px)),
    linear-gradient(180deg, transparent 10.7vh, var(--ink) 10.7vh, var(--ink) calc(10.7vh + 1px), transparent calc(10.7vh + 1px)),
    linear-gradient(180deg, transparent 71.3vh, var(--ink) 71.3vh, var(--ink) calc(71.3vh + 1px), transparent calc(71.3vh + 1px)),
    linear-gradient(180deg, var(--blue-a) 0%, var(--blue-b) 20%, var(--paper) 50%),
    linear-gradient(180deg, var(--pink-a) 0%, var(--pink-b) 20%, var(--paper) 50%);
  background-position: 0 0, 0 0, 0 0, left top, right top;
  background-size: 100% 100%, 100% 100%, 100% 100%, 50% 100%, 50% 100%;
  background-repeat: no-repeat;
  color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

#app {
  position: relative;
  min-height: 100vh;
  min-height: 100dvh;
  isolation: isolate;
}

#app::before,
#app::after {
  position: absolute;
  top: 2.9vh;
  height: 3rem;
  background: var(--ink);
  clip-path: polygon(0.75rem 0, calc(100% - 0.75rem) 0, 100% 50%, calc(100% - 0.75rem) 100%, 0.75rem 100%, 0 50%);
  content: "";
  z-index: 0;
}

#app::before {
  left: 1.75vw;
  width: 28vw;
}

#app::after {
  right: 1.75vw;
  width: 5.6vw;
}

.top-band {
  position: relative;
  z-index: 2;
  display: grid;
  min-height: 33.333vh;
  padding: clamp(1rem, 3vw, 2.5rem) clamp(1rem, 4vw, 3.5rem) 0;
  justify-items: center;
  align-content: start;
  gap: clamp(0.45rem, 1.2vh, 0.85rem);
  text-align: center;
}

.tag {
  width: min(92vw, 33rem);
  padding: 0.72rem 1.4rem;
  background: var(--ink);
  clip-path: polygon(1rem 0, calc(100% - 1rem) 0, 100% 50%, calc(100% - 1rem) 100%, 1rem 100%, 0 50%);
  color: var(--paper);
  font-size: clamp(0.72rem, 1.5vw, 0.98rem);
  font-weight: 800;
  letter-spacing: 0.1em;
  line-height: 1;
  text-transform: uppercase;
}

h1 {
  width: min(92vw, 55rem);
  margin: 0.15rem 0 0;
  font-size: clamp(2.2rem, 7.8vw, 6.2rem);
  font-weight: 950;
  letter-spacing: 0;
  line-height: 0.9;
  text-transform: uppercase;
  text-wrap: balance;
}

.top-band p {
  margin: 0;
  font-size: clamp(0.75rem, 1.6vw, 1.02rem);
  font-weight: 700;
  text-transform: uppercase;
}

.timer {
  position: absolute;
  top: 3vh;
  right: clamp(1rem, 3.2vw, 2.3rem);
  min-width: 5.6rem;
  padding: 0.7rem 0.85rem;
  background: var(--ink);
  clip-path: polygon(0.8rem 0, calc(100% - 0.8rem) 0, 100% 50%, calc(100% - 0.8rem) 100%, 0.8rem 100%, 0 50%);
  color: var(--paper);
  font-size: clamp(0.82rem, 1.4vw, 1rem);
  font-weight: 900;
  line-height: 1;
}

.board {
  position: absolute;
  inset: 33.333vh 0 0;
  z-index: 1;
  border-top: 1px solid var(--ink);
}

.board::before,
.board::after {
  position: absolute;
  left: 0;
  right: 0;
  height: 1px;
  background: var(--ink);
  content: "";
}

.board::before {
  top: 18vh;
}

.board::after {
  bottom: 10.5vh;
}

.panel {
  position: absolute;
  top: 5.7vh;
  width: calc(50vw - 3.6vw);
  height: 31.5vh;
  border: 1px solid var(--ink);
  background: var(--ink);
}

.panel-a {
  left: 1.75vw;
}

.panel-b {
  right: 1.75vw;
}

.panel-head {
  position: absolute;
  inset: 0 0 auto;
  height: 3.7vh;
  min-height: 1.55rem;
  background: linear-gradient(180deg, var(--blue-a), var(--blue-b) 55%, var(--paper));
}

.panel-b .panel-head {
  background: linear-gradient(180deg, var(--pink-a), var(--pink-b) 55%, var(--paper));
}

.strip {
  position: absolute;
  bottom: 4.3vh;
  width: calc(50vw - 3.6vw);
  height: 10vh;
  border: 1px solid var(--ink);
}

.strip-left {
  left: 1.75vw;
}

.strip-right {
  right: 1.75vw;
}

.arrow {
  position: absolute;
  right: 1.75vw;
  bottom: 4.3vh;
  width: 2.65rem;
  height: 2.65rem;
  background: var(--ink);
}

.arrow::before {
  position: absolute;
  inset: 0.78rem 1rem;
  border-top: 2px solid var(--paper);
  border-right: 2px solid var(--paper);
  content: "";
  transform: rotate(45deg);
}

.arrow-left {
  left: calc(50vw - 5.25vw);
  right: auto;
}

.arrow-right {
  right: 1.75vw;
}

@media (max-width: 640px) {
  #app::before {
    width: 35vw;
  }

  #app::after,
  .timer {
    display: none;
  }

  h1 {
    font-size: clamp(2.05rem, 12vw, 4.4rem);
  }

  .top-band {
    padding-top: 1.1rem;
  }

  .panel {
    width: calc(100vw - 2rem);
    height: 21vh;
  }

  .panel-a {
    left: 1rem;
    right: 1rem;
  }

  .panel-b {
    top: 30vh;
    left: 1rem;
    right: 1rem;
  }

  .strip {
    display: none;
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
