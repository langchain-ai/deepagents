"""Utilities for resetting a web-vibe round project."""

from __future__ import annotations

from pathlib import Path


BLANK_INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Vibe Round</title>
  </head>
  <body>
    <main id="app" aria-label="Site placeholder">
      <section class="stage" aria-labelledby="ready-title">
        <div class="signal" aria-hidden="true">
          <span></span>
          <span></span>
          <span></span>
        </div>
        <p class="eyebrow">LangChain x Interrupt 2026</p>
        <h1 id="ready-title">Site will be built here...</h1>
        <p class="lede">A fresh Vite canvas is online and waiting for the next prompt.</p>
        <div class="terminal" aria-hidden="true">
          <div class="terminal-bar">
            <span></span>
            <span></span>
            <span></span>
          </div>
          <div class="terminal-body">
            <p><b>$</b> create-vibe-site --round live</p>
            <p><b>></b> compiling ideas</p>
            <p><b>></b> preview ready at localhost</p>
          </div>
        </div>
        <div class="status" aria-label="Ready status">
          <span class="pulse"></span>
          ready for build
        </div>
      </section>
    </main>
    <script type="module" src="/src/main.js"></script>
  </body>
</html>
"""
BLANK_MAIN_JS = """import "./style.css";

localStorage.clear();
sessionStorage.clear();
"""
BLANK_STYLE_CSS = """:root {
  color-scheme: dark;
  --bg: #070807;
  --ink: #f7fff5;
  --muted: #a7b6aa;
  --line: rgba(247, 255, 245, 0.16);
  --green: #2dfc89;
  --cyan: #43d9ff;
  --yellow: #ffd166;
  --red: #ff5c7a;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-height: 100vh;
  min-height: 100dvh;
  overflow-x: hidden;
  background:
    linear-gradient(rgba(45, 252, 137, 0.08) 1px, transparent 1px),
    linear-gradient(90deg, rgba(67, 217, 255, 0.08) 1px, transparent 1px),
    radial-gradient(circle at 20% 20%, rgba(45, 252, 137, 0.22), transparent 32rem),
    radial-gradient(circle at 82% 72%, rgba(255, 92, 122, 0.18), transparent 30rem),
    #070807;
  background-size: 48px 48px, 48px 48px, auto, auto, auto;
  color: var(--ink);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

#app {
  position: relative;
  display: grid;
  min-height: 100vh;
  min-height: 100dvh;
  place-items: center;
  padding: 2rem;
  text-align: center;
}

#app::before,
#app::after {
  position: fixed;
  inset: auto;
  width: 24rem;
  height: 24rem;
  border: 1px solid var(--line);
  content: "";
  transform: rotate(18deg);
}

#app::before {
  top: -9rem;
  left: -7rem;
  border-color: rgba(45, 252, 137, 0.32);
}

#app::after {
  right: -8rem;
  bottom: -10rem;
  border-color: rgba(67, 217, 255, 0.28);
}

.stage {
  position: relative;
  z-index: 1;
  display: grid;
  width: min(100%, 58rem);
  justify-items: center;
  gap: 1.25rem;
}

.signal {
  position: relative;
  width: min(52vw, 22rem);
  aspect-ratio: 1;
  margin-bottom: -12rem;
  pointer-events: none;
}

.signal span {
  position: absolute;
  inset: 0;
  border: 1px solid rgba(247, 255, 245, 0.18);
  animation: orbit 14s linear infinite;
}

.signal span:nth-child(1) {
  border-color: rgba(45, 252, 137, 0.5);
  border-radius: 42% 58% 46% 54%;
}

.signal span:nth-child(2) {
  border-color: rgba(67, 217, 255, 0.4);
  border-radius: 58% 42% 55% 45%;
  animation-duration: 18s;
  animation-direction: reverse;
}

.signal span:nth-child(3) {
  inset: 16%;
  border-color: rgba(255, 209, 102, 0.34);
  border-radius: 50%;
  animation-duration: 10s;
}

.eyebrow,
.status {
  border: 1px solid var(--line);
  background: rgba(7, 8, 7, 0.62);
  letter-spacing: 0.14em;
  text-transform: uppercase;
}

.eyebrow {
  margin: 0;
  padding: 0.55rem 0.8rem;
  color: var(--green);
  font-size: 0.76rem;
  font-weight: 800;
}

h1 {
  margin: 0;
  max-width: 14ch;
  color: var(--ink);
  font-size: clamp(3.5rem, 10vw, 8rem);
  font-weight: 900;
  line-height: 0.86;
  text-wrap: balance;
  text-shadow:
    0 0 2rem rgba(45, 252, 137, 0.34),
    0.06em 0.04em 0 rgba(255, 92, 122, 0.34),
    -0.04em -0.03em 0 rgba(67, 217, 255, 0.32);
}

.lede {
  max-width: 34rem;
  margin: 0;
  color: var(--muted);
  font-size: clamp(1rem, 2.4vw, 1.35rem);
  line-height: 1.5;
}

.terminal {
  width: min(100%, 34rem);
  overflow: hidden;
  border: 1px solid var(--line);
  background: rgba(7, 8, 7, 0.76);
  box-shadow: 0 1.5rem 4rem rgba(0, 0, 0, 0.32);
  text-align: left;
}

.terminal-bar {
  display: flex;
  gap: 0.45rem;
  padding: 0.75rem;
  border-bottom: 1px solid var(--line);
}

.terminal-bar span {
  width: 0.7rem;
  height: 0.7rem;
  border-radius: 999px;
  background: var(--red);
}

.terminal-bar span:nth-child(2) {
  background: var(--yellow);
}

.terminal-bar span:nth-child(3) {
  background: var(--green);
}

.terminal-body {
  display: grid;
  gap: 0.45rem;
  padding: 1rem;
  color: #d7eadb;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: clamp(0.78rem, 2vw, 0.98rem);
}

.terminal-body p {
  margin: 0;
}

.terminal-body b {
  color: var(--green);
}

.status {
  display: inline-flex;
  align-items: center;
  gap: 0.55rem;
  padding: 0.55rem 0.8rem;
  color: #d7eadb;
  font-size: 0.72rem;
  font-weight: 800;
}

.pulse {
  width: 0.58rem;
  height: 0.58rem;
  border-radius: 999px;
  background: var(--green);
  box-shadow: 0 0 0 0 rgba(45, 252, 137, 0.6);
  animation: ping 1.8s ease-out infinite;
}

@keyframes orbit {
  to {
    transform: rotate(1turn);
  }
}

@keyframes ping {
  70%,
  100% {
    box-shadow: 0 0 0 0.7rem rgba(45, 252, 137, 0);
  }
}

@media (max-width: 640px) {
  #app {
    padding: 1rem;
  }

  .signal {
    width: min(74vw, 20rem);
    margin-bottom: -10rem;
  }

  h1 {
    font-size: clamp(3.25rem, 18vw, 5.5rem);
  }
}
"""


def clear_round_project(path: Path) -> bool:
    """Blank the user-visible app files in a Vite round project.

    Args:
        path: The round project directory created by `play.sh`.

    Returns:
        `True` when the project files were reset, otherwise `False`.
    """
    if not path.exists() or not path.is_dir():
        return False
    if not (path / "package.json").is_file():
        return False

    src = path / "src"
    src.mkdir(exist_ok=True)
    (path / "index.html").write_text(BLANK_INDEX_HTML, encoding="utf-8")
    (src / "main.js").write_text(BLANK_MAIN_JS, encoding="utf-8")
    (src / "style.css").write_text(BLANK_STYLE_CSS, encoding="utf-8")
    (src / "counter.js").write_text("", encoding="utf-8")
    return True
