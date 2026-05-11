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
    <main id="app" aria-label="Ready screen">
      <h1>Ready to vibe code!</h1>
      <p>LangChain x Interrupt 2026</p>
    </main>
    <script type="module" src="/src/main.js"></script>
  </body>
</html>
"""
BLANK_MAIN_JS = """import "./style.css";

localStorage.clear();
sessionStorage.clear();
"""
BLANK_STYLE_CSS = """* {
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
