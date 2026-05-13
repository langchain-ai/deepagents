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
      <h1>site will load here</h1>
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
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
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
  font-size: clamp(1.4rem, 3.2vw, 3.4rem);
  font-weight: 800;
  letter-spacing: 0;
  line-height: 1;
  text-transform: uppercase;
}

@media (max-width: 640px) {
  h1 {
    font-size: clamp(1.2rem, 8vw, 2.4rem);
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
