# Deep Agents Code Development Guide

New to the package? Start with [`ARCHITECTURE.md`](./ARCHITECTURE.md) for a high-level map of how the TUI, the `langgraph dev` server subprocess, and the agent graph fit together.

## Contents

- [Quickstart](#quickstart) — get a local checkout running and run the checks CI enforces
- [Local dev installs](#local-dev-installs) — keep an editable `dcode-dev` separate from a released install
- [Debugging](#debugging) — diagnose startup crashes and client-side issues
- [Live CSS development with Textual devtools](#live-css-development-with-textual-devtools) — UI/CSS hot-reload

## Quickstart

This package uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency management. Install it first if you haven't:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the monorepo and bootstrap the `code` package. This creates the virtualenv, installs test dependencies, and installs local git hooks (`pre-commit` + `commit-msg`) so the same checks can run before you push:

```bash
git clone https://github.com/langchain-ai/deepagents.git
cd deepagents/libs/code
make bootstrap
```

If you only want to sync dependencies without installing hooks, run `uv sync --group test` instead.

Run the TUI from `libs/code` in your local checkout:

```bash
uv run deepagents-code
```

`uv run` uses the project environment with the package installed editable, so source changes take effect on the next launch. If you want a persistent `dcode-dev` command that stays separate from a released install, use the local dev install setup below.

### Running the tests and linters

All commands run from `libs/code`. Before opening a PR, run the full local check suite:

```bash
make check
```

This runs linting, import checks, unit tests, and lockfile/version/extras checks.

For targeted checks while iterating:

```bash
# Unit tests (no network)
make test

# A single test file
make test TEST_FILE=tests/unit_tests/test_specific.py

# Integration tests (network permitted)
make integration_test
```

```bash
# Auto-format (ruff format + autofix)
make format

# Lint + type-check (ruff, ty, commands-catalog check)
make lint
```

Run `make help` to see every available target.

## Debugging

Deep Agents Code runs as two processes: the **Textual TUI** you interact with, and a **`langgraph dev` subprocess** that hosts the agent graph. Each writes its own log, and a single switch turns both on:

```bash
cd libs/code
export DEEPAGENTS_CODE_DEBUG=1
uv run deepagents-code
```

| Variable | Effect |
| --- | --- |
| `DEEPAGENTS_CODE_DEBUG` | Master switch. Preserves the server subprocess log on exit (printing its path to stderr) and attaches the client `DEBUG` file handler. Truthy: `1`/`true`/`yes`/`on` (case-insensitive). Falsy: `0`/`false`/`no`/`off`/empty/unset. |
| `DEEPAGENTS_CODE_DEBUG_FILE=<path>` | Overrides the client log path (default `/tmp/deepagents_debug.log`). **Only takes effect when `DEEPAGENTS_CODE_DEBUG` is truthy**; does **not** affect the server subprocess log. |

Then pick the log you need by symptom:

| Symptom | Log you want | Default location |
| --- | --- | --- |
| App crashes on launch (one-line failure banner) | **Server subprocess log** — the real traceback | `$TMPDIR/deepagents_server_log_*.txt` |
| App starts, then misbehaves (UI, model calls, slash commands) | **Client app log** — `deepagents_code` at `DEBUG` | `/tmp/deepagents_debug.log` |

### Startup crash -> server subprocess log

The TUI only surfaces a one-line banner; the actual exception lives in the subprocess's combined stdout/stderr. To get it:

1. **Re-run with debugging on** (see above). On exit, the log is preserved and its path is printed to stderr as `Server log preserved at: ...`. Textual's fullscreen mode can hide that line, but the file is still on disk.
2. **Open the newest log.** On macOS, `tempfile` resolves to `$TMPDIR` (a path under `/var/folders/.../T/`):

   ```bash
   # Newest first
   ls -lt ${TMPDIR:-/tmp}/deepagents_server_log_*.txt | head -5

   # Or tail the latest live while you reproduce the crash
   tail -F "$(ls -t ${TMPDIR:-/tmp}/deepagents_server_log_*.txt | head -1)"
   ```

3. **Search for `Failed to initialize server graph`.** The traceback beneath it names the concrete failure — MCP config validation, sandbox init, model resolution, subagent load, and so on. Everything above that line is uvicorn/lifespan unwinding and can be ignored.

### Client-side issue -> app log

For problems that appear after the app is up, tail the client log in another terminal while reproducing:

```bash
tail -f /tmp/deepagents_debug.log
```

To send it elsewhere, also `export DEEPAGENTS_CODE_DEBUG_FILE=<path>`. The handler appends across runs, so a single file accumulates every session.

### In-app Debug Console (`Ctrl+\`)

Press `Ctrl+\` (or run the hidden `/debug` command) inside a session to toggle a read-only Debug Console overlay. It shows a point-in-time session/runtime snapshot (version, model, thread, cwd, auto-approve, sandbox, MCP servers, token usage, debug-log path) plus a live tail of recent `deepagents_code.*` log records.

The tail is fed by an always-on in-memory ring buffer (`_debug_buffer.install_log_buffer`, installed on the package logger in `__init__.py`), so it works **without** `DEEPAGENTS_CODE_DEBUG` — though enabling that switch raises the captured level to `DEBUG` and adds the file handler above. In the console, `Ctrl+L` clears the on-screen view (the buffer keeps accruing) and `c` copies the visible lines.

## Local dev installs

A *local dev install* gives you a persistent `dcode-dev` command that launches your checkout directly. It lives in a dedicated editable venv under `~/.local/share/dcode-dev`, symlinked into `~/.local/bin/dcode-dev`. It can sit alongside a released `dcode` without interfering:

- `dcode` / `deepagents-code` — the released tool, installed via `curl -LsSf https://langch.in/dcode | bash` (the install script).
- `dcode-dev` — your local checkout.

That lets you compare released behavior against local, and fall back to a known-good build if your checkout breaks. Either way, the dedicated venv keeps the dev binary's dependency experiments out of the repo's locked `uv sync` environment.

### Setup

`~/.local/bin` must be on your `PATH` for the symlink to resolve (`uv tool install` adds its own shim directory automatically, but a hand-rolled symlink does not). Replace `<repo>` with your local checkout path:

```bash
# 1. Create an isolated venv for the dev binary
uv venv ~/.local/share/dcode-dev --python 3.13

# 2. Install your checkout into it, editable
uv pip install --python ~/.local/share/dcode-dev/bin/python -e <repo>/libs/code

# 3. Expose it as `dcode-dev` on your PATH
ln -sf ~/.local/share/dcode-dev/bin/dcode ~/.local/bin/dcode-dev
```

The `--python 3.13` is illustrative — any interpreter satisfying the package's `requires-python` (currently `>=3.11`) works; omit the flag to let `uv` pick.

> **Why `uv venv` + `uv pip install -e` rather than `uv sync` or `uv tool install --editable`?** This builds an isolated venv *outside* the workspace's locked environment, so the dev binary can be re-resolved on demand without disturbing the released tool or the repo's `uv.lock`. (`uv pip` and `uv venv` are first-class `uv` subcommands here, not bare `pip`.)

### Updating

When dependency constraints change in `libs/code/pyproject.toml`, refresh the dev venv:

```bash
uv pip install --python ~/.local/share/dcode-dev/bin/python -e <repo>/libs/code --upgrade
```

### Verifying

Confirm command resolution and editable imports (the `dcode` checks assume the released tool is installed separately, per above):

```bash
which dcode
which dcode-dev
dcode --version
dcode-dev --version
~/.local/share/dcode-dev/bin/python -c 'import deepagents_code; print(deepagents_code.__file__)'
```

## Live CSS development with Textual devtools

After completing the [Quickstart](#quickstart), use Textual's devtools console for CSS hot-reload and live `self.log()` output during development.

Create the dev wrapper script once:

```bash
cat > /tmp/dev_deepagents.py << 'PYEOF'
"""Dev wrapper to run Deep Agents Code with textual devtools."""
import sys
sys.argv = ["deepagents"] + sys.argv[1:]

from deepagents_code.main import cli_main
cli_main()
PYEOF
```

Run both commands from `libs/code`:

**Terminal 1** — devtools console:

```bash
uv run --group test textual console
```

**Terminal 2** — TUI with live reload:

```bash
uv run --group test textual run --dev /tmp/dev_deepagents.py
```

Edit any `.tcss` file and save — changes appear immediately. Any `self.log()` calls in widget code show in the console.

### Console options

- `textual console -v` — verbose mode, shows all events (key presses, mouse, etc.)
- `textual console -x EVENT` — exclude noisy event groups
- `textual console --port 7342` — custom port (pass matching `--port` to `textual run`)

### Why the wrapper script?

`textual run --dev` handles the devtools connection, but it needs to run inside the project's virtualenv to import `deepagents_code`. The wrapper script bridges the gap — `uv run --group test textual run --dev` ensures both `textual-dev` (from the `test` group) and `deepagents_code` are available in the same environment.
