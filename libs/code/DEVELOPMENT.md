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

Clone the monorepo and sync the `code` package with its test dependencies (this creates the virtualenv and installs everything you need to run and test the app):

```bash
git clone https://github.com/langchain-ai/deepagents.git
cd deepagents/libs/code
uv sync --group test
```

Run the TUI from `libs/code` in your local checkout:

```bash
uv run deepagents-code
```

`uv run` uses the project environment with the package installed editable, so source changes take effect on the next launch. If you want a persistent `dcode-dev` command that stays separate from a released install, use the local dev install setup below.

### Running the tests and linters

All commands run from `libs/code`. These mirror what CI enforces, so run them before opening a PR.

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

The app runs a `langgraph dev` subprocess when it starts. When the subprocess crashes during startup, the TUI shows a one-line failure banner; the actual exception lives in the subprocess's stdout/stderr, which is captured to a temp file.

### Environment variables

| Variable | Effect |
| --- | --- |
| `DEEPAGENTS_CODE_DEBUG=1` | Preserves the server subprocess log on shutdown and prints its path to stderr. Without this, the log is deleted when the process stops. Also enables the app-process file handler below. Accepts `1`/`true`/`yes`/`on` (case-insensitive) as enabled; `0`/`false`/`no`/`off`/empty/unset as disabled. |
| `DEEPAGENTS_CODE_DEBUG_FILE=<path>` | Overrides the default path (`/tmp/deepagents_debug.log`) for the app-process file handler, which attaches at `DEBUG` level to the `deepagents_code` package logger. **Only takes effect when `DEEPAGENTS_CODE_DEBUG` is truthy.** Useful for diagnosing client-side app issues; does **not** capture the server subprocess. |

`DEEPAGENTS_CODE_DEBUG` is what you want for startup crashes (graph init, MCP config, sandbox): the preserved subprocess log contains the real traceback. The optional `DEEPAGENTS_CODE_DEBUG_FILE` override is for post-startup client-side debugging.

To capture client-side logs while reproducing an issue:

```bash
cd libs/code
DEEPAGENTS_CODE_DEBUG=1 uv run deepagents-code
```

Then in another terminal:

```bash
tail -f /tmp/deepagents_debug.log
```

### Finding the server subprocess log

On macOS, `tempfile` resolves to `$TMPDIR` (a path under `/var/folders/.../T/`). Each `ServerProcess` writes its combined stdout+stderr to a file matching `deepagents_server_log_*.txt`:

```bash
# Newest first
ls -lt ${TMPDIR:-/tmp}/deepagents_server_log_*.txt | head -5

# Tail the latest while reproducing the crash
tail -F "$(ls -t ${TMPDIR:-/tmp}/deepagents_server_log_*.txt | head -1)"
```

The interesting line is `Failed to initialize server graph: <exc>` followed by a traceback — everything above that is uvicorn/lifespan unwinding.

### Triage flow for a startup crash

1. **Rerun with `DEEPAGENTS_CODE_DEBUG=1`.** The log is preserved and a "Server log preserved at: ..." line is printed to stderr. Textual's fullscreen mode can hide that line, but the file itself is still on disk.
2. **Locate the log** via the `ls` command above. Open it in your editor.
3. **Search for `Failed to initialize server graph`.** The stack trace beneath names the concrete failure point (MCP config validation, sandbox init, model resolution, subagent load, etc.).
4. **Pre-flight validators run in the app process** for the common failure modes (e.g., `--mcp-config` is validated in `start_server_and_get_agent` before the subprocess spawns). When the banner shows `MCPConfigError: <path>: <reason>`, the subprocess never started — fix the file and retry.

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
