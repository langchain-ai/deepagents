# CLI Development Guide

## Live CSS development with Textual devtools

Textual's devtools console enables CSS hot-reload and live `self.log()` output during development.

### Prerequisites

Sync the `test` dependency group (includes `textual-dev`):

```bash
cd libs/cli && uv sync --group test
```

Create the dev wrapper script (one-time):

```bash
cat > /tmp/dev_deepagents.py << 'PYEOF'
"""Dev wrapper to run Deep Agents CLI with textual devtools."""
import sys
sys.argv = ["deepagents"] + sys.argv[1:]

from deepagents_cli.main import cli_main
cli_main()
PYEOF
```

### Running

**Terminal 1** — devtools console:

```bash
cd libs/cli && uv run --group test textual console
```

**Terminal 2** — CLI with live reload:

```bash
cd libs/cli && uv run --group test textual run --dev /tmp/dev_deepagents.py
```

Edit any `.tcss` file and save — changes appear immediately. Any `self.log()` calls in widget code show in the console.

### Console options

- `textual console -v` — verbose mode, shows all events (key presses, mouse, etc.)
- `textual console -x EVENT` — exclude noisy event groups
- `textual console --port 7342` — custom port (pass matching `--port` to `textual run`)

### Why the wrapper script?

`textual run --dev` handles the devtools connection, but it needs to run inside the project's virtualenv to import `deepagents_cli`. The wrapper script bridges the gap — `uv run --group test textual run --dev` ensures both `textual-dev` (from the `test` group) and `deepagents_cli` are available in the same environment.

## Debugging

The CLI runs a `langgraph dev` subprocess for every interactive session. When the subprocess crashes during startup, the TUI shows a one-line failure banner; the actual exception lives in the subprocess's stdout/stderr, which is captured to a temp file.

### Environment variables

| Variable | Effect |
| --- | --- |
| `DEEPAGENTS_CLI_DEBUG=1` | Preserves the server subprocess log on shutdown and prints its path to stderr. Without this, the log is deleted when the process stops. Also enables the CLI-process file handler below. Accepts `1`/`true`/`yes`/`on` (case-insensitive) as enabled; `0`/`false`/`no`/`off`/empty/unset as disabled. |
| `DEEPAGENTS_CLI_DEBUG_FILE=<path>` | Overrides the default path (`/tmp/deepagents_debug.log`) for the CLI-process file handler, which attaches at `DEBUG` level to `textual_adapter` and `remote_client`. **Only takes effect when `DEEPAGENTS_CLI_DEBUG` is truthy.** Useful for diagnosing streaming/client-side issues; does **not** capture the server subprocess. |

`DEEPAGENTS_CLI_DEBUG` is what you want for startup crashes (graph init, MCP config, sandbox): the preserved subprocess log contains the real traceback. The optional `DEEPAGENTS_CLI_DEBUG_FILE` override is for post-startup client-side debugging.

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

1. **Rerun with `DEEPAGENTS_CLI_DEBUG=1`.** The log is preserved and a "Server log preserved at: ..." line is printed to stderr. Textual's fullscreen mode can hide that line, but the file itself is still on disk.
2. **Locate the log** via the `ls` command above. Open it in your editor.
3. **Search for `Failed to initialize server graph`.** The stack trace beneath names the concrete failure point (MCP config validation, sandbox init, model resolution, subagent load, etc.).
4. **Pre-flight validators run in the CLI process** for the common failure modes (e.g., `--mcp-config` is validated in `start_server_and_get_agent` before the subprocess spawns). When the banner shows `MCPConfigError: <path>: <reason>`, the subprocess never started — fix the file and retry.

### Common startup failure patterns

- **`MCPConfigError: Invalid MCP config at <path>: ...`** — malformed `--mcp-config`. The pre-flight wraps the underlying `ValueError`/`TypeError` with the offending path. See `_preflight_validate_mcp_config` in `server_manager.py`.
- **`Server 'X' missing required 'command' field`** (from a discovered project `.mcp.json`, not `--mcp-config`) — an stdio server config without `command`. For remote servers, just use `{"url": "..."}`; transport is inferred as `http` when no `type`/`transport` is present.
- **Uncaught exception inside a bare `sys.exit(1)`** — usually means the surrounding `make_graph()` raised. Look one traceback up in the subprocess log for the real cause.
