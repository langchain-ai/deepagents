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
