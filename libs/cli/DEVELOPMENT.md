# CLI Development Guide

`deepagents-cli` now contains only the deployment subcommands (`init`, `dev`, `deploy`). For interactive-REPL/headless coding agent (`deepagents-code`/`dcode`) development guidance, see [`libs/code/DEVELOPMENT.md`](../code/DEVELOPMENT.md).

## Local setup

```bash
cd libs/cli && uv sync --group test
```

Run the CLI from a checkout:

```bash
uv run python -m deepagents_cli init my-agent
uv run python -m deepagents_cli dev --config my-agent/deepagents.toml
uv run python -m deepagents_cli deploy --config my-agent/deepagents.toml --dry-run
```

## Tests

```bash
make test       # unit tests (no network)
make lint       # ruff + ty
```

Integration tests in `tests/integration_tests/` exercise the LangSmith Hub seeding path and require `LANGSMITH_API_KEY` to be set.

## `langgraph` subcommand interop

`dev` and `deploy` shell out to the `langgraph` CLI (`langgraph-cli[inmem]` runtime dependency). When debugging dev-server startup failures, run the generated command manually from the build directory printed by `print_bundle_summary`:

```bash
cd /tmp/deepagents-dev-XXXX
langgraph dev --port 2024 --allow-blocking
```

The bundle is self-contained — re-running `langgraph dev` from the build directory reproduces the failure without re-bundling.

## Package layout

- Entry points: the `deepagents` and `deepagents-cli` console scripts dispatch through `deepagents_cli.cli_main`.
- `deepagents_cli/main.py` — argparse wiring and `cli_main` dispatch.
- `deepagents_cli/deploy/` — the entire deploy/dev/init pipeline (`commands.py`, `bundler.py`, `config.py`, `templates.py`, `context_hub.py`, `frontend_dist/`).
- `deepagents_cli/config.py` — slim `_load_dotenv` helper used by deploy/dev.
- `deepagents_cli/model_config.py` — slim `resolve_env_var` helper for the `DEEPAGENTS_CLI_` env-var prefix.
- `deepagents_cli/_version.py` — `__version__` (managed by release-please).

Bare `deepagents` invocations print a deprecation notice pointing at `deepagents-code` and exit non-zero. Everything from the old interactive REPL (Textual widgets, MCP, skills, sandbox bootstrap, slash commands, the drift tests) moved to `libs/code/` in `deepagents-cli==0.1.0`.
