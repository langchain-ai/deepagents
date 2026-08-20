# CLI Development Guide

`deepagents-cli` now contains only the deployment subcommands (`init`, `deploy`, `agents`, `mcp-servers`). For interactive-REPL/headless coding agent (`deepagents-code`/`dcode`) development guidance, see [`libs/code/DEVELOPMENT.md`](../code/DEVELOPMENT.md).

## Local setup

```bash
cd libs/cli && uv sync --group test
```

Run the CLI from a checkout:

```bash
uv run python -m deepagents_cli init my-agent
uv run python -m deepagents_cli deploy --config my-agent/deepagents.toml --dry-run
```

## Tests

```bash
make test       # unit tests (no network)
make lint       # ruff + ty
```

Integration tests in `tests/integration_tests/` exercise the LangSmith Hub seeding path and require `LANGSMITH_API_KEY` to be set.

## Package layout

- Entry points: the `deepagents` and `deepagents-cli` console scripts dispatch through `deepagents_cli.cli_main`.
- `deepagents_cli/main.py` — argparse wiring and `cli_main` dispatch.
- `deepagents_cli/deploy/` — the whole deployment pipeline. Read the directory for the current module list; `commands.py` holds the subparsers and command entry points, and `api_client.py` talks to the managed Deep Agents API.
- `deepagents_cli/config.py` — slim `_load_dotenv` helper used by the deploy commands.
- `deepagents_cli/model_config.py` — slim `resolve_env_var` helper for the `DEEPAGENTS_CLI_` env-var prefix.
- `deepagents_cli/_version.py` — `__version__` (managed by release-please).

Bare `deepagents` invocations print a deprecation notice pointing at `deepagents-code` and exit non-zero. Everything from the old interactive REPL (Textual widgets, MCP, skills, sandbox bootstrap, slash commands, the drift tests) moved to `libs/code/` in `deepagents-cli==0.1.0`.
