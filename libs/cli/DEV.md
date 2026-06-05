# CLI Development Guide

`deepagents-cli` now contains only the deployment subcommands (`init`, `dev`,
`deploy`). For interactive-REPL development guidance, see
[`libs/code/DEV.md`](../code/DEV.md).

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

Integration tests in `tests/integration_tests/` exercise the LangSmith Hub
seeding path and require `LANGSMITH_API_KEY` to be set.

## `langgraph` subcommand interop

`dev` and `deploy` shell out to the `langgraph` CLI (`langgraph-cli[inmem]`
runtime dependency). When debugging dev-server startup failures, run the
generated command manually from the build directory printed by
`print_bundle_summary`:

```bash
cd /tmp/deepagents-dev-XXXX
langgraph dev --port 2024 --allow-blocking
```

The bundle is self-contained — re-running `langgraph dev` from the build
directory reproduces the failure without re-bundling.
