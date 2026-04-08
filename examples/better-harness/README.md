# better-harness

Vendored copy of the `better-harness` research artifact, included here as a
standalone example. This folder is intentionally self-contained:

- its own `pyproject.toml`
- its own `uv.lock`
- its own tests
- its own `better_harness` package
- a runnable Deep Agents example config in [`examples/deepagents_example.toml`](examples/deepagents_example.toml)

This means the example does not depend on a separate `better-harness` checkout.

## What it is

`better-harness` runs one outer Deep Agent against another agent harness.

You define:

- editable harness surfaces
- explicit `train`, `holdout`, and optional `scorecard` eval cases
- the outer-agent model

The loop:

1. runs a baseline
2. builds a proposer workspace for the outer agent
3. lets the outer agent edit the allowed surfaces
4. tests the edited inner agent on `train` and `holdout`
5. keeps the edit only if the combined pass count improves

## Quick start

From this directory:

```bash
cd examples/better-harness
uv sync --extra dev
```

Validate the bundled Deep Agents example:

```bash
uv run better-harness validate examples/deepagents_example.toml
```

Run the bundled Deep Agents example:

```bash
export ANTHROPIC_API_KEY=...
uv run better-harness run examples/deepagents_example.toml \
  --output-dir runs/deepagents-better-harness
```

Run the local tests for the harness artifact itself:

```bash
uv run pytest
```

## Deep Agents example

The bundled config in [`examples/deepagents_example.toml`](examples/deepagents_example.toml):

- points at this local Deep Agents checkout
- targets the real `deepagents.graph:BASE_AGENT_PROMPT` surface
- uses the focused `tool_selection` and `followup_quality` eval slices

It is prompt-only by default because that is the real, loaded surface that was
used in the focused harness work.

If you want to optimize tools, skills, or middleware too, add those as real
`workspace_file` surfaces that your agent actually imports.

For middleware in particular, you usually need two surfaces:

1. the middleware implementation file
2. the file that wires that middleware into `create_deep_agent(..., middleware=[...])`

Useful docs:

- [Deep Agents repo](https://github.com/langchain-ai/deepagents)
- [Custom middleware in LangChain](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- [Middleware in Deep Agents customization](https://docs.langchain.com/oss/python/deepagents/customization#middleware)
