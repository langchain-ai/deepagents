# Topic Wiki Runner

A script-first DeepAgents example that builds a per-topic wiki and syncs via `langsmith` CLI.

## Structure

- `topic_wiki_runner.py` - thin CLI entrypoint
- `topic_wiki_helpers.py` - runner helpers and workflow logic
- `README.md` - setup and usage
- `pyproject.toml` - example-local dependency config

## Requirements

- Python 3.11+
- LangSmith CLI installed (`langsmith` or `langsmith-cli`) with `hub` commands available
- Access to `langsmith.sandbox` (via `langsmith[sandbox]`)
- `LANGSMITH_API_KEY` set for `ingest`, `query`, and `lint` modes

## Setup

```bash
cd examples/topic-wiki-runner
uv sync
```

## Preflight checks

```bash
# Verify your CLI supports Hub commands.
langsmith hub --help
# If your binary name is `langsmith-cli`, run:
langsmith-cli hub --help

# Verify auth env var for sandbox-backed modes.
echo "${LANGSMITH_API_KEY:+set}"
```

If `hub` commands are missing (`No such command 'hub'`), install a LangSmith CLI build that includes Hub operations.

## Usage

```bash
# Initialize topic workspace + first Hub push
uv run python topic_wiki_runner.py --mode init --topic "Ada Lovelace"

# Ingest local text sources
uv run python topic_wiki_runner.py \
  --mode ingest \
  --topic "Ada Lovelace" \
  --source ./notes/ada.md \
  --source ./notes/timeline.txt

# Query from wiki
uv run python topic_wiki_runner.py \
  --mode query \
  --topic "Ada Lovelace" \
  --question "What did Ada contribute to computing?"

# Lint wiki consistency/linking
uv run python topic_wiki_runner.py \
  --mode lint \
  --topic "Ada Lovelace"
```

## Notes

- Sync flow is always CLI-driven: `hub init/pull/push`.
- v1 is text-only (`md/txt/json/yaml/yml/csv`); binary files are rejected.
- Runtime backend routes `/memories/` to local workspace and uses LangSmith sandbox for `execute`.
- `ingest`, `query`, and `lint` require `LANGSMITH_API_KEY` because they create a sandbox backend.
