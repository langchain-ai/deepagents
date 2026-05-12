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
- A recent LangSmith CLI build with both `hub` and `api` commands
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

# Verify your CLI supports the API helper used to enforce source=internal.
langsmith api info POST repos

# If you have multiple binaries, confirm which one is first on PATH.
which -a langsmith
which -a langsmith-cli

# Verify auth env var for sandbox-backed modes.
echo "${LANGSMITH_API_KEY:+set}"
```

If `hub` commands are missing (`No such command 'hub'`), install a LangSmith CLI build that includes Hub operations.

If multiple `langsmith` binaries are installed, put the hub-capable, up-to-date one first on `PATH` before running this example.

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
- `init` pre-creates the repo via `langsmith api` with `source=internal` before the first push.
- v1 is text-only (`md/txt/json/yaml/yml/csv`); binary files are rejected.
- Runtime backend routes `/memories/` to local workspace and uses LangSmith sandbox for `execute`.
- `ingest`, `query`, and `lint` require `LANGSMITH_API_KEY` because they create a sandbox backend.

## Troubleshooting

If the repo opens at `/hub/<owner>/<repo>` but does not appear in the `/context` table:

- Confirm you are viewing the same organization/workspace where the repo was created.
- If multiple `langsmith` binaries are installed, ensure the newer one is first on `PATH`.
- Upgrade the CLI and re-run `--mode init` with a new repo handle if listing behavior is inconsistent.
