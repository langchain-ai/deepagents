# Wiki Runner

A script-first DeepAgents example that builds a persistent topic wiki and syncs it through `langsmith hub` commands.

## Structure

- `wiki_runner.py` - thin CLI entrypoint
- `wiki_helpers.py` - shared helpers, CLI parsing, and mode orchestration
- `models.py` - shared config/dependency/result dataclasses
- `init.py` - `init` mode workflow and internal-source enforcement
- `ingest.py` - `ingest` mode source expansion + review/apply flow
- `query.py` - `query` mode analysis + optional durable filing flow
- `README.md` - setup and usage
- `pyproject.toml` - example-local dependency config

## Workspace layout

`init` creates this top-level layout in the wiki repo:

- `AGENTS.md` - schema and workflow rules the LLM follows for ingest/query/lint.
- `raw/` - immutable source files dropped in for ingest (articles, notes, datasets).
- `wiki/` - LLM-maintained knowledge pages (entities, concepts, summaries, syntheses).
- `wiki/index.md` - content catalog for wiki navigation; read first during query flows.
- `log.md` - append-only chronological operation log (`ingest`, `query`, `lint` runs).

## Requirements

- Python 3.11+
- LangSmith CLI installed as `langsmith` with `hub` commands available
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

# Verify init exposes an internal source option.
langsmith hub init --help

# Verify auth env var for sandbox-backed modes.
echo "${LANGSMITH_API_KEY:+set}"
```

## Usage

```bash
# Initialize workspace + first Hub push
uv run python wiki_runner.py \
  --mode init \
  --repo "ada-lovelace-wiki" \
  --description "Persistent research wiki about Ada Lovelace"

# Initialize with explicit owner
uv run python wiki_runner.py \
  --mode init \
  --repo "ada-lovelace-wiki" \
  --description "Persistent research wiki about Ada Lovelace" \
  --owner "acme"

# Default ingest (no approval)
uv run python wiki_runner.py \
  --mode ingest \
  --repo "ada-lovelace-wiki" \
  --source ./notes/ada.md \
  --source ./speeches/

# Optional review + confirmation before apply
uv run python wiki_runner.py \
  --mode ingest \
  --repo "ada-lovelace-wiki" \
  --source ./notes/timeline.txt \
  --source ./speeches/ \
  --review

# Query from wiki
uv run python wiki_runner.py \
  --mode query \
  --repo "ada-lovelace-wiki" \
  --question "What did Ada contribute to computing?"

# Lint wiki consistency/linking
uv run python wiki_runner.py \
  --mode lint \
  --repo "ada-lovelace-wiki"
```

## Ingest workflow

`ingest` applies directly by default.

If you pass `--review`, ingest becomes a two-phase, operator-in-the-loop flow:

1. Review phase (read-only): the model reads staged source files and returns key takeaways, proposed wiki updates, contradictions, and index/log changes.
2. Apply phase (write): after your confirmation, the model writes source summary updates, concept/entity updates, index updates, and a log entry.
3. If you decline confirmation, ingest exits without applying wiki changes.

Batch ingest is the default. A single run can process multiple files and directories.

## Query workflow

`query` runs in two phases automatically:

1. Analysis phase (read-only): the model reads `wiki/index.md`, then (when helpful) checks prior `wiki/query/*.md` pages first for discovery/routing, expands into canonical wiki pages for grounding, answers with citations, and decides whether the result should be filed for future reuse. Query pages are treated as routing hints rather than primary evidence.
2. Filing phase (write, conditional): if the answer is durable, the runner files it into `wiki/query/<question-slug>.md`, refreshes `wiki/index.md`, appends a query entry to `log.md`, and pushes.
