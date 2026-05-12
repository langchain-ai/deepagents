# Wiki Runner

A script-first DeepAgents example that builds a persistent topic wiki and syncs it through `langsmith hub` commands.

## Structure

- `wiki_runner.py` - thin CLI entrypoint
- `wiki_helpers.py` - shared helpers, CLI parsing, and mode orchestration
- `index.py` - `wiki/index.md` catalog builder and categorization logic
- `log.py` - `log.md` append-only timeline formatter and writer
- `models.py` - shared config/dependency/result dataclasses
- `init.py` - `init` mode workflow and internal-source enforcement
- `ingest.py` - `ingest` mode source expansion + review/apply flow
- `query.py` - `query` mode analysis + optional durable filing flow
- `lint.py` - `lint` mode health-check reconciliation flow
- `README.md` - setup and usage
- `pyproject.toml` - example-local dependency config

## Workspace layout

`init` creates this top-level layout in the wiki repo:

- `AGENTS.md` - schema and workflow rules the LLM follows for ingest/query/lint.
- `raw/` - immutable source files dropped in for ingest (articles, notes, datasets).
- `wiki/` - LLM-maintained knowledge pages (entities, concepts, summaries, syntheses).
- `wiki/index.md` - content-oriented catalog for wiki navigation and retrieval: categorized page links with one-line summaries and optional metadata (for example date/source count). Query flows read this first.
- `log.md` - append-only chronological interaction log. Every ingest/query/lint phase appends a parseable heading: `## [YYYY-MM-DD] mode.phase | outcome=...`, plus timestamp/summary bullets.

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

1. Review phase (read-only): the model reads staged source files and returns key takeaways, proposed wiki updates, contradictions, and index updates.
2. Apply phase (write): after your confirmation, the model writes source summary updates and concept/entity updates.
3. The runner refreshes `wiki/index.md` and appends structured `ingest.review` / `ingest.apply` timeline entries in `log.md`.
4. In `--review` mode, declining confirmation skips wiki edits, but still appends an `ingest.apply | outcome=canceled` entry and pushes so the timeline remains complete.

Batch ingest is the default. A single run can process multiple files and directories.

## Query workflow

`query` runs in two phases automatically:

1. Analysis phase (read-only): the model reads `wiki/index.md`, then recent `log.md` entries for recency context, then (when helpful) checks prior `wiki/query/*.md` pages for discovery/routing, expands into canonical wiki pages for grounding, answers with citations, and decides whether the result should be filed for future reuse. Query pages are treated as routing hints rather than primary evidence.
2. Filing phase (write, conditional): if the answer is durable, the runner files it into `wiki/query/<question-slug>.md` and refreshes `wiki/index.md`.
3. The runner always appends structured query timeline entries (`query.review` and optionally `query.apply`) to `log.md` and pushes so query history is complete, even on `skip`.

## Lint workflow

`lint` is single-pass and applies immediately:

1. Health-check phase (apply): the model reads recent `log.md` entries for recency context, then reconciles contradictions, stale/superseded claims, orphan pages, missing cross-references, and key concept coverage directly in `/wiki/` (creating new canonical pages when needed).
2. Gap reporting phase (in response): the model returns a concise summary with reconciled changes, remaining gaps, and suggested next questions/sources.
3. The runner refreshes `wiki/index.md`, appends a structured `lint.apply` entry to `log.md`, and pushes.

## Log timeline

`log.md` is runner-managed, append-only, and designed to be parseable with simple shell tools.

- The agent should not edit `log.md` directly; the runner appends entries.
- Every interaction is recorded:
  - `ingest.review` (when `--review` is enabled)
  - `ingest.apply` (`outcome=applied` or `outcome=canceled`)
  - `query.review` (`outcome=file` or `outcome=skip`)
  - `query.apply` (only when filing, `outcome=filed`)
  - `lint.apply` (`outcome=applied`)
- Entry shape:
  - Heading: `## [YYYY-MM-DD] mode.phase | outcome=... key=value ...`
  - Body bullets: `timestamp` (UTC) and `summary`

```bash
# Show the latest 5 timeline entries.
grep "^## \\[" log.md | tail -5

# Show the latest query review outcomes.
grep "^## \\[.*\\] query.review \\|" log.md | tail -10
```
