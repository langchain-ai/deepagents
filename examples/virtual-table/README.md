# Virtual table middleware prototype

This example explores an in-process Deep Agents middleware for large textual
datasets. It combines two operations:

1. **Semantic enrichment:** run a configured Deep Agents subagent independently
over selected rows and materialize its structured response as new columns.
2. **Deterministic analysis:** run bounded, read-only SQLite queries over those
materialized rows for filtering, grouping, joins, and aggregation.

It is deliberately an example, not library API. It requires no sandbox, no CLI,
and no dependency beyond Deep Agents and Python's standard library.

## Why this differs from `js_eval + task`

The existing REPL can already map `task()` across an array and aggregate in
JavaScript. This prototype tests whether a table-shaped control plane is more
reliable by owning stable row IDs, structured column materialization, bounded
concurrency, per-row errors, private cross-turn state, and SQL semantics. Raw
code remains the better escape hatch for adaptive workflows.

The `virtual_table_enrich` tool finds the same `task` tool installed by
`SubAgentMiddleware` and calls it through its async LangChain tool interface. It
does not launch a separate agent runtime and does not route through `js_eval`.
Other subagent state updates are discarded, so enrichment subagents should be
read-only.

## Run

From this directory:

```bash
uv run python agent.py
```

The default model is `anthropic:claude-sonnet-4-6`, so set
`ANTHROPIC_API_KEY`. Override the model or question as needed:

```bash
uv run python agent.py \
  --model openai:gpt-5.4 \
  "Classify each row, then summarize only enterprise complaints."
```

The script uses `agent.ainvoke()` because row enrichment dispatches subagents
concurrently. The synchronous enrichment tool returns a message directing the
caller to async invocation.

## Tools

- `virtual_table_create`: materialize JSON rows with stable `_row_id` values.
- `virtual_table_describe`: return columns, row count, and a bounded sample.
- `virtual_table_enrich`: add structured output fields plus
  `<name>_status` and `<name>_error` columns.
- `virtual_table_query`: execute one read-only `SELECT` or `WITH` query using
  parameter binding, an SQLite authorizer, a time limit, and a row limit.

Tables live in `_virtual_tables`, a `PrivateStateAttr`. With a checkpointer they
can persist across turns without appearing in public agent input/output schemas.
The included script prints the state returned from its single run.

## Current prototype limits

- At most 500 rows and 2 MB per materialized table by default.
- At most 10 concurrent subagents; five by default.
- One SQLite table per query call.
- Query results are capped at 100 rows and 100 KB.
- SQL uses a small allowlist of deterministic functions and cannot write data.
- Enrichment is materialized; a `SELECT` never triggers model calls.
- Successful rows are committed only when the enrichment tool finishes; there
  is not yet checkpoint-per-row resume or a provenance/cache layer.
- JSON Schema size/property count are bounded, but this example delegates full
  schema enforcement to the existing Deep Agents structured-response path.

## Test

```bash
uv run pytest test_virtual_table.py
```

The focused tests use a fake `task` tool and make no network calls.
