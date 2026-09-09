# Virtual table middleware prototype

This example explores an in-process Deep Agents middleware for large textual
datasets. It combines two operations:

1. **Semantic enrichment:** define a temporary row worker with a prompt and JSON
Schema, run it independently over selected rows, and materialize its structured
response as new columns.
2. **Deterministic analysis:** run bounded, read-only SQLite queries over those
materialized rows for filtering, grouping, joins, and aggregation.

It is deliberately an example, not library API. Each row is a document dictionary
with a mandatory `file` backend path and optional queryable metadata. The file may
live in any Deep Agents filesystem backend, including the default virtual state
filesystem or a per-thread sandbox; the table stores the path rather than the blob.

## Why this differs from `js_eval + task`

The existing REPL can already map `task()` across an array and aggregate in
JavaScript. This prototype tests whether a table-shaped control plane is more
reliable by owning stable row IDs, structured column materialization, bounded
concurrency, per-row errors, private cross-turn state, and SQL semantics. Raw
code remains the better escape hatch for adaptive workflows.

The middleware adds its own tool-usage and table-lifecycle instructions to the
parent model's system message, including that `initial_tables` already exist and
other tables require `virtual_table_create`. The host agent does not need a virtual-table-specific system prompt or a
predeclared subagent. Each `virtual_table_enrich` call defines a temporary,
read-only row worker with `worker_prompt` and `output_schema`; it inherits the
parent model unless `worker_model` is set. Each worker gets only its row's file
path and selected metadata, then uses paginated `read_file` calls as needed. The
parent should inspect table metadata first and avoid eagerly reading every source
file; sampling one or two files is enough when needed to design the enrichment.
The workers do not route through `js_eval`.

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

The script uses `agent.ainvoke()` because row enrichment dispatches row workers
concurrently. The synchronous enrichment tool returns a message directing the
caller to async invocation.

## Tools

- `virtual_table_create`: materialize document rows shaped like
  `{"file": "/docs/a.txt", ...metadata}` with stable `_row_id` values.
- `virtual_table_enrich`: define a row worker from a prompt and strict output
  schema, then add its structured fields plus `<name>_status` and `<name>_error`
  columns. Each worker can read only that row's file and can page through large
  documents instead of receiving the entire document in its initial context.
- `virtual_table_query`: inspect rows/columns and execute one read-only `SELECT`
  or `WITH` query using parameter binding, an SQLite authorizer, a time limit,
  and a row limit.

Tables live in the middleware-owned `_virtual_tables` state field, while document
blobs stay in the configured filesystem backend. The field is omitted from the
input schema so callers cannot seed it directly, but included in invocation output
so applications can consume the materialized rows as structured state. Enrichment
passes each worker only its validated file path and selected metadata; the worker's
read-only filesystem tool pages through the source as needed. With a checkpointer,
both virtual files and table pointers persist across turns without copying large
blobs into every table row. The included script
uses one `StateBackend` for both `create_deep_agent` and `VirtualTableMiddleware`.

## Current prototype limits

- At most 500 rows and 2 MB of paths/metadata per materialized table by default.
- Row workers have only `read_file` access to their own document and must page through large text files.
- At most 10 concurrent row workers; five by default.
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

The focused tests use a fake row worker and filesystem backend and make no network calls.
