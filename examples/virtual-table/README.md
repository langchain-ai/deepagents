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
concurrency, per-row errors, cross-turn state, and SQL semantics. Raw
code remains the better escape hatch for adaptive workflows.

The middleware adds its own tool-usage and table-lifecycle instructions to the
parent model's system message, including that tables supplied in invocation state
already exist and other tables require `virtual_table_create`. The host agent does
not need a virtual-table-specific system prompt or a predeclared subagent.

An enrichment operator is chosen when the middleware is initialized, not by the
model on each tool call. The example configures `DeepAgentOperator`, which owns its
worker model, backend, concurrency, and timeout. A different implementation could
use a fast classifier or embeddings without changing the agent-facing tool. Each
Deep Agent worker gets its row's file path and selected metadata, and can use the
shared filesystem and standard tools to gather the context it needs. The parent
should inspect table metadata first and avoid eagerly reading every source file;
sampling one or two files is enough when needed to design the enrichment.

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
- `virtual_table_enrich`: select rows using `source_table` plus an optional
  parameterized `where`, project `input_columns`, and create a new `output_table`
  using the configured operator. The call supplies instructions and a strict JSON
  Schema. Its result names the derived table, exact schema, and success/failure
  counts; every row also has `_enrichment_status` and `_enrichment_error`.
- `virtual_table_query`: inspect rows/columns and execute one read-only `SELECT`
  or `WITH` query using parameter binding, an SQLite authorizer, a time limit,
  and a row limit.

The tool schema intentionally exposes `source_table` plus `where` rather than an
arbitrary `source_query` in this first version. That supports the common filter and
projection case while keeping validation simple; a future relation input could add
joins and computed columns without changing the operator contract.

For example, the parent can first materialize a relational transformation:

```json
{
  "source_table": "feedback",
  "output_table": "classified_feedback",
  "input_columns": ["file", "customer", "plan"],
  "where": "plan = ?",
  "parameters": ["enterprise"],
  "instructions": "Read `file`; classify sentiment and product area.",
  "output_schema": {
    "type": "object",
    "properties": {
      "sentiment": {"type": "string"},
      "product_area": {"type": "string"}
    },
    "required": ["sentiment", "product_area"]
  }
}
```

It can then issue ordinary SQL against `classified_feedback`. The source table is
left unchanged, so the materialized result has an explicit name and can be queried
repeatedly without rerunning inference.

## Why enrichment is a separate relational operation

A model call embedded directly in SQL is more concise for cheap, row-local
classification: filtering, inference, and aggregation fit in one expression. This
prototype instead gives enrichment **job semantics** and SQL **query semantics**.
The split is useful when an enrichment can launch tool-using agents or take long
enough that retries, partial failures, and accidental reevaluation matter:

- Inference runs at one explicit boundary. A retried query, join, or scan cannot
  silently repeat expensive model calls.
- The derived table is durable state. Several SQL analyses can reuse it, and the
  original relation remains available for comparison.
- Selection is still relational. `where`, bound parameters, and `input_columns`
  push filtering and projection ahead of the operator instead of asking workers
  to discard irrelevant rows.
- The returned table name and schema tell the parent exactly what it can query
  next; it does not need to infer which columns were added by a mutation.
- Operator policy stays outside model control. Credentials, model choice,
  filesystem access, concurrency, timeout, caching, and retries can be fixed at
  middleware initialization.

The cost is an extra tool call and an explicitly named intermediate table. It is
less elegant than one SQL statement for interactive analytics, and materialized
results need lifecycle/provenance rules as the prototype evolves. The design makes
that trade deliberately: predictable execution for expensive, potentially
agentic work, followed by familiar and side-effect-free SQL.

Tables are passed under `virtual_tables` in invocation state alongside `files`, so
the document rows and the files they reference enter the run together. `virtual_tables`
is the state-field name, not a queryable SQL table; its keys (for example, `feedback`)
are the table names. The middleware lists those names in its runtime instructions,
normalizes paths and row IDs before the model runs, and returns materialized rows in
the same structured state field. Document blobs stay in the configured filesystem
backend. With a checkpointer, both virtual files and table pointers persist across
turns without copying large blobs into every table row. The included script uses one
`StateBackend` for both `create_deep_agent` and `VirtualTableMiddleware`.

## Current prototype limits

- At most 500 rows and 2 MB of paths/metadata per materialized table by default.
- The configured example operator uses normal Deep Agents with shared filesystem access; their runs are bounded by its timeout.
- `DeepAgentOperator` allows at most 10 concurrent row workers; five by default.
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
