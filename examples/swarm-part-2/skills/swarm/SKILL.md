---
name: swarm
description: >-
  Dispatches many independent items in parallel: create a table, fan out to
  subagents, aggregate results. One row = one unit of work.
compatibility: >-
  Requires @langchain/quickjs code interpreter with swarm_task PTC tool
metadata:
  entrypoint: scripts/index.ts
  required-ptc-tools: swarm_task read_file write_file edit_file glob
---

# Swarm

Process many independent items in parallel. `create` builds a table handle;
`run` fans work out across rows and merges results back. One row = one unit
of work — swarm handles batching automatically.

## Flow

1. **Create.** Build a table from a source — files, a glob pattern, or
   pre-parsed records. One row per item. Returns a handle.
2. **Run.** Dispatch an `instruction` template across rows. Results are merged
   back into the table. Returns `{ completed, failed, skipped, failures }`.
3. **Aggregate.** Use `rows()` and plain JS to count, filter, or summarize.
   Do not spawn additional subagents for aggregation.
4. **Retry.** Re-run with `filter: { column: "<col>", exists: false }` to
   reprocess only failed rows.

## Choosing a source

**`glob` / `filePaths`** — one file = one row. Use when each file is an
independent unit of work. Each row gets `{ id, file }`; the subagent reads
the file itself via the `{file}` placeholder.

**`tasks`** — pass pre-built records directly. Use when the data lives inside
a file (JSONL, CSV, JSON array). Read and parse the file first inside
`eval`, then pass the records. One record = one row — do not group
multiple items into a single row.

For small files (under ~500 lines), parse and create in one block:

```javascript
const { create } = await import("@/skills/swarm");
const raw = await tools.readFile({ file_path: "/data.jsonl" });
const records = raw.trim().split("\n").map(l => JSON.parse(l));
const table = await create({ tasks: records });
console.log(table);
```

For large files, read in chunks of 500 lines to avoid truncation:

```javascript
const { create } = await import("@/skills/swarm");
let records = [];
let offset = 0;
while (true) {
  const chunk = await tools.readFile({ file_path: "/data.txt", offset, limit: 500 });
  const lines = chunk.split("\n").filter(l => l.trim());
  for (const l of lines) { records.push({ id: `r${records.length}`, text: l }); }
  if (lines.length < 500) break;
  offset += 500;
}
const table = await create({ tasks: records });
console.log(table);
```

When the file is too large to parse and dispatch in one `eval` call, split
across two blocks. Only the block that calls swarm functions needs the import:

```javascript
// eval 1: parse only — no swarm import needed
const raw = await tools.readFile({ file_path: "/data.jsonl" });
globalThis.records = raw.trim().split("\n").map(l => JSON.parse(l));
console.log(`Parsed ${globalThis.records.length} records`);
```

```javascript
// eval 2: create and dispatch
const { create, run } = await import("@/skills/swarm");
const table = await create({ tasks: globalThis.records });
const result = await run(table.id, {
  instruction: "Classify {text}",
  responseSchema: {
    type: "object",
    properties: { label: { type: "string" } },
    required: ["label"],
  },
});
console.log(result);
```

Passing `filePaths: ["/data.jsonl"]` would produce a table with **one row**
pointing at the file — not one row per record inside it.

## When to use `subagentType`

Omit `subagentType` for classification, extraction, labeling, and any task
where a single model call with structured output is sufficient. This is the
default and is significantly cheaper and faster — each dispatch is a direct
model call, no tools, no iteration.

Set `subagentType` when the task requires tools, file access, or multi-step
reasoning. Each dispatch runs a full agentic loop with the named subagent.

```javascript
// Direct model call — classification, no tools needed
await run(table.id, {
  instruction: "Classify {text}",
  responseSchema: { type: "object", properties: { label: { type: "string" } }, required: ["label"] },
});

// Subagent — needs to read files and reason over multiple steps
await run(table.id, {
  subagentType: "reviewer",
  instruction: "Review {file} for security issues.",
  responseSchema: { type: "object", properties: { finding: { type: "string" } }, required: ["finding"] },
});
```

## Instruction + context

`instruction` is a per-item template with `{column}` placeholders.
Placeholders are resolved by the framework — your column names appear in
prompts as references to the values listed alongside, never as raw
template syntax. Subagents do the work — do not process items yourself in
JS and write the results into rows.

`context` is free-form prose prepended to every subagent prompt. Use it for
shared background: domain terms, classification rules, examples, etc.

```javascript
const { create, run } = await import("@/skills/swarm");

const table = await create({ glob: "src/**/*.ts" });
const r = await run(table.id, {
  subagentType: "reviewer",
  instruction: "Review {file} for security issues. List findings or write 'no issues'.",
  context: "TypeScript Express backend using Prisma ORM. Focus on injection, auth bypass, path traversal.",
  responseSchema: {
    type: "object",
    properties: { review: { type: "string" } },
    required: ["review"],
  },
});
console.log(r);
// → { completed: 45, failed: 2, skipped: 0, failures: [...] }
```

## Structured output

`responseSchema` is required. Schema properties become top-level columns on
each row and constrain what subagents can return.

```javascript
const { run } = await import("@/skills/swarm");
await run(table.id, {
  instruction: "Classify: {text}",
  responseSchema: {
    type: "object",
    properties: {
      sentiment: { type: "string", enum: ["positive", "negative", "neutral"] },
    },
    required: ["sentiment"],
  },
});
// Row after: { id: "r1", text: "...", sentiment: "positive" }
```

## Batching

By default, swarm auto-batches to keep total dispatches under 10. For small
tables (≤10 rows) each row gets its own subagent call. For larger tables,
rows are grouped automatically.

Set `batchSize` to control grouping:

- **Number** — uniform batch size for all rows. `batchSize: 1` forces per-row
  dispatch; `batchSize: 20` groups in twenties.
- **Function** — `(row, rowCount) => number`. Returns the desired batch size
  for each row. Rows with the same batch size are grouped together, then
  chunked. Allows mixed dispatch where some rows go solo and others batch.

```javascript
const { create, run } = await import("@/skills/swarm");
const table = await create({ tasks: items });

// Complex items get individual attention; simple ones batch together
await run(table.id, {
  instruction: "Analyze {text}",
  responseSchema: {
    type: "object",
    properties: { analysis: { type: "string" } },
    required: ["analysis"],
  },
  batchSize: (row) => (row.token_count > 1000 ? 1 : 10),
});
```

Batch sizes are clamped to [1, 50] after evaluation.

## Aggregation

After `run()`, use `rows()` and plain JS — no additional subagents needed.

```javascript
const { rows } = await import("@/skills/swarm");
const data = await rows(table.id, { columns: ["sentiment"] });
const counts = {};
data.forEach(r => { counts[r.sentiment] = (counts[r.sentiment] || 0) + 1 });
console.log(counts);
// → { positive: 120, negative: 45, neutral: 35 }
```

## Chaining passes

`run` updates the table in place — chain calls to accumulate columns.

```javascript
const { create, run } = await import("@/skills/swarm");
const table = await create({ tasks: interviews });
await run(table.id, {
  instruction: "Classify sentiment of {text}",
  responseSchema: {
    type: "object",
    properties: { sentiment: { type: "string", enum: ["positive", "negative", "neutral"] } },
    required: ["sentiment"],
  },
});
await run(table.id, {
  filter: { column: "sentiment", equals: "negative" },
  instruction: "Summarize why {text} had negative sentiment.",
  responseSchema: {
    type: "object",
    properties: { summary: { type: "string" } },
    required: ["summary"],
  },
});
```

## Action-only tasks

When subagents perform actions (write a file, apply a fix) rather than return
data, use a simple schema with a status or marker field. The `exists: false`
filter still works for retries.

```javascript
const { create, run } = await import("@/skills/swarm");
const fixedSchema = {
  type: "object",
  properties: { fixed: { type: "string" } },
  required: ["fixed"],
};
const table = await create({ glob: "src/**/*.ts" });
await run(table.id, {
  subagentType: "fixer",
  instruction: "Add missing JSDoc to all exported functions in {file}.",
  responseSchema: fixedSchema,
});
// retry any that failed
await run(table.id, {
  subagentType: "fixer",
  instruction: "Add missing JSDoc to all exported functions in {file}.",
  responseSchema: fixedSchema,
  filter: { column: "fixed", exists: false },
});
```

## Filtering

```javascript
{ column: "status", equals: "done" }
{ column: "status", notEquals: "done" }
{ column: "category", in: ["A", "B"] }
{ column: "result", exists: false }      // not yet processed
{ and: [filter1, filter2] }
{ or: [filter1, filter2] }
```

## Technical notes

- **Only import `@/skills/swarm` in blocks where you call swarm functions.**
  Data preparation (reading files, parsing, storing in `globalThis`) does not
  need the import. Destructure only what you use: `{ create }`, `{ run }`,
  `{ create, run }`, etc.
- **Console output is capped at ~5 KB.** Never log raw file contents —
  log only counts and short samples.
- **`readFile` inside `eval` returns raw content — no line-number
  prefixes.** Request at most 500 lines per call. For files with more
  than 500 lines, loop with incrementing `offset`.
- **When building a table from a file, read it inside `eval`.** Data read
  inside the sandbox stays there; it never enters the agent's context window.
- **Never write to `.swarm/` directly.** Always use `create()`.
- **Everything the subagent needs must be in `instruction` + `context`.**
  Subagents can't see the agent's context.
- **Row ids must be unique.** `create()` rejects sources that produce
  duplicate ids. For `tasks`, that's a caller-side responsibility; for
  `glob` / `filePaths`, ids are auto-disambiguated by parent directory.
- **Unknown columns fail fast.** If `instruction` references `{foo}` and
  no matched row provides `foo`, `run()` throws before any subagent is
  dispatched.

## API Reference

### `create(source)`

Create a table. Returns a handle `{ id, count, columns }`.

| Source | Description |
|--------|------------|
| `{ glob: "src/**/*.ts" }` or `{ glob: ["src/**/*.ts", "lib/**/*.ts"] }` | Match files by one or more patterns. Columns: `id`, `file` |
| `{ filePaths: ["a.ts", "b.ts"] }` | Explicit file list. Columns: `id`, `file` |
| `{ tasks: [{ id: "t1", text: "..." }] }` | Custom rows. Each must have `id` |

### `run(tableId, options)`

Dispatch work across rows. Returns `{ completed, failed, skipped, failures }`.

| Option | Default | Description |
|--------|---------|------------|
| `instruction` | (required) | Template with `{column}` placeholders |
| `responseSchema` | (required) | JSON Schema (`type: "object"`) — properties become row columns |
| `context` | — | Prose prepended to every subagent prompt |
| `filter` | — | Only dispatch matching rows |
| `subagentType` | — | Name of subagent to dispatch to. When set, runs a full agentic loop. When omitted, runs a direct model call |
| `batchSize` | auto | Number or `(row, rowCount) => number`. Auto caps dispatches at 10; `1` = per-row; function = per-row sizing |
| `concurrency` | `10` | Max concurrent subagent dispatches (clamped to 1–10) |

### `rows(tableId, options?)`

Retrieve rows. Use for inspection and JS-based aggregation.

| Option | Description |
|--------|------------|
| `filter` | Only return matching rows |
| `columns` | Project to specific columns |
| `limit` | Max rows returned |
