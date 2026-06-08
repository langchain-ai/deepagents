---
name: swarm
description: >-
  Dispatches many independent items in parallel: build a row table, fan out
  to subagents via tools.task(), merge results back. One row = one unit of work.
compatibility: >-
  Requires @langchain/quickjs code interpreter with task, read_file, glob PTC tools
metadata:
  required-ptc-tools: task read_file write_file glob
---

# Swarm

Process many independent items in parallel. Read the reference scripts, then write
a single eval block that defines the helpers inline and uses them.

**Do not** try to `import` this skill — there is no module to load.

## Flow

1. **Read the scripts.** Use `read_file` to load `scripts/table.ts` and
   `scripts/executor.ts`. Understand `create`/`createFromGlob`/`createFromFiles`,
   `rows`, `run`, and `validate`.
2. **Write one eval block.** Define those helpers inline, build your table,
   dispatch with `run()`, filter and retry if needed.

## Reading the scripts

```javascript
const tableTs = await tools.readFile({ file_path: "/skills/swarm/scripts/table.ts" });
const executorTs = await tools.readFile({ file_path: "/skills/swarm/scripts/executor.ts" });
console.log(tableTs, "\n---\n", executorTs);
```

## Sources for `create`

**`create(tasks)`** — pass pre-built records. Each must include a string `id`.

```javascript
const table = create([
  { id: "r1", text: "Love this product!" },
  { id: "r2", text: "Terrible experience." },
]);
```

**`createFromGlob(pattern)`** — one file = one row with `{ id, file }`. Requires `glob` PTC.

```javascript
const table = await createFromGlob("src/**/*.ts");
// → [{ id: "src_auth_ts", file: "src/auth.ts" }, ...]
```

**`createFromFiles(paths)`** — explicit file list, same row shape as glob.

```javascript
const table = createFromFiles(["/notes/a.md", "/notes/b.md"]);
```

## Dispatching with `run()`

`run()` dispatches each row, merges results onto rows in place, and returns
`{ completed, failed, failures }`.

```javascript
const summary = await run(table, {
  instruction: 'Classify the sentiment of: "{text}"',
  responseSchema: {
    type: "object",
    properties: { sentiment: { type: "string", enum: ["positive", "negative", "neutral"] } },
    required: ["sentiment"],
  },
});
console.log(summary);
// → { completed: 2, failed: 0, failures: [] }
// Rows are mutated: [{ id: "r1", text: "...", sentiment: "positive" }, ...]
```

### Options

| Option | Default | Description |
|---|---|---|
| `instruction` | required | Template with `{column}` placeholders |
| `responseSchema` | — | JSON Schema — used as prompt hint and for validation |
| `context` | — | Prose prepended to every subagent prompt |
| `subagentType` | — | Named subagent. Omit for default general-purpose |
| `concurrency` | `10` | Max parallel dispatches |

### Using `context`

```javascript
await run(table, {
  instruction: "Review {file} for security issues.",
  context: "TypeScript Express backend using Prisma ORM. Focus on injection and auth bypass.",
  subagentType: "reviewer",
  responseSchema: { type: "object", properties: { review: { type: "string" } }, required: ["review"] },
});
```

## Querying rows

Use `rows()` with a plain JS predicate for aggregation and retry targeting:

```javascript
const failed   = rows(table, r => !!r.error);
const negative = rows(table, r => r.sentiment === "negative");
const all      = rows(table);
```

## Retry pattern

```javascript
// First pass
await run(table, { instruction: "...", responseSchema: { ... } });

// Retry only failed rows
const failed = rows(table, r => !!r.error);
if (failed.length > 0) {
  await run(failed, { instruction: "...", responseSchema: { ... } });
}
```

## Chaining passes

`run()` mutates rows in place — chain calls to accumulate columns:

```javascript
await run(table, {
  instruction: "Classify sentiment of {text}",
  responseSchema: { type: "object", properties: { sentiment: { type: "string" } }, required: ["sentiment"] },
});

const negative = rows(table, r => r.sentiment === "negative");
await run(negative, {
  instruction: "Summarize why {text} had negative sentiment.",
  responseSchema: { type: "object", properties: { summary: { type: "string" } }, required: ["summary"] },
});
```

## Action-only tasks (no structured output)

```javascript
const table = await createFromGlob("src/**/*.ts");
await run(table, {
  subagentType: "fixer",
  instruction: "Add missing JSDoc to all exported functions in {file}.",
  responseSchema: { type: "object", properties: { fixed: { type: "string" } }, required: ["fixed"] },
});
// Retry any that failed
const failed = rows(table, r => !!r.error);
if (failed.length > 0) {
  await run(failed, { subagentType: "fixer", instruction: "Add missing JSDoc to all exported functions in {file}.", responseSchema: { type: "object", properties: { fixed: { type: "string" } }, required: ["fixed"] } });
}
```

## Technical notes

- **All helpers defined inline.** Copy function bodies from the scripts into your eval block.
- **`tools.task` is the only dispatch primitive.** Accepts `description` and optional `subagent_type`.
- **`run()` mutates rows in place.** Results are merged as top-level columns; errors set `row.error`.
- **`responseSchema` drives both prompt hint and validation.** Invalid responses are marked as failures with `validationErrors` detail in `failures[]`.
- **Console output is capped at ~5 KB.** Log counts and short samples, not raw data.
- **Everything the subagent needs must be in `instruction` + `context`.** Subagents cannot see the agent's context.
