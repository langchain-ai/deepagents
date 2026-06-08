---
name: swarm
description: >-
  Dispatches many independent items in parallel: build a row table, fan out
  to subagents via tools.task(), merge results back. One row = one unit of work.
compatibility: >-
  Requires @langchain/quickjs code interpreter with task PTC tool
metadata:
  required-ptc-tools: task read_file write_file glob
---

# Swarm

Process many independent items in parallel by writing a single eval block
that defines the table and dispatch helpers inline — no import needed.

## Flow

1. **Read the scripts.** Use `read_file` to load `scripts/table.ts` and
   `scripts/executor.ts`. Understand the `create`, `rows`, `mergeResults`,
   and `run` helper signatures.
2. **Write one eval block.** Define those helpers inline, build your row table
   with `create()`, dispatch with `run()`, and log the results.

Everything lives in a single eval — no state to carry across blocks.

## Reading the reference scripts

```javascript
const tableTs = await tools.readFile({ file_path: "/skills/swarm/scripts/table.ts" });
const executorTs = await tools.readFile({ file_path: "/skills/swarm/scripts/executor.ts" });
console.log(tableTs, executorTs);
```

Then write a second eval that implements the pattern with your data.

## Example

```javascript
// Define table helpers (from table.ts)
function create(tasks) {
  return tasks.map((t) => ({ ...t }));
}
function rows(table, filter) {
  return filter ? table.filter(filter) : [...table];
}
function mergeResults(table, results) {
  const byId = new Map(results.map((r) => [r.id, r]));
  for (const row of table) {
    const result = byId.get(row.id);
    if (result) Object.assign(row, result);
  }
}

// Define dispatch helper (from executor.ts)
async function run(table, opts) {
  const { instruction, responseSchema, subagentType, concurrency = 10 } = opts;
  const schemaHint = responseSchema
    ? `\n\nRespond with JSON matching: ${JSON.stringify(responseSchema)}`
    : "";
  const out = [];
  for (let i = 0; i < table.length; i += concurrency) {
    const chunk = table.slice(i, i + concurrency);
    const results = await Promise.all(chunk.map(async (row) => {
      const prompt = instruction.replace(/\{(\w+)\}/g, (_, k) => String(row[k] ?? "")) + schemaHint;
      try {
        const raw = await tools.task({ description: prompt });
        const match = raw.match(/\{[\s\S]*\}/);
        return { id: row.id, result: match ? JSON.parse(match[0]) : { output: raw } };
      } catch (err) {
        return { id: row.id, error: String(err) };
      }
    }));
    out.push(...results);
  }
  return out;
}

// Use them
const table = create([
  { id: "r1", text: "Love this product!" },
  { id: "r2", text: "Terrible experience." },
]);

const results = await run(table, {
  instruction: 'Classify the sentiment of: "{text}"',
  responseSchema: {
    type: "object",
    properties: { sentiment: { type: "string", enum: ["positive", "negative", "neutral"] } },
    required: ["sentiment"],
  },
});

mergeResults(table, results);
console.log(JSON.stringify(table, null, 2));
```

## Retrying failures

```javascript
const failed = rows(table, (r) => r.error != null);
if (failed.length > 0) {
  const retries = await run(failed, { instruction: "...", responseSchema: { ... } });
  mergeResults(table, retries);
}
```

## Technical notes

- **All helpers defined inline.** Copy the function bodies from the scripts into your
  eval block — do not try to import them.
- **`tools.task` is the only PTC dispatch primitive.** It accepts `description` and
  an optional `subagent_type`. Ask for JSON in the description to get structured output.
- **Console output is capped at ~5 KB.** Log counts and short samples, not raw data.
- **Parallel dispatches are capped at 10 by default.** The `run()` helper chunks
  larger tables automatically.
