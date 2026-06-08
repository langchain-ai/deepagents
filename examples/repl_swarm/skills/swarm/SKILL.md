---
name: swarm
description: Dispatch a batch of tasks to subagents in parallel with bounded concurrency. Returns a summary with {total, completed, failed, results[]}.
metadata:
  required-ptc-tools: task read_file
---

# Swarm

Fan out a list of tasks to subagents in parallel, collect results, and return a summary.

## How to use

Read the reference script first, then write your own eval block based on it.
Do **not** try to `import` this skill — there is no module to load.

```javascript
// eval 1: read the reference
const script = await tools.readFile({ file_path: "/skills/swarm/scripts/index.ts" });
console.log(script);
```

Then write a second eval that implements the same `runSwarm` pattern with your tasks.

## Example

```javascript
// Define runSwarm inline (from reading index.ts)
async function runSwarm({ tasks, concurrency = 5, subagentType = "general-purpose" }) {
  const results = new Array(tasks.length);
  let next = 0;
  const worker = async () => {
    while (next < tasks.length) {
      const i = next++;
      const t = tasks[i];
      try {
        const out = await tools.task({
          description: t.description,
          subagent_type: t.subagentType ?? subagentType,
        });
        results[i] = { id: i, status: "completed", output: String(out) };
      } catch (err) {
        results[i] = { id: i, status: "failed", error: String(err) };
      }
    }
  };
  const workers = [];
  for (let w = 0; w < Math.min(concurrency, tasks.length); w++) workers.push(worker());
  await Promise.all(workers);
  const completed = results.filter(r => r.status === "completed").length;
  return { total: tasks.length, completed, failed: tasks.length - completed, results };
}

// Use it
const { completed, failed, results } = await runSwarm({
  tasks: [
    { description: "Write the number 1 to /tmp_swarm/a" },
    { description: "Write the number 2 to /tmp_swarm/b" },
    { description: "Write the number 3 to /tmp_swarm/c" },
  ],
  concurrency: 3,
});

console.log(`${completed} ok, ${failed} failed`);
for (const r of results) console.log(r.id, r.status, r.output ?? r.error);
```

## Notes

- Failures are caught per-task — one failure does not abort the swarm.
- `tools.task` is the only PTC dispatch primitive needed.
- Check `failed` count and retry specific tasks if needed.
