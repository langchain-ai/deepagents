---
name: swarm
description: Dispatch a batch of tasks to subagents in parallel with bounded concurrency, then collect results.
module: ./index.ts
---

# Swarm

Fan out a list of tasks to subagents with bounded
concurrency, collect results, and return a compact summary.

## When to use

You have many independent tasks (e.g. "summarize each of these 20
files", "classify each of these 50 tickets", "research these 15
topics") and want them to run concurrently rather than sequentially.

## Usage

```javascript
const { runSwarm } = await import("@/skills/swarm");

const summary = await runSwarm({
  tasks: [
    { description: "Summarize /notes/alpha.md" },
    { description: "Summarize /notes/beta.md" },
    { description: "Summarize /notes/gamma.md" },
  ],
  concurrency: 3,           // optional, defaults to 5, capped at 10
  subagentType: "general-purpose",  // optional; per-task override wins
});

console.log(summary);
```

## Contract

The `runSwarm(opts)` function accepts:

- `tasks` (required): an array of `{ description: string, subagentType?: string }`.
- `concurrency` (optional, default `5`, capped at `10`): max parallel
  subagent invocations.
- `subagentType` (optional, default `"general-purpose"`): the default
  subagent to dispatch each task to. A task's own `subagentType`
  takes precedence.

Returns a summary object:

```typescript
{
  total: number;          // tasks.length
  completed: number;      // subagents that returned a result
  failed: number;         // subagents that threw
  results: {              // one entry per task, in input order
    id: number;           // 0-indexed position in `tasks`
    status: "completed" | "failed";
    result?: string;      // on success — subagent's final message
    error?: string;       // on failure — error message
  }[];
}
```

## Design notes

- Dispatch goes through `tools.task`, which the REPL's PTC layer
  exposes for us. The skill does not register any subagent itself —
  it's a fan-out pattern on top of what the agent already has.
- Failures are caught per-task: one failed subagent does not abort
  the swarm. Check the `failed` count and the per-task `status`.
- Concurrency is bounded with a semaphore-style pool rather than
  `Promise.all` on everything. For 100 tasks with `concurrency=5`,
  this keeps memory and tool-call rate predictable.
