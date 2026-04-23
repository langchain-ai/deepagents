# REPL Swarm

Dispatch many subagent calls in parallel from inside the REPL, with
bounded concurrency, by importing a skill the agent pulls in on
demand.

The interesting part is that the orchestration logic — the semaphore
pool, task dispatch, result collection — lives inside a skill
(`skills/swarm/index.ts`), not in Python. The agent imports it with:

```javascript
const { runSwarm } = await import("@/skills/swarm");
```

…runs it, and gets a structured summary back.

## Structure

```
repl_swarm/
├── skills/
│   └── swarm/
│       ├── SKILL.md      # frontmatter (module: ./index.ts) + prose docs
│       └── index.ts      # the runSwarm() executor
├── swarm_agent.py        # thin driver that wires everything up
└── README.md             # this file
```

## How it works

1. `swarm_agent.py` creates a `create_deep_agent` with
   `skills=[".../skills"]` and
   `REPLMiddleware(ptc=True, skills_backend=backend)`.
2. `SkillsMiddleware` parses `SKILL.md` frontmatter, including the new
   `module` key, and writes a `SkillMetadata` entry into state.
3. When the model writes
   `await import("@/skills/swarm")` inside an `eval` call,
   `REPLMiddleware` scans the source, loads the skill dir via the
   backend, builds a `ModuleScope` with `index.ts` (oxidase strips
   TypeScript types at install time), and calls `ctx.install`.
4. Guest code imports `runSwarm`, which calls `tools.task(...)`
   through the PTC layer. `Promise.all` plus a worker-pool keeps at
   most `concurrency` calls in flight.
5. Results come back in input order; the skill returns a
   `SwarmSummary` with completion counts.

## Running the demo

```bash
uv run python swarm_agent.py
# custom task:
uv run python swarm_agent.py "Use the swarm skill to summarize these files: ..."
```

The default task asks the agent to write three different numbers to
three different paths in parallel — a concrete proof the fan-out
actually runs.

## When to use it

- **Many independent subtasks.** If the subtasks don't share state and
  don't depend on each other's output, a swarm saves round trips.
- **Stable subtask shape.** Each task is just a `description` string
  (optionally an override `subagentType`). If your subtasks need
  richer structured input, extend the skill.

## When not to use it

- **Sequential dependencies.** If task B needs task A's output, use
  plain `tools.task` with an ordinary `await` chain.
- **Hundreds of tasks.** The default `concurrency=5` cap (hard-max 10)
  is intentional — subagent invocations aren't free. For mass fan-out
  at thousands of tasks, batch them in a job queue, not a REPL call.
