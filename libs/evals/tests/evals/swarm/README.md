# Swarm-skill smoke eval

Confirms the REPL can load the `swarm` skill and that a `runSwarm`
call actually dispatches parallel subagent tasks with correct results.
Off-catalog (not tagged with `eval_category` / `eval_tier`) — run by
path.

## What it checks

One test, five items, one agent invocation:

1. Agent gets five files under `/items/NN.txt`, each containing one
   noun (e.g. `elephant`, `granite`, `oak`).
2. Prompt asks the agent to classify each as animal / plant / mineral
   and return a JSON dict, using the `swarm` skill to parallelize.
3. Assertions:
   - **Skill invoked** — at least one `eval` tool call references
     `runSwarm(` in the code body. Without this, correctness would be
     provable only about the model, not the skill wiring.
   - **Fan-out happened** — the `SwarmSummary` emitted by `runSwarm`
     (via `console.log(JSON.stringify(summary))`) shows `completed`
     ≥ 5. `tools.task(...)` dispatches happen inside the QuickJS
     REPL, so they don't appear on the root agent's tool-call list;
     the summary is how we prove the fan-out actually ran. If the
     model skips the `console.log` the summary won't be parseable
     and this check fails — SKILL.md's Usage example shows the
     pattern to follow.
   - **Correctness** — returned JSON dict matches the expected mapping.

## Wiring

Backend is a `CompositeBackend` that routes `/skills/*` to a
`FilesystemBackend` rooted at this package's `skills/` directory
(symlinked to `examples/repl_swarm/skills/`) and sends everything else
to `StateBackend`. Task-scoped data (`/items/NN.txt`) flows through
state via `run_agent_async(initial_files=...)` as usual — only skill
source comes off the filesystem.

Matches the shape used by `examples/repl_swarm/swarm_agent.py` and
`libs/evals/tests/evals/oolong/runners/swarm/`. The skill source is
one source of truth across all three — edit the files in
`examples/repl_swarm/skills/swarm/` and the symlinks pick it up.

## Running

```bash
export LANGSMITH_API_KEY=...        # optional; enables LangSmith logging
export LANGSMITH_TRACING=true
export ANTHROPIC_API_KEY=...        # required — the test hits a real model

uv run --group test pytest \
    libs/evals/tests/evals/swarm/test_swarm_fanout.py -v -s
```

Uses the same `model` fixture as the rest of the eval suite.

## Relationship to `oolong/`

The `runner.py` here uses the same backend shape as
`oolong/runners/swarm/__init__.py`. Two copies let either side evolve
independently; if they drift far, promote the shared bits to a module
under `tests/evals/`. Skill source is already shared via symlink to
`examples/repl_swarm/skills/` — the duplication is just the Python
wiring (~10 lines).
