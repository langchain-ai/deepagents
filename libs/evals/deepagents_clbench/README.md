# deepagents_clbench

Canonical source for the **`deepagents`** system in
[continual-learning-bench](https://github.com/pgasawa/continual-learning-bench)
(clbench) — a Deep Agent evaluated as a `ContinualLearningSystem`.

## Why it lives here but runs there

clbench discovers systems by scanning its own `src/systems/<name>/` tree on disk
(`src/registry.py:_discover_system_modules`). The adapter therefore has to
physically sit under a clbench checkout to be runnable, and it imports against
clbench's package layout (`from ...interface import ...`). It cannot run from
inside the deepagents repo.

So this directory is the **version-controlled source of truth**; running happens
by deploying it into a clbench checkout. This mirrors how `deepagents_harbor/`
is the deepagents-side integration code for the Harbor framework.

## Layout

```
deepagents_clbench/
├── README.md
├── sync_to_clbench.sh        # deploy the payload into a clbench checkout
└── system/                   # payload -> <clbench>/src/systems/deepagents/
    ├── __init__.py
    └── system.py             # DeepAgentsSystem
```

## Deploy & run

```bash
# 1. Deploy into a local clbench checkout
./sync_to_clbench.sh /path/to/continual-learning-bench

# 2. In the clbench checkout, ensure deepagents is installed in its env
uv add deepagents            # pulls langchain + langchain-anthropic too

# 3. Run
clbench run exploitable_poker --schedule quick_test --system deepagents
clbench run <task> --system deepagents --system-params model=anthropic:claude-opus-4-8
```

## How it learns

The benchmark scores improvement across a sequence of related instances. The
learning substrate is the agent's **persistent memory**, wired through
`create_deep_agent(memory=[...])` (i.e. `MemoryMiddleware`):

- Each turn, the configured memory files are loaded into the prompt (wrapped in
  `<agent_memory>` boundary markers, treated as untrusted reference data).
- New knowledge is written back by an explicit **reflection step** in
  `observe()`: at each completed instance the system distils the outcome into
  `AGENTS.md`. A one-shot decision agent won't reliably spend a tool call to
  update its own notes mid-decision, so the write is made deliberate (the same
  pattern clbench's `mem0`/`ace` systems use). The agent may also edit memory via
  its `edit_file` tool, but reflection guarantees it.

| File | Author | Purpose |
|---|---|---|
| `/memory/AGENTS.md` | reflection step (`observe()`) | distilled, generalizable strategy |
| `/memory/outcomes.md` | the framework (`observe()`) | bounded log of recorded reward/feedback |

Verified: in a 5-hand `quick_test` rollout, `AGENTS.md` grows from the seed to
~1.7 KB of distilled opponent reads (e.g. "value-bet strong hands; opponent
calls too wide; avoid weak unconnected hands"), captured per-step in the trace's
`system_memory.history`.

Both live in the in-state filesystem (`DeepAgentState["files"]`). The adapter
threads that filesystem from one `respond()` call to the next — this is what
makes the agent *continual* rather than one-shot. `reset()` clears it, so the
stateless baseline is genuinely stateless and `mean_gain` reflects only what was
learned.

## Notes

- **Backend / security**: uses the default in-state `StateBackend`, so the agent
  has no real shell or host filesystem access (its `execute` tool errors on a
  non-sandbox backend). If you swap in a shell-capable backend, scrub provider
  API keys from the environment first (see `deepagents_harbor`'s
  `_scrub_shell_env`), since the agent could otherwise read them.
- **Structured output**: each task supplies a per-turn `response_schema`; the
  adapter runs the agent for reasoning, then does a separate
  `with_structured_output` call to coerce the final answer into that schema.
- This directory is intentionally excluded from this project's `ruff`/`ty`
  config (it targets clbench's package layout, not deepagents'), matching how
  other external-benchmark code is handled in `libs/evals/pyproject.toml`.
