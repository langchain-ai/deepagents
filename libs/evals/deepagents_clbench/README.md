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

- Each turn, `/memory/AGENTS.md` is loaded into the prompt (wrapped in
  `<agent_memory>` boundary markers, treated as untrusted reference data).
- The **agent itself** distils and updates that file with its own `edit_file` /
  `write_file` tools as it learns — there is no separate reflection or
  extraction process. `observe()` only captures the latest outcome so the next
  turn's prompt can surface it; whether and how to record a lesson is the
  agent's decision.

| File | Author | Purpose |
|---|---|---|
| `/memory/AGENTS.md` | the agent (via `edit_file`) | its own distilled, generalizable strategy |

The file lives in the in-state filesystem (`DeepAgentState["files"]`); the
adapter threads it from one `respond()` call to the next — this is what makes
the agent *continual* rather than one-shot. `reset()` clears it, so the
stateless baseline is genuinely stateless and `mean_gain` reflects only what the
agent learned.

This means whether the agent maintains good notes is part of what's measured —
if it under-invests in memory, that's a real result, not something the harness
papers over.

## Notes

- **Backend / security**: uses the default in-state `StateBackend`, so the agent
  has no real shell or host filesystem access (its `execute` tool errors on a
  non-sandbox backend). If you swap in a shell-capable backend, scrub provider
  API keys from the environment first (see `deepagents_harbor`'s
  `_scrub_shell_env`), since the agent could otherwise read them.
- **Structured output**: each task supplies a per-turn `response_schema`; the
  agent emits it natively via `create_deep_agent(response_format=...)` (read
  from `structured_response`) — no separate extraction call. The agent is cached
  per schema and rebuilt only when the schema changes. Net result: one model
  interaction per turn.
- This directory is intentionally excluded from this project's `ruff`/`ty`
  config (it targets clbench's package layout, not deepagents'), matching how
  other external-benchmark code is handled in `libs/evals/pyproject.toml`.
