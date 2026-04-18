# Recursive REPL Mode (RLM)

`create_rlm_agent` is a tiny wrapper over `create_deep_agent` that adds a
[`REPLMiddleware`](../../libs/deepagents-repl) with **programmatic tool
calling (PTC)** at every level of a nested subagent chain, down to a
caller-chosen `max_depth`.

## Why

A plain Deep Agent can delegate to a subagent via the `task` tool. That's
one call per subtask. If the task decomposes into ten independent
subtasks, the top-level agent issues ten `task` calls sequentially and
pays the round-trip cost on each.

With `REPLMiddleware(ptc=True)`, the agent can instead write:

```javascript
// inside one `eval` tool call
const results = await Promise.all([
  tools.task({ description: "subtask 1" }),
  tools.task({ description: "subtask 2" }),
  // ...
]);
```

One model turn kicks off the whole fan-out. The `recursive` subagent
`create_rlm_agent` injects has the same REPL + PTC, so the decomposition
can continue at the next level — which is the whole point of the
"recursion" in the name.

## Structure

```
YourAgent (depth 0)
├── eval + PTC over [add, task]     ← one eval can Promise.all()
└── task → recursive subagent
    ├── eval + PTC over [add, task]  ← depth 1 can also fan out
    └── task → recursive subagent
        ├── eval + PTC                ← depth 2 (leaf when max_depth=2)
        └── (no recursive child)
```

Every level shares the same tools and shares the same subagent set,
plus a synthetic `recursive` subagent pointing at the depth-N-1 build.

## Usage

```python
from rlm_agent import create_rlm_agent
from langchain_core.tools import tool

@tool
def lookup(key: str) -> str:
    """Fetch a value by key."""
    ...

agent = create_rlm_agent(
    model="claude-sonnet-4-6",
    tools=[lookup],
    max_depth=2,
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Look up A, B, C in parallel."}],
})
```

## Running the demo

```bash
uv run python rlm_agent.py
# custom task:
uv run python rlm_agent.py "Use eval to add 1+2 and 3+4 in parallel."
# deeper recursion:
uv run python rlm_agent.py --max-depth 2
```

## Tradeoffs

- **Parallelism is cheap, depth isn't.** Each recursion level builds a
  full Deep Agent graph. `max_depth=2` is plenty for most decomposition
  patterns; `max_depth=5+` is a sign the task should be rethought.
- **State isn't shared across recursion levels.** Each subagent runs in
  its own graph. Pass data through the `task` tool's `description`
  argument or let results flow back through tool return values.
- **Every level inherits the full tool set.** This is the point — but
  if a dangerous tool shouldn't reach depth 2, wire your own
  `SubAgent` entry that filters tools and hand it to `create_rlm_agent`
  via the `subagents` kwarg.
