# Recursive REPL Mode (RLM)

`create_rlm_agent` is a wrapper over `create_deep_agent` that adds
[`REPLMiddleware`](../../libs/partners/quickjs) to the agent and — for
`max_depth > 0` — replaces the default `general-purpose` subagent
with a `CompiledSubAgent` whose runnable is a depth-(N-1) RLM agent.

The model delegates the normal way, via
`tools.task({subagent_type: "general-purpose", ...})`. Behind the
scenes the call lands on a fully realized deeper agent that itself
has REPL + a compiled `general-purpose`, until the chain bottoms out
at depth 0, where `general-purpose` is the plain built-in.

## Why

A plain Deep Agent can delegate to a subagent via the `task` tool.
That's one call per subtask, serialized across model turns.

With `REPLMiddleware(ptc=True)` on the agent, the model can write:

```javascript
// inside one `eval` tool call
const results = await Promise.all([
  tools.task({ subagent_type: "general-purpose", description: "subtask 1" }),
  tools.task({ subagent_type: "general-purpose", description: "subtask 2" }),
  // ...
]);
```

One model turn kicks off the whole fan-out. Each `general-purpose`
call lands on a freshly-built deeper agent that itself has REPL and
can fan out again.

## Structure

```
root (depth=2, has REPLMiddleware)
└── general-purpose → compiled depth-1 graph (has REPLMiddleware)
    └── general-purpose → compiled depth-0 graph (has REPLMiddleware)
        └── general-purpose → built-in default (no REPL, no recursion)
```

Each `general-purpose` entry at depth > 0 is an independent compiled
graph — not a cycle. At depth 0 we leave `general-purpose` to
`create_deep_agent`'s auto-injection, so it ends the chain.

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

Extra subagents pass through to every level:

```python
agent = create_rlm_agent(
    tools=[lookup],
    subagents=[
        {"name": "writer", "description": "...", "system_prompt": "..."},
    ],
    max_depth=1,
)
```

What you cannot do: pass your own `general-purpose` spec. RLM
manages that subagent at every depth > 0; a caller-provided override
would break the recursion contract. The helper raises `ValueError`
if it finds one.

## Running the demo

```bash
uv run python rlm_agent.py
# custom task:
uv run python rlm_agent.py "Use eval to add 1+2 and 3+4 in parallel."
# deeper recursion:
uv run python rlm_agent.py --max-depth 2
```

## Tradeoffs

- **Each recursion level builds a full Deep Agent graph.** `max_depth=2`
  is plenty for most decomposition patterns; deeper tends to be a
  sign the task should be rethought.
- **State is not shared across recursion levels.** Each
  `general-purpose` call runs in its own graph. Pass data through
  the `task` tool's `description` argument or let results flow back
  through tool return values.
- **Only the root agent has REPL at each level.** If you want REPL
  on a non-`general-purpose` subagent too, add `REPLMiddleware` to
  its `middleware` list yourself when you define it.
