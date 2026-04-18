# Recursive REPL Mode (RLM)

`create_rlm_agent` is a wrapper over `create_deep_agent` that replaces
the default `general-purpose` subagent with one that has
[`REPLMiddleware`](../../libs/deepagents-repl) attached, then — for
`max_depth > 0` — adds a `deeper-agent` subagent pointing at another
whole agent graph built the same way, one level shallower.

The result: the general-purpose agent at any depth can delegate a
sub-task via `tools.task({ subagent_type: "deeper-agent", ... })` to a
structurally separate agent that itself has REPL + the same decision.
Recursion bottoms out at depth 0, whose general-purpose has REPL but
no `deeper-agent`.

## Why

A plain Deep Agent can delegate to a subagent via the `task` tool.
That's one call per subtask, serialized across model turns.

With `REPLMiddleware(ptc=True)` on the general-purpose subagent, the
agent can instead write:

```javascript
// inside one `eval` tool call on general-purpose
const results = await Promise.all([
  tools.task({ subagent_type: "deeper-agent", description: "subtask 1" }),
  tools.task({ subagent_type: "deeper-agent", description: "subtask 2" }),
  // ...
]);
```

One model turn kicks off the whole fan-out. Each `deeper-agent` call
lands on a freshly-built general-purpose one level down, which itself
has REPL and can fan out again until the chain bottoms out.

## Structure

```
root (depth=2)
├── general-purpose (REPL + PTC)
│   └── can task `deeper-agent` →
└── deeper-agent  (a full compiled depth-1 agent)
    ├── general-purpose (REPL + PTC)
    │   └── can task `deeper-agent` →
    └── deeper-agent  (a full compiled depth-0 agent)
        └── general-purpose (REPL + PTC, no deeper-agent peer)
```

Each `deeper-agent` entry is an independent compiled graph — not a
cycle. The system prompt at each level tells the model how much
recursion budget is left.

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

Extra subagents pass through untouched:

```python
agent = create_rlm_agent(
    tools=[lookup],
    subagents=[
        {"name": "writer", "description": "...", "system_prompt": "..."},
    ],
    max_depth=1,
)
```

What you cannot do: pass your own `general-purpose` spec. RLM manages
that subagent's middleware and system prompt at every depth; a
caller-provided override would break the recursion contract. The
helper raises `ValueError` if it finds one.

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
  sign that the task should be rethought.
- **State is not shared across recursion levels.** Each `deeper-agent`
  call runs in its own graph. Pass data through the `task` tool's
  `description` argument or let results flow back through tool
  return values.
- **Only the general-purpose subagent gets REPL.** If you want REPL
  on a custom subagent too, add `REPLMiddleware` to its `middleware`
  list yourself when you define it.
