# langchain-quickjs

`langchain-quickjs` provides a QuickJS-backed REPL middleware for Deep Agents. It adds a `repl` tool that can evaluate small JavaScript snippets for computation, control flow, JSON manipulation, and calls to exposed Python foreign functions.

## Basic usage

```python
from deepagents import create_deep_agent
from langchain_quickjs import QuickJSMiddleware


def normalize_name(name: str) -> str:
    return name.strip().lower()


agent = create_deep_agent(
    model="openai:gpt-4.1",
    tools=[],
    middleware=[
        QuickJSMiddleware(
            ptc=[normalize_name],
            add_ptc_docs=True,
        )
    ],
)
```

With this middleware installed, the agent receives a `repl` tool that runs each JavaScript evaluation in a fresh QuickJS context. If you expose Python callables through `ptc`, they are available inside the REPL as foreign functions.

## REPL behavior

- The REPL is stateless. Each call starts from a fresh QuickJS context, so variables, functions, and other values defined in one `repl` call are not available in the next one.
- Execution uses [QuickJS](https://bellard.org/quickjs/), so JavaScript support is limited to what QuickJS provides. It is good for small computations, control flow, JSON manipulation, and calling exposed foreign functions, but it is not a browser or Node.js runtime and does not provide their APIs.
- Foreign functions support passing primitive values between JavaScript and Python, including `int`, `float`, `bool`, `str`, and `None`. Lists and dictionaries returned from Python are also supported and are bridged back into JavaScript arrays and objects.
- Async foreign functions are supported. Because QuickJS callbacks are synchronous, awaitables are delegated to a dedicated daemon-thread event loop and their resolved results are returned back into the REPL call.

## Current limitations

- Does not work with HIL in the REPL.
- Does not support `ToolRuntime` yet.
