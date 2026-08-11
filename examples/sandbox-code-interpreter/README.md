# Sandbox Code Interpreter

Moves data **both ways** across the host/sandbox boundary in a single JavaScript
REPL call: a host tool's output becomes a file in a remote LangSmith sandbox,
Python in the sandbox analyzes it, and the file that Python writes is fed back
into a second host tool.

## The problem it solves

A tool that runs on the host returns data to the *model*, not to the sandbox —
nothing in the sandbox can see it. The reverse is also true: a file a sandbox
process writes is invisible to a host tool. Files are the only crossing, and
each crossing normally costs a model round trip.

Programmatic tool calling (PTC) collapses the whole chain into one call:

```js
// host -> sandbox
const raw = await tools.generateReadings({ count: 200 });          // host
await tools.writeFile({ file_path: `${dir}/readings.json`, content: raw });
await tools.writeFile({ file_path: `${dir}/analyze.py`, content: script });
await tools.execute({ command: `python3 ${dir}/analyze.py` });     // sandbox

// sandbox -> host
const summary = await tools.readFile({ file_path: `${dir}/summary.json` });
await tools.renderReport({ summary_json: strip(summary) });        // host
```

## How it works

`CodeInterpreterMiddleware` runs QuickJS inside the host process. The REPL has
no ambient capabilities — no filesystem, no network, no `require`. Its only
route outward is PTC: tools named in `ptc=[...]` are exposed as
`tools.<camelCase>(input) => Promise<...>`.

Those names resolve against the *live* tool registry, which mixes both worlds:

- `generate_readings` and `render_report` are plain `@tool`s on the host.
- `write_file` / `read_file` / `ls` / `execute` come from `backend=sandbox`, so
  they act on the remote filesystem.

```python
agent = create_deep_agent(
    model="openai:gpt-5.6-luna",
    tools=[generate_readings, render_report],
    backend=LangSmithSandbox(sandbox),
    middleware=[
        CodeInterpreterMiddleware(
            tool_name="js_eval",
            ptc=[
                "generate_readings",
                "render_report",
                "write_file",
                "read_file",
                "ls",
                "execute",
            ],
        )
    ],
)
```

The JS glues them together without the 200 readings ever passing back through
the model's context — only the final rendered table does.

## Trust boundary

Two properties matter when combining a sandbox with the interpreter:

- **The JS runs on the host, not in the sandbox.** Only the PTC calls cross into
  the remote. QuickJS has no capabilities of its own, so the blast radius is
  whatever you allowlist — but this is not sandbox isolation for the JS itself.
- **PTC bypasses HITL.** PTC calls go through the host-function bridge, not the
  normal `ToolNode` path, so `interrupt_on` does not fire per call. This example
  allowlists `write_file` and `execute`, which means the REPL writes to and runs
  commands in the sandbox **unprompted**. That is the point of the demo, and it
  is a deliberate trade: gate the `js_eval` tool itself if you need approval.

`deepagents-code` (the `dcode` CLI) rejects this combination outright —
`enable_interpreter=True` with a remote sandbox raises `ValueError` for exactly
these reasons. This example uses base `deepagents`, which has no such guard.

## Running it

Requires a LangSmith API key with sandbox access, and an OpenAI API key.

```bash
uv sync
```

```bash
export LANGSMITH_API_KEY=... OPENAI_API_KEY=... && uv run agent.py
```

The sandbox boots from the default image (Python 3.12, no snapshot) and is
deleted on exit. Expected output:

```
SENSOR    COUNT     MEAN      MIN      MAX
------------------------------------------
kiln         67    74.13    47.75   101.79
press        55    71.13    39.78    99.08
dryer        78    70.78    41.89    94.26
```

## Notes

- PTC returns each tool's **human-readable text output**, not structured data.
  `read_file` is line-numbered (`1  {"name": ...}`) and `ls` returns a Python
  list literal (`['a.json']`) — both break `JSON.parse`. The system prompt
  describes these formats; without those hints the model burns three to five
  calls rediscovering them.
- PTC functions take a **single object argument**. `tools.glob("*.json", dir)`
  fails with `_bridge() takes from 0 to 1 positional arguments but 2 were given`.
