# langchain-quickjs

A [`deepagents`](../deepagents) middleware that gives an agent a persistent, sandboxed **JavaScript REPL** tool, backed by [`quickjs-rs`](../../../quickjs-wasm) (QuickJS embedded via PyO3 + rquickjs).

Instead of issuing N serial tool calls, the model can write one block of JavaScript that orchestrates work in-loop — variables and functions defined in one call survive into the next, `Promise.all` runs concurrent work, configured subagents are dispatchable as `await task({...})`, and (opt-in) agent tools are callable from inside the REPL as `await tools.<name>(...)`.

```python
from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

agent = create_deep_agent(
    model="claude-sonnet-4-6",
    middleware=[CodeInterpreterMiddleware()],
)
```

- [Why](#why)
- [Install](#install)
- [Quick start](#quick-start)
- [What the REPL is](#what-the-repl-is)
  - [Persistence](#persistence)
  - [Sandbox](#sandbox)
  - [Console capture](#console-capture)
  - [Timeouts and memory](#timeouts-and-memory)
  - [Result formatting](#result-formatting)
- [Dispatching subagents (`task`)](#dispatching-subagents-task)
- [Programmatic tool calling (PTC)](#programmatic-tool-calling-ptc)
- [Configuration reference](#configuration-reference)
- [Errors the model can see](#errors-the-model-can-see)
- [License](#license)

## Why

Tool calling is fine for small, discrete requests. It falls apart when the model needs to:

- loop over a list and call a tool per item
- run two independent tool calls concurrently
- compute something between calls (aggregate, filter, dedupe, format)
- reuse intermediate state across several turns

Each of those currently costs one round-trip to the model per step. With a REPL, all of it happens in one `eval` call. This enables [programmatic tool calling](#programmatic-tool-calling-ptc) where the model writes JavaScript that invokes the agent's own tools.

## Install

```bash
uv add langchain-quickjs
```

`langchain-quickjs` depends on `quickjs-rs`, a PyO3 extension module that ships prebuilt wheels for macOS, Linux, and Windows on CPython 3.11+.

## Quick start

```python
from deepagents import create_deep_agent
from langchain_quickjs import CodeInterpreterMiddleware

agent = create_deep_agent(
    model="claude-sonnet-4-6",
    middleware=[CodeInterpreterMiddleware()],
)

# Use `ainvoke` — PTC bridges register as async QuickJS host functions,
# and sync `invoke` on a REPL with async bridges raises ConcurrentEvalError.
result = await agent.ainvoke({"messages": [{"role": "user", "content": "..."}]})
```

The middleware:

1. registers an `eval` tool (configurable name) that runs JS in a persistent context;
2. appends a short system-prompt snippet explaining the tool's semantics (sandbox, timeout, memory limit);
3. gives every LangGraph `thread_id` its own QuickJS `Runtime`, so two conversations can't see each other's globals.

## What the REPL is

### Persistence

The REPL is module-flavoured: top-level `let`/`const`/`function` persist across `eval` calls in the same thread. The `mode` setting controls how far that persistence reaches:

- `mode="thread"` (default) — state persists across calls **and** across turns in the same LangGraph `thread_id`.
- `mode="turn"` — state persists across calls within a turn only.
- `mode="call"` — each `eval` call runs in a fresh REPL.

```js
// call 1
const fib = (n) => (n < 2 ? n : fib(n - 1) + fib(n - 2));

// call 2
fib(10)  // 55
```

### Sandbox

The REPL runs in a QuickJS context with **no ambient capabilities**. There is no filesystem, no network, no `fetch`, no `require`, no real clock (`Date.now()` is whatever QuickJS provides, not wall-clock for security-sensitive uses), no `process`, no `import` of anything you didn't explicitly install.

Capabilities can be explicitly added using **PTC** to call into the agent's own tools (see below).

### Console capture

`console.log` / `console.warn` / `console.error` are captured by default and returned as a `<stdout>` block alongside the result, separately truncated. Disable with `capture_console=False` if you'd rather the guest see no `console` at all.

```js
console.log("hi", 2);
1 + 1
```

```xml
<stdout>
hi 2
</stdout>
<result>2</result>
```

### Timeouts and memory

Each call has a per-call wall-clock timeout (default 5 s). Breaching it produces:

```xml
<error type="Timeout">...</error>
```

The runtime has a shared memory limit across every context under it (default 64 MiB). OOM surfaces as:

```xml
<error type="OutOfMemory">...</error>
```

PTC host-function calls are also budgeted per eval call (default 256 `tools.*`
invocations). Exceeding the budget surfaces as:

```xml
<error type="PTCCallBudgetExceeded">...</error>
```

Set `max_ptc_calls=None` only in trusted environments. Disabling the
budget allows unbounded PTC-call loops and increases DoS risk.

Top-level `await` works on the async path — the promise settles before the call returns. An un-resolvable top-level promise (no host work in flight, no resolver) surfaces as `<error type="Deadlock">`.

### Result formatting

Every eval renders into one wire format consumed by the model:

| Outcome | Rendered as |
| --- | --- |
| Marshalable value | `<result>{json-ish}</result>` |
| Function or unmarshalable | `<result kind="handle">[Function] arity=2</result>` |
| JS-level throw | `<error type="TypeError">{message}\n{stack}</error>` |
| Timeout / deadlock / OOM | `<error type="Timeout" \| "Deadlock" \| "OutOfMemory">...</error>` |
| `console.*` output | separate `<stdout>...</stdout>` block |

Results and stdout are independently truncated to `max_result_chars` (default 4000) before being sent back to the model.

Numeric rendering follows Node's REPL convention — whole-valued floats (`42.0`) render as integers (`42`) so the model isn't confused by JS's single numeric type.

## Dispatching subagents (`task`)

When the host agent is a Deep Agents agent that has a `task` tool, the middleware exposes a top-level `task(...)` primitive inside the REPL (on by default; disable with `subagents=False`). The model dispatches a configured subagent and orchestrates the rest — fan-out, filtering, multi-stage flow, synthesis — in plain JavaScript:

```js
const result = await task({
  description: "Review src/auth.ts for SQL injection. Cite line numbers.",
  subagentType: "reviewer",          // configured subagent name (required)
  responseSchema: { /* JSON Schema */ }, // optional structured output
});
```

`task` runs a full agentic loop for the selected subagent and resolves to its final result. With `responseSchema`, the resolved value is already a typed JS value matching the schema. Each dispatch is stateless — `description` is the only prompt the subagent receives, so make it self-contained.

Because the model holds intermediate state in JS, it can fan out concurrently (the bridge caps concurrency per REPL), filter results between passes, and merge everything in one `eval` call instead of one model round-trip per subagent.

This is distinct from PTC: `task` is always a top-level global (never under `tools`), and it is available whenever the host exposes a Deep Agents `task` tool — independent of the `ptc=` allowlist.

```js
// One eval call: classify in parallel, then deep-review only the risky files.
const tagged = await Promise.all(files.map(async (f) => ({
  file: f,
  ...(await task({
    description: `Classify ${f} as handler, util, test, or config.`,
    subagentType: "reviewer",
    responseSchema: {
      type: "object",
      properties: { kind: { type: "string" }, risky: { type: "boolean" } },
      required: ["kind", "risky"],
    },
  })),
})));

const reviews = await Promise.all(
  tagged.filter((t) => t.kind === "handler" && t.risky).map(async (t) => ({
    file: t.file,
    review: await task({
      description: `Deep security review of ${t.file}. Cite line numbers.`,
      subagentType: "reviewer",
    }),
  })),
);
```

> **Approval:** `task(...)` runs inside an already-approved `eval` invocation and does **not** trigger parent-level `interrupt_on` / HITL approval per dispatch. Gate the `eval` tool itself, add approval middleware inside subagent specs, or set `subagents=False` if per-dispatch parent approval is required.

## Programmatic tool calling (PTC)

PTC is the reason to use this middleware over a plain code-interpreter tool. When configured, each exposed tool is available inside the REPL as:

```ts
async tools.<camelCaseName>(input: {...}): Promise<string>
```

So an agent with a `search_web` tool and a `summarize` tool can do:

```js
const results = await Promise.all([
  tools.searchWeb({ query: "deepagents" }),
  tools.searchWeb({ query: "quickjs" }),
]);

await tools.summarize({ text: results.join("\n\n") })
```

...in **one** `eval` call — three tool invocations, zero round-trips to the model between them.

### Enabling it

```python
CodeInterpreterMiddleware()                              # disabled (default)
CodeInterpreterMiddleware(ptc=["search_web"])            # explicit allowlist
CodeInterpreterMiddleware(ptc=[search_tool])             # explicit tool object allowlist
```

The REPL's own tool is always excluded from PTC; `tools.eval("tools.eval(...)")` would be pointless recursion, and if the model wants nested code it can just write nested code in one call.

### What the model sees

When PTC is on, the system-prompt snippet grows an *API Reference — `tools` namespace* section listing every exposed tool as a TypeScript-ish signature derived from the tool's args schema:

```ts
/** Search the web for the given query. */
async tools.searchWeb(input: {
  /** The query string. */
  query: string;
  /** Max results. */
  limit?: number;
}): Promise<string>
```

Enums, `anyOf` unions, nested objects, and arrays are all supported by the schema renderer. Opaque types fall back to `Record<string, unknown>` — the description is usually enough.

### How it works (so you can debug it)

- Each PTC-exposed tool gets a QuickJS host-function bridge registered under a generated `__tools_*` global symbol. The bridge is async, so the guest sees `tools.x(...)` as returning a `Promise`.
- `globalThis.tools` is rebuilt every turn from the currently-exposed name set. So if an upstream middleware filters tools on a per-turn basis, the `tools` namespace follows along.
- When the bridge invokes a tool, it forwards the `ToolRuntime` captured from the outer `eval` call — so the tool sees graph `state`, `store`, `context`, and a synthesised child `tool_call_id`. (Subagent dispatch has its own top-level `task(...)` primitive — see [Dispatching subagents](#dispatching-subagents-task) — and is not routed through the `tools` namespace.)
- Tool return values are coerced to strings: strings pass through, `ToolMessage`s get unwrapped, a `Command` has its last-message content extracted, everything else gets `json.dumps`'d.

## Configuration reference

```python
CodeInterpreterMiddleware(
    memory_limit=64 * 1024 * 1024,  # bytes, shared across contexts
    timeout=5.0,                     # per-call seconds
    max_ptc_calls=256,     # per-eval `tools.*` bridge calls, None disables (DoS risk)
    tool_name="eval",                # what the model calls it
    max_result_chars=4000,           # result/stdout truncation, each
    capture_console=True,            # install console.log/warn/error bridge
    subagents=True,                  # expose global `task(...)` when host has a Deep Agents task tool
    mode="thread",                   # "thread" | "turn" | "call"
    max_snapshot_bytes=None,         # defaults to `memory_limit`; larger snapshots are dropped
    ptc=None,                        # None | list[str] | list[BaseTool]
)
```

## Errors the model can see

| Type | Cause |
| --- | --- |
| `SyntaxError`, `TypeError`, `ReferenceError`, ... | User-code error. Re-surfaces the JS error name verbatim. |
| `Timeout` | Call exceeded `timeout=`. |
| `OutOfMemory` | Runtime hit `memory_limit=`. |
| `PTCCallBudgetExceeded` | Uncaught `tools.*` call-budget overflow in one eval (`max_ptc_calls=`). |
| `Deadlock` | Top-level promise never resolved with no async host work in flight. |
| `ConcurrentEval` | Shouldn't happen under locks; defensive mapping for QuickJS `ConcurrentEvalError`. |

`asyncio.CancelledError` propagates out cleanly when JS declines to catch a `HostCancellationError` — so LangGraph cancellation semantics work end-to-end.

## License

MIT. See [`LICENSE`](LICENSE)

## Resources

- [LangChain Academy](https://academy.langchain.com/) — Comprehensive, free courses on LangChain libraries and products, made by the LangChain team.
- [Code of Conduct](https://github.com/langchain-ai/langchain/?tab=coc-ov-file) — community guidelines and standards
