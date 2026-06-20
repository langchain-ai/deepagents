# Architecture

This document maps how a deep agent is put together and what happens when it runs, so that when you see the agent do something you can find the code responsible for it. It is written for people working on the `deepagents` package itself rather than for people building agents *with* it — for that, see the [user documentation](https://docs.langchain.com/oss/python/deepagents/overview). For setup and day-to-day commands, see [`DEVELOPMENT.md`](./DEVELOPMENT.md).

## The three layers

Deep Agents is the top layer of a three-layer stack. Each layer owns a different concern, and understanding which layer a behavior lives in is the fastest way to find the code responsible for it.

```txt
┌─────────────────────────────────────────────────────────────┐
│ deepagents          opinionated harness                     │
│                     create_deep_agent() assembles a default │
│                     middleware stack, backends, subagents,  │
│                     skills, and profiles.                   │
├─────────────────────────────────────────────────────────────┤
│ langchain           agent loop                              │
│ create_agent()      turns a model + tools + middleware into │
│                     a runnable agent graph.                 │
├─────────────────────────────────────────────────────────────┤
│ langgraph           runtime                                 │
│                     state channels, checkpointing,          │
│                     streaming, persistence, interrupts.     │
└─────────────────────────────────────────────────────────────┘
```

- **LangGraph** is the runtime. It owns the compiled state graph, state channels, checkpointing, streaming, and human-in-the-loop interrupts. Deep Agents never re-implements any of this.
- **LangChain `create_agent`** is the agent loop. It takes a model, a tool list, and a list of middleware and produces a `CompiledStateGraph` that runs the model/tool loop.
- **Deep Agents** is an opinionated harness *on top of* `create_agent`. It does not introduce a new runtime; instead, `create_deep_agent()` assembles a curated middleware stack, wires in pluggable backends, and configures subagents, skills, memory, and profiles, then hands all of it to `create_agent`.

The practical consequence: most Deep Agents behavior is implemented as **middleware** and **backends**, not as bespoke graph code. When debugging, ask "which middleware owns this?" before reaching for the runtime.

## Request flow

There are two distinct phases: **construction** (once, when you call `create_deep_agent`) and **execution** (every turn, when the agent runs).

### Construction

`create_deep_agent()` (in `deepagents/graph.py`) does the assembly:

1. Resolves the model and the active harness/provider **profile**.
2. Resolves the **backend** (filesystem / shell / store) used by the filesystem, skills, and memory middleware.
3. Builds the **middleware stack** (see ordering below).
4. Builds the general-purpose subagent (and any inline subagents) with their own middleware stacks.
5. Composes the final system prompt (caller prompt + base prompt + profile suffix).
6. Calls `create_agent(...)`, which compiles everything into a LangGraph `CompiledStateGraph`.

The return value is a LangGraph `CompiledStateGraph`. That means you invoke, stream, checkpoint, and resume it exactly like any other LangGraph graph.

### Execution

On each turn, LangGraph drives the agent loop. Middleware participate via hooks; the most important is `wrap_model_call()`, which **intercepts every LLM request** before it is sent. This is how middleware can:

- filter the tool list per call (e.g. `FilesystemMiddleware` drops the `execute` tool when the backend has no shell),
- inject system-prompt context (e.g. `MemoryMiddleware`, `SkillsMiddleware`),
- transform message history (e.g. `SummarizationMiddleware` truncates or summarizes when the context window fills), and
- read/write typed state that persists across turns.

A plain tool in the `tools=[]` list cannot do any of this — it is only invoked *by* the model, never *before* the model call. That distinction (middleware vs. plain tool) is the core extensibility decision and is documented inline in `deepagents/middleware/__init__.py`.

## Middleware stack ordering

`create_deep_agent` assembles the stack in a fixed order. User-supplied `middleware` is inserted between the base stack and the tail stack. This ordering is authoritative and is kept in sync with the `create_deep_agent` docstring (the source of truth):

Base stack:

- `TodoListMiddleware`
- `SkillsMiddleware` (if `skills` is provided)
- `FilesystemMiddleware`
- `SubAgentMiddleware` (if inline subagents are available)
- `SummarizationMiddleware`
- `PatchToolCallsMiddleware`
- `AsyncSubAgentMiddleware` (if async subagents are provided)

*User-supplied `middleware` is inserted here.*

Tail stack:

- Harness profile `extra_middleware` (if any)
- tool-exclusion middleware (if the profile excludes tools)
- `AnthropicPromptCachingMiddleware` (unconditional; no-ops for non-Anthropic models)
- `MemoryMiddleware` (if `memory` is provided)
- `HumanInTheLoopMiddleware` (if `interrupt_on` is provided)

After assembly, any entries listed in the profile's `excluded_middleware` are filtered out. A scaffolding subset (e.g. `FilesystemMiddleware`, `SubAgentMiddleware`) is protected and cannot be excluded.

## State and persistence

State lives in LangGraph, not in Deep Agents. `DeepAgentState` extends LangChain's `AgentState` and wraps `messages` in a `DeltaChannel` so checkpoint growth stays O(N) instead of O(N^2) across a long thread. Middleware contribute additional typed state fields via their `state_schema`; private fields are tracked so subagents don't leak internal state.

## Extension points

These are the supported, public ways to customize a Deep Agent without forking. Anything exported from a package's `__init__.py` is public; modules and symbols prefixed with an underscore (`_tools.py`, `_models.py`, `_api/`, `_excluded_middleware.py`, etc.) are internal and may change without notice.

| Want to change... | Use | Where |
| --- | --- | --- |
| The model | `model=` | `create_deep_agent` |
| Add capabilities the model calls | `tools=` (plain tools) | `create_deep_agent` |
| Intercept/transform requests | `middleware=` (`AgentMiddleware`) | `deepagents.middleware` |
| Delegate to isolated agents | `subagents=` (`SubAgent`, `CompiledSubAgent`, `AsyncSubAgent`) | `deepagents` |
| Where files/shell/state live | `backend=` (`BackendProtocol`) | `deepagents.backends` |
| Reusable loadable behaviors | `skills=` | `deepagents.middleware.skills` |
| Cross-session recall | `memory=` | `deepagents.middleware.memory` |
| Approve/edit/reject tool calls | `interrupt_on=` / `permissions=` | `create_deep_agent` |
| Model/harness defaults | profiles | `deepagents.profiles` |

Backends are pluggable via `BackendProtocol`; built-ins include `StateBackend`, `FilesystemBackend`, `StoreBackend`, `LocalShellBackend`, `CompositeBackend`, `ContextHubBackend`, and `LangSmithSandbox`. Sandbox providers (Daytona, Modal, Vercel, Runloop, QuickJS) ship as separate partner packages under `libs/partners/`.

## Where things live

```txt
libs/deepagents/deepagents/
├── graph.py            # create_deep_agent: assembly + middleware ordering
├── __init__.py         # public SDK surface
├── middleware/         # the default middleware stack (+ overview docstring)
├── backends/           # pluggable file/shell/state backends + BackendProtocol
├── profiles/           # harness and provider profiles (defaults per model)
├── _tools.py           # internal helpers (underscore = not public)
├── _models.py          # internal model resolution
└── _api/               # internal deprecation helpers
```

For the relationship between Deep Agents, LangChain, and LangGraph, see the [LangChain ecosystem overview](https://docs.langchain.com/oss/python/concepts/products).
