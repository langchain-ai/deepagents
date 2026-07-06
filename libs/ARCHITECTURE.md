# Deep Agents Architecture

This document explains how a deep agent is put together and where to look when you want to understand or change a behavior. It is primarily for new maintainers and contributors, but it should also be useful to readers who want a practical map of the system without reading the whole implementation first.

For setup and day-to-day commands, see [`DEVELOPMENT.md`](./DEVELOPMENT.md). For supported user-facing configuration, see the [Deep Agents docs](https://docs.langchain.com/oss/python/deepagents/overview) and the [`create_deep_agent()` API reference](https://reference.langchain.com/python/deepagents/graph/create_deep_agent).

- [The three layers](#the-three-layers) explains what Deep Agents adds on top of LangChain and LangGraph.
- [Construction and execution](#construction-and-execution) follows what happens before and during a run.
- [Middleware stack](#middleware-stack), [Tool surface and filesystem access](#tool-surface-and-filesystem-access), and [State and persistence](#state-and-persistence) cover the main implementation axes.
- [Common starting points](#common-starting-points) lists the code most maintainers inspect first.

## The three layers

Deep Agents sits on top of LangChain and LangGraph. Most questions become easier once you know which layer owns the behavior you are looking at.

```txt
Deep Agents      opinionated harness: defaults, middleware, backends, profiles
LangChain        agent abstraction: model + tools + middleware -> agent loop
LangGraph        runtime: state, checkpoints, streaming, interrupts
```

Starting from the bottom:

- **[LangGraph](https://docs.langchain.com/oss/python/langgraph/overview)** is the runtime. It runs the agent as a graph: a set of steps connected by transitions, where each step can read and update shared [state](https://docs.langchain.com/oss/python/langgraph/graph-api#state). LangGraph carries that state between steps, exposes [streaming](https://docs.langchain.com/oss/python/langgraph/event-streaming) APIs for observing a run as it happens, saves [checkpoints](https://docs.langchain.com/oss/python/langgraph/checkpointers), and pauses or resumes runs through [interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts).
- **[LangChain's `create_agent()`](https://docs.langchain.com/oss/python/langchain/agents)** is the agent abstraction on top of LangGraph. Callers describe an agent in terms of a [model](https://docs.langchain.com/oss/python/langchain/models), [tools](https://docs.langchain.com/oss/python/langchain/tools), and [middleware](https://docs.langchain.com/oss/python/langchain/middleware/overview). LangChain builds the loop that calls the model, executes tools, and repeats until the model finishes; the runtime layer provides durable execution.
- **Deep Agents** is an opinionated harness *on top of* `create_agent()`. It does not introduce a new runtime. Instead, [`create_deep_agent()`](https://reference.langchain.com/python/deepagents/graph/create_deep_agent) assembles the default middleware stack and configures [backends](https://docs.langchain.com/oss/python/deepagents/backends), [subagents](https://docs.langchain.com/oss/python/deepagents/subagents), [skills](https://docs.langchain.com/oss/python/deepagents/skills), [memory](https://docs.langchain.com/oss/python/deepagents/memory), and [profiles](https://docs.langchain.com/oss/python/deepagents/profiles).

The reason to start with Deep Agents instead of bare `create_agent()` is not that it is a different kind of runtime; it is that it packages the pieces most long-running agents need by default. The [LangChain overview](https://docs.langchain.com/oss/python/langchain/overview) has the full guidance on when to choose Deep Agents, LangChain, or LangGraph.

In this repository, most Deep Agents-specific code lives in three places: **middleware** adds harness behavior to the agent loop, **backends** decide where files, memory, and shell execution live, and **profiles** tune the harness for particular providers or models.

## Construction and execution

There are two phases: **construction**, when application code calls [`create_deep_agent()`](https://reference.langchain.com/python/deepagents/graph/create_deep_agent), and **execution**, when the returned graph is invoked.

### Construction

`create_deep_agent()` is the assembly point. At a high level, it:

1. Resolves the requested chat model and any provider or harness profile that applies to it,
2. Resolves the backend used by filesystem, skills, memory, and `execute` behavior,
3. Assembles the main-agent middleware stack,
4. Builds the default general-purpose subagent and any caller-provided subagents,
5. Composes the system prompt (from caller instructions, SDK defaults, and profile text), and
6. Calls LangChain's `create_agent(...)` to produce the runnable agent graph.

The exact parameter semantics and current middleware list are documented on `create_deep_agent()` itself and in [Customize Deep Agents](https://docs.langchain.com/oss/python/deepagents/customization).

### Execution

On each turn, LangGraph drives the agent loop. The model receives a message history, a system prompt, and a set of tools. It may respond directly or request tool calls. Tool results are appended to state, and the loop continues until the model produces a final response.

Deep Agents changes that loop mainly through middleware. Middleware can run before a model call, around a model call, around tool execution, or while state is being prepared. This lets the harness do things that plain tools cannot do, such as:

- Adding or removing tools from the current model request,
- Injecting filesystem, memory, skills, subagent, or human-in-the-loop instructions into the final system prompt,
- Summarizing/compacting or offloading message history as context grows,
- Storing typed values in graph state for later middleware or tools, and
- Enforcing filesystem permissions before a (built-in) filesystem tool runs.

A callable passed through `tools=` is different: it is only invoked after the model *chooses to use* that tool. It cannot rewrite the tool list or prompt before the model call. The docstring in `deepagents.middleware` explains this distinction in more detail.

## Middleware stack

Deep Agents installs scaffolding middleware first, inserts caller-supplied middleware in the middle, then appends tail behavior that depends on the final prompt and tool surface.

The mental model is:

- **Base scaffolding** creates the capabilities expected of a deep agent: planning, filesystem access, subagent delegation, summarization, and request cleanup.
- **Caller middleware** is where applications add their own behavior without rebuilding the harness from scratch.
- **Profile and tail middleware** tune the final request surface: provider-specific behavior, tool exclusions, prompt caching, memory injection, and human approval.

Subagents have their own middleware stacks. A behavior can therefore come from the main-agent stack, a declarative subagent stack, a compiled subagent supplied by the caller, or an async/remote subagent. If you are debugging a behavior that appears only during delegated work, check which subagent type handled the task before changing main-agent middleware.

For the exact default stack and ordering, use the [Customization docs](https://docs.langchain.com/oss/python/deepagents/customization#default-stack-main-agent) or `create_deep_agent()`'s `middleware` parameter docs.

## Tool surface and filesystem access

The model's visible tools are the result of several layers working together:

- Built-in middleware contributes the standard Deep Agents tools, including todo management, filesystem tools, and subagent delegation.
- Caller-provided `tools=` are added to that set.
- The resolved backend determines whether shell execution is available. If it is not, the `execute` tool is removed from the model request and shell-specific prompt text is omitted.
- Harness profiles can hide tools with `excluded_tools` when a provider or model needs a smaller or differently described surface.
- Filesystem permissions enforce path-level policy on built-in filesystem tools. Permissions are not a visibility mechanism: a model may still see a tool whose call can later be denied or interrupted.

This split matters when debugging access issues. If a tool is missing, look at middleware assembly and profile exclusions. If a tool is visible but fails, look at backend capability and permission enforcement. The detailed user-facing references are [Backends](https://docs.langchain.com/oss/python/deepagents/backends), [Permissions](https://docs.langchain.com/oss/python/deepagents/permissions), and [Profiles](https://docs.langchain.com/oss/python/deepagents/profiles).

## State and persistence

State lives in LangGraph. Deep Agents extends LangChain's `AgentState` with [`DeepAgentState`](https://reference.langchain.com/python/deepagents/graph/DeepAgentState), whose `messages` field uses a `DeltaChannel` reducer so checkpoint growth stays linear across long threads.

Middleware can contribute additional typed state fields through `state_schema`. Private middleware fields are tracked so internal state does not leak into subagents unexpectedly. If you are adding state for an application feature, prefer middleware-owned state when the state is only meaningful to that middleware; use a custom `state_schema` when callers need a graph-level field that tools or multiple middleware can share.

Persistence has two related but separate pieces:

- **Graph state and checkpoints** come from LangGraph. They preserve conversation state, message history, interrupts, and resumability.
- **Filesystem and memory persistence** come through Deep Agents backends. The default state backend is thread-scoped; store-backed or filesystem-backed routes can make files durable across threads or map them to disk or sandbox storage.

For long-term memory and backend routing patterns, use the [Memory](https://docs.langchain.com/oss/python/deepagents/memory) and [Backends](https://docs.langchain.com/oss/python/deepagents/backends) docs rather than duplicating setup recipes here.

---

## Common starting points

- Agent construction, middleware ordering, prompt assembly, default model behavior: `graph.py`.
- Tool visibility, prompt injection, and request-time behavior: `middleware/`.
- Filesystem persistence, shell support, and route behavior: `backends/`.
- Provider- or model-specific harness changes: `profiles/`.
- Public imports and compatibility expectations: `__init__.py` plus the API reference.

When in doubt, trace from the public argument on `create_deep_agent()` to the middleware or backend it installs, then follow how that component participates during execution.
