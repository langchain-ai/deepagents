---
type: architecture
title: SDK Construction and Execution
description: Trace how create_deep_agent resolves configuration into a LangChain-compiled LangGraph agent and how Deep Agents middleware, backends, state, delegation, persistence, interrupts, and streams participate at execution time.
tags: [deepagents, create_deep_agent, langchain, langgraph, middleware, subagents, streaming, state]
verified:
  - by: openwiki/0.4.2
    at: 2026-09-15T08:05:27.526Z
sources:
  - id: openwiki-source-68ae2141dbec1e0915410ac3
    resource: repo://libs/ARCHITECTURE.md
  - id: openwiki-source-fd64c1b88759a3b897a5452c
    resource: repo://libs/deepagents/deepagents/__init__.py
  - id: openwiki-source-b93533cac55718d75277d1cf
    resource: repo://libs/deepagents/deepagents/_excluded_middleware.py
  - id: openwiki-source-822ae989625ba99d4c7cc08b
    resource: repo://libs/deepagents/deepagents/_messages_reducer.py
  - id: openwiki-source-50173942904153d619b9ae0d
    resource: repo://libs/deepagents/deepagents/_models.py
  - id: openwiki-source-e7c7a0d6e6f2fa82362f1c56
    resource: repo://libs/deepagents/deepagents/_tools.py
  - id: openwiki-source-07f9eac13e71bcbdb4e6994b
    resource: repo://libs/deepagents/deepagents/backends/state.py
  - id: openwiki-source-0fc0e47059e4d07e23e50be2
    resource: repo://libs/deepagents/deepagents/graph.py
  - id: openwiki-source-7a16b9a53a07e882b7305459
    resource: repo://libs/deepagents/deepagents/middleware/_prompt_caching.py
  - id: openwiki-source-421bc4b065189ae1165ca326
    resource: repo://libs/deepagents/deepagents/middleware/_state.py
  - id: openwiki-source-e51c4102234507d1529a2440
    resource: repo://libs/deepagents/deepagents/middleware/async_subagents.py
  - id: openwiki-source-114a1c7a58992fa867a94ef0
    resource: repo://libs/deepagents/deepagents/middleware/subagents.py
  - id: openwiki-source-59612eea63cbfafbd628feda
    resource: repo://libs/deepagents/deepagents/profiles/harness/harness_profiles.py
  - id: openwiki-source-a8ed6d2b681c0b2af3bf4699
    resource: repo://libs/deepagents/tests/unit_tests/test_deep_agent_streaming.py
  - id: openwiki-source-6d183faf1a4bc5a5ba451aba
    resource: repo://libs/deepagents/tests/unit_tests/test_graph.py
  - id: openwiki-source-dc64f28a66d10932b86fcd61
    resource: repo://libs/deepagents/tests/unit_tests/test_messages_reducer.py
generated: { by: "openwiki/0.4.2", at: "2026-09-15T08:05:27.526Z" }
---

# SDK Construction and Execution

`create_deep_agent` is Deep Agents' public assembly API. It is re-exported from `deepagents`, resolves Deep Agents defaults and policy, then delegates graph compilation to LangChain `create_agent()`. The result is a configured LangGraph `CompiledStateGraph`, not a separate Deep Agents runtime: LangChain owns the agent loop and LangGraph owns state transitions, checkpointing, interrupts, and streaming. Deep Agents owns the opinionated construction of the prompt, middleware, backend wiring, and delegation surface.

## Construction and execution sequence

```mermaid
sequenceDiagram
    participant App as Application
    participant Builder as create_deep_agent
    participant Policy as Model and profile resolution
    participant Stack as Middleware and subagent assembly
    participant LC as LangChain create_agent
    participant Graph as Configured LangGraph
    participant Chat as Chat model
    participant Surface as Tool middleware

    App->>Builder: model tools backend and options
    Builder->>Policy: resolve model and select harness profile
    Policy-->>Builder: model and construction policy
    Builder->>Stack: compose prompt backend subagents and middleware
    Stack-->>Builder: graph inputs and state policy
    Builder->>LC: model prompt tools middleware and runtime services
    LC-->>Graph: compiled graph
    Builder-->>App: graph with Deep Agents config
    App->>Graph: invoke or stream events
    loop Until no tool calls
        Graph->>Chat: prompt messages and effective tools
        Chat-->>Graph: final response or tool calls
        Graph->>Surface: execute selected tool calls
        Surface-->>Graph: results and state updates
    end
    Graph-->>App: final state or stream projections
```

Caption: construction stops at LangChain compilation; the configured LangGraph later drives model/tool turns while middleware shapes each request and tool execution.

## Input resolution and policy

### Model and profile selection

A supplied `BaseChatModel` passes through `resolve_model` unchanged. A string model spec is initialized through `init_chat_model`, after a registered provider profile contributes provider-specific initialization settings. The original string spec is retained for harness-profile lookup; otherwise lookup can inspect the resolved model. This separates **provider profiles** (model construction) from **harness profiles** (post-construction agent behavior). Harness profiles can contribute prompt text, tool-description rewrites and exclusions, extra middleware, default-general-purpose-subagent settings, and middleware exclusions.

`model=None` remains a deprecated compatibility path: it warns and creates `ChatAnthropic(model_name="claude-sonnet-4-6")`. Pass an explicit model instead. Each declarative subagent resolves its own model and harness profile, so it can intentionally have provider-appropriate policy different from its parent.

Tool-description overrides are applied to copies of dictionary tools and `BaseTool` instances, never to caller-owned objects; plain callables are not wrapped or rewritten. Tool *exclusion* is deliberately later and runtime-oriented: the final `_ToolExclusionMiddleware` filters the model request after custom middleware has had a chance to inject tools.

### Backend and authored prompt

Absent `backend=`, the builder creates a single `StateBackend()` and gives that instance to the main and constructed-subagent filesystem, skills, memory, and summarization middleware. `StateBackend` is not a standalone file store: inside graph execution it reads and queues partial writes through LangGraph configuration keys and the `files` state channel. Its files last within a checkpointed conversation thread, not across threads; calls outside a graph context fail with `RuntimeError`. Initialize state-backed files through graph input, for example `agent.invoke({"messages": [...], "files": {...}})`.

The authored system prompt begins with the harness profile applied to an empty base. With `system_prompt=None`, that is the entire authored prompt. A caller string precedes it, separated by a blank line. A caller `SystemMessage` retains its existing content blocks and receives profile text as another text block, preserving caller cache-control markers. Skills and memory middleware can subsequently inject their dynamic prompt content at execution time.

## Subagents: construction and ownership

The builder partitions `subagents=` by shape:

- A spec with `graph_id` becomes an `AsyncSubAgent` managed by `AsyncSubAgentMiddleware`. It launches a background run through the LangGraph SDK and returns a task ID; the main agent can later check, update, cancel, or list it. Remote approval and schema belong to that remote graph. A URL-less local ASGI transport requires an asynchronous parent entrypoint.
- A spec with `runnable` is a caller-compiled `CompiledSubAgent`. Its graph, state schema, and approval policy remain caller-owned; its runnable must have a `messages` state key to return a result.
- Every other spec is a declarative synchronous `SubAgent`, compiled by delegation middleware with a resolved model/profile, tools, permissions, prompt, and middleware. Omitted tools, permissions, and `interrupt_on` inherit parent values; a supplied permission list replaces the inherited rules.

Unless the active profile disables it or a synchronous subagent is already named `general-purpose`, construction inserts a default general-purpose synchronous subagent at the front of the inline list. This supplies the `task` path; when it is disabled and no other inline subagent exists, no `task` tool is exposed. Profile-specific general-purpose description and prompt can override the default; that specific prompt wins over a profile base prompt, while the profile suffix still applies.

`mode="fork"` is experimental. Unlike an isolated declarative subagent, it receives the parent conversation/state, mirrors parent prompt-producing middleware, and appends its own prompt as an addendum. A fork cannot declare its own skills and refuses recursive task delegation. Its output is still isolated as a child projection rather than becoming the parent projection.

## Middleware assembly and policy guards

The main stack has a meaningful order:

1. optional `SkillsMiddleware`, `FilesystemMiddleware`, optional `SubAgentMiddleware`, summarization, `PatchToolCallsMiddleware`, and optional `AsyncSubAgentMiddleware` form the core;
2. profile extra middleware, provider-aware prompt-caching middleware, optional `MemoryMiddleware`, and optional `HumanInTheLoopMiddleware` form the tail;
3. caller middleware replaces an existing same-name entry in place, or is spliced after the core and before the tail;
4. exclusions are applied before and after caller insertion, then `_ToolExclusionMiddleware` is appended last when required.

The caching helper always installs Anthropic prompt caching in ignore-unsupported mode; Bedrock and Fireworks caching are added only when their integration middleware can be imported, and also ignore unsupported models. This permits one assembled stack to support several model families without making optional integrations mandatory. Memory follows caching so memory-driven prompt changes do not invalidate the cached prefix.

A profile cannot remove `FilesystemMiddleware` or `SubAgentMiddleware`: they back filesystem tools, permission enforcement, and synchronous dispatch. Exclusion is guarded at construction time. A protected exclusion, an unmatched configured exclusion, or a name matching multiple concrete middleware classes raises `ValueError` rather than silently changing the capability/security boundary. Exclusions aggregate across the main and default-general-purpose stacks, because a profile may legitimately target middleware present in only one.

Filesystem permissions are enforced in `FilesystemMiddleware`, not by direct backend operations. Permission-derived interruption settings merge with `interrupt_on`; a caller setting wins for the same tool name. Any resulting configuration adds `HumanInTheLoopMiddleware`, which interrupts the LangGraph run for approval. Supply a checkpointer when an approval must survive and resume a later execution.

## Graph assembly, state, and lifecycle

The final `create_agent()` receives the resolved model, composed prompt, rewritten caller tools, middleware, response format, context schema, checkpointer, store, debug option, name, cache, and either custom `state_schema` or `DeepAgentState`. The returned graph is configured with recursion limit `9999` plus Deep Agents LangSmith integration, version, and agent-name metadata.

`DeepAgentState` extends LangChain `AgentState`; its `messages` field uses a `DeltaChannel` reducer with snapshot frequency 50, changing checkpoint growth from quadratic to linear. The reducer accepts raw message-like values, replaces/deduplicates by message ID, honors individual removal tombstones and `REMOVE_ALL_MESSAGES`, and treats an absent replay state as empty. LangGraph assigns stable IDs before checkpoint serialization, avoiding replay-time random IDs.

A custom `state_schema` is forwarded to `SubAgentMiddleware`, allowing declarative subagents to share application fields. Before compilation, private fields marked by graph or middleware schemas are derived and assigned to the delegation middleware so they are not forwarded across delegation. Annotation-resolution failures only warn, however; affected fields will not remain private. Precompiled and remote subagents retain their own schemas. Custom schemas are a typed contract rather than runtime-checked `TypedDict` inheritance, so extensions must preserve the `messages` reducer.

## Running and observing the graph

During `invoke`, `ainvoke`, or streaming, LangGraph repeatedly gives the model message history, the effective prompt, and the middleware-produced tool surface. A model response without calls completes the loop. Otherwise the selected tools run, their results and state updates are appended, and the next model turn begins. Middleware can transform requests before model calls, add or remove visible tools, summarize or offload history, write typed state, and enforce a policy around tool runs. Ordinary callables in `tools=` run only after model selection and cannot alter the preceding request.

The compiled graph supports synchronous and asynchronous v3 event streams. Consumers can use parent projections such as messages, tool calls, values, and subgraphs, plus typed subagent handles identifying the subagent, originating tool-call ID, status, and output. Drain relevant projections—concurrent parent and child draining is supported. Fork output stays separate from the parent message projection. If a delegated subagent fails, its child handle reaches `failed` with an error and the upstream runtime exception can propagate while projections drain.

## Focused verification

`test_graph.py` exercises profile selection, prompt ordering, non-mutating tool rewrites, caching wiring, middleware ordering/exclusion, default and caller-provided subagents, interruption configuration, state propagation, and graph metadata. `test_messages_reducer.py` covers replacement, removal, reset, and replay behavior. `test_deep_agent_streaming.py` executes regular, forked, and failing subagents through synchronous and asynchronous v3 projections. `backends/test_state_backend.py` verifies the graph-context failure boundary of the default backend.

## Related pages

- [Architecture overview](overview.md) — layer ownership across Deep Agents, LangChain, and LangGraph.
- [Middleware stack](middleware-stack.md) — middleware hooks and feature behavior.
- [Backends](../concepts/backends.md) — backend choices and storage/execution capabilities.
- [Profiles and models](../concepts/profiles-models.md) — provider and harness profile configuration.
- [State persistence](../concepts/state-persistence.md) — checkpoints, threads, and resumption.
- [Build a Deep Agent](../workflows/build-a-deep-agent.md) — application-level construction workflow.
