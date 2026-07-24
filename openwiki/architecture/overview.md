---
type: Architecture Overview
title: Deep Agents runtime and package architecture
description: How the core SDK composes LangChain and LangGraph, and how the terminal agent, deployment CLI, ACP adapter, and integrations attach to that runtime.
tags: [architecture, sdk, langchain, langgraph, integrations]
---
# Runtime and package architecture

## Layering: harness over an existing runtime

The SDK is deliberately not a competing runtime:

```text
Deep Agents  -> opinionated harness: middleware, backends, profiles, subagents
LangChain    -> agent abstraction: model + tools + middleware
LangGraph    -> execution runtime: state, checkpoints, streaming, interrupts
```

`create_deep_agent()` in `libs/deepagents/deepagents/graph.py` is the assembly point. It resolves a model/profile and backend, normalizes main/subagent middleware and prompts, builds the default general-purpose subagent where enabled, and delegates to LangChain’s `create_agent()`. The returned compiled graph is then invoked or streamed through LangGraph. This relationship matters: durable state, resumability, and interrupt mechanics belong to LangGraph; Deep Agents changes what the agent sees and does through composition.

The [Deep Agents Code workflow](../workflows/deep-agents-code.md) configures this assembly point into a coding-agent server graph. The [operations guide](../engineering/operations-and-testing.md) maps the packages and checks that protect the public interfaces around it.

## Core execution path

1. **Application creates a graph.** Public consumers import `create_deep_agent` from `libs/deepagents/deepagents/__init__.py` and provide a model, tools, system prompt, backends, subagents, persistence, and/or middleware.
2. **SDK composes runtime behavior.** `graph.py` supplies the standard tool surface—planning/todos, filesystem operations, optional shell execution, and task delegation—and configures model/provider and harness profiles.
3. **LangChain runs the loop.** The model receives the assembled prompt, messages, and current tool surface; it can respond or call tools, with results appended to state.
4. **LangGraph manages execution.** It carries state/checkpoints, streams progress, and pauses/resumes through interrupts. Deep Agents’ `DeepAgentState` uses a delta-style message reducer to avoid superlinear checkpoint growth in long threads.

### Middleware versus tools

Caller-provided `tools=` are callable only when the model selects them. Middleware can instead alter a model request, tool visibility, prompt, or state before/around model and tool calls. The default stack uses that leverage for filesystem/memory/skills/subagent instructions, request cleanup, compaction/offload, permissions, and provider-specific behavior.

When a behavior differs in delegated work, inspect the subagent type and its own middleware stack before modifying the main stack. The core source anchors are `libs/deepagents/deepagents/graph.py`, `middleware/`, `backends/`, and `profiles/`; `libs/ARCHITECTURE.md` explains the same ownership model at length.

## Tool, backend, and state boundaries

- **Backends** select storage and execution capabilities. Public exports include state, filesystem, store, composite, local-shell, LangSmith-sandbox, and ContextHub variants (`libs/deepagents/deepagents/backends/`). If a backend cannot execute shell commands, the SDK removes `execute` and shell-specific prompt text rather than merely denying a call later.
- **Permissions** constrain built-in filesystem tools at call time. They do not hide tools and do not automatically authorize arbitrary caller tools; the containment boundary belongs to tools/backends, consistent with the root README’s trust-the-LLM posture.
- **State/checkpoints** are LangGraph-owned, while filesystem and cross-thread memory persistence are routed by Deep Agents backends. Keep middleware-private state private where possible; recent SDK history includes a fix to isolate private custom state from subagents.
- **Profiles** tune provider and harness behavior. Plugin registrations are additive through the `deepagents.provider_profiles` and `deepagents.harness_profiles` entry-point groups.

## Adjacent package boundaries

| Package | Relationship to the core SDK |
| --- | --- |
| `libs/code` | Builds a coding-specific middleware/tool/approval stack around `create_deep_agent` and exposes it through a terminal client plus LangGraph server. See [Deep Agents Code](../workflows/deep-agents-code.md). |
| `libs/acp` | Adapts a compiled graph (or session-aware graph factory) to ACP events: messages, tool progress/diffs, todos, and supported HITL flows. Free-form LangGraph interrupts are not generally representable there. |
| `libs/cli` | Deploys managed Deep Agents projects, agents, and MCP server registrations through `langgraph-sdk`/HTTP; it is not the `dcode` interactive REPL. |
| `libs/partners` | Supplies provider/sandbox integration packages. Shell/filesystem behavior remains backend-dependent, so partner changes can affect the SDK tool surface. |
| root GitHub Action | `action.yml` runs `dcode` non-interactively. Its raw output is explicitly unfiltered, so downstream workflows should not blindly echo it into other services (`ACTION.md`). |

## Change guidance and caveats

- Preserve public SDK signatures and exports; repository guidance requires keyword-only defaulted parameters for new public options where feasible. Confirm imports, examples, and tests before altering an exported API.
- Trace a user-facing argument from `create_deep_agent()` through middleware/backend installation before changing it. A missing tool typically indicates profile/middleware assembly; a visible-but-failing tool suggests backend capability or permission enforcement.
- Applications should pass a model explicitly. The inspected `graph.py` still permits an implicit default but marks it deprecated for removal in 1.0.0.
- Use package-local unit tests under `libs/deepagents/tests/unit_tests/` for harness changes and integration tests for networked behavior. The exact commands and cross-package checks are in [Operations and testing](../engineering/operations-and-testing.md).
