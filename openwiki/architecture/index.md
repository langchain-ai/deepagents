# Files

- [Deep Agents Code Architecture](code-agent.md) - dcode separates client interaction from the server-hosted Deep Agents graph. This page describes execution surfaces, Hooks v2 ownership and interrupt resumption, terminal program-status reporting, and server-authoritative model selection.
- [SDK Middleware Assembly](middleware-stack.md) - Explains how Deep Agents assembles the main and subagent middleware stacks, applies profile policy, and projects synchronous task results. Covers fork state and history, checkpoint and tracing metadata, and provider prompt-caching adapters.
- [Repository Architecture Overview](overview.md) - Package roles, published Python compatibility contracts, development-only uv source substitutions, lockfile propagation, and release baselines in the Deep Agents monorepo.
- [Runtime Behavior and Failure Boundaries](runtime-behavior.md) - Request-scoped Talon graph lifecycle, approval and hook context, detached and scheduled subagents, delivery settlement, and scoped file delivery.
- [SDK Construction and Execution](sdk-construction-execution.md) - How Deep Agents assembles LangChain agents, defines its default state schema, and delegates execution to LangChain and LangGraph. Covers local task delegation, remote Agent Protocol tasks, middleware ordering, and state-transfer boundaries.
- [Source Map](source-map.md) - Behavior-oriented routing for Talon invocation scope, host-owned channel delivery, model selection, MCP reload, and agent-to-channel file delivery.
