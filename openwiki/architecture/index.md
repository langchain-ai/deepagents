# Files

- [Deep Agents Code Architecture](code-agent.md) - dcode separates terminal presentation from a local agent server and starts that server with guarded local-process lifecycle management. This page explains startup readiness, server-owned model metadata, client rendering state, and resilient lazy tool-detail expansion.
- [SDK Middleware Assembly](middleware-stack.md) - Explains how Deep Agents assembles the main and subagent middleware stacks, applies profile policy, and projects synchronous task results. Covers fork state and history, checkpoint and tracing metadata, and provider prompt-caching adapters.
- [Repository Architecture Overview](overview.md) - Package roles, published Python and dependency contracts, development-only uv source wiring, lockfile closure, and release baselines for the Deep Agents monorepo.
- [Runtime Behavior and Failure Boundaries](runtime-behavior.md) - Execution ownership, synchronous subagent delegation boundaries, persistence, retry, interruption, and delivery semantics across Deep Agents, dcode, and Talon.
- [SDK Construction and Execution](sdk-construction-execution.md) - How Deep Agents assembles a LangChain graph and separates synchronous forked subagents from remote Agent Protocol background tasks. Covers middleware ordering, state-transfer contracts, asynchronous task lifecycle, and regression invariants.
- [Repository Source Map](source-map.md) - System-oriented change routing from public SDK, dcode, Talon, ACP, partner, and release surfaces to state owners, enforcement boundaries, and focused regression suites.
