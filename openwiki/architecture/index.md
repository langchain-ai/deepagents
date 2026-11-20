# Files

- [Deep Agents Code Architecture](code-agent.md) - dcode separates client interaction from the server-hosted Deep Agents graph. This page describes execution surfaces, Hooks v2 ownership and interrupt resumption, terminal program-status reporting, and server-authoritative model selection.
- [SDK Middleware Assembly](middleware-stack.md) - Explains how Deep Agents assembles the main and subagent middleware stacks, applies profile policy, and projects synchronous task results. Covers fork state and history, checkpoint and tracing metadata, and provider prompt-caching adapters.
- [Repository Architecture Overview](overview.md) - Package roles, published Python dependency contracts, development-only uv source wiring, lockfile propagation, and release baselines for the Deep Agents monorepo.
- [Runtime Behavior and Failure Boundaries](runtime-behavior.md) - Execution ownership, synchronous subagent delegation boundaries, persistence, retry, interruption, and delivery semantics across Deep Agents, dcode, and Talon.
- [SDK Construction and Execution](sdk-construction-execution.md) - How Deep Agents assembles LangChain agents, defines its default state schema, and delegates execution to LangChain and LangGraph. Covers local task delegation, remote Agent Protocol tasks, middleware ordering, and state-transfer boundaries.
- [Source Map](source-map.md) - Navigation-oriented ownership map for Deep Agents graph construction, dcode configuration and JavaScript execution, Talon runtime replacement, QuickJS internals, and CI release and Harbor evaluation paths.
