# Files

- [Deep Agents Code Architecture](code-agent.md) - Architecture of dcode's normal terminal-client and local LangGraph-server boundary, workspace-scoped runtimes, streaming, persistence, cleanup, and separate ACP stdio mode.
- [Middleware Stack and Ordering](middleware-stack.md) - How create_deep_agent assembles, orders, customizes, and filters middleware for the main agent and subagents. Covers profile exclusions, state boundaries, context-overflow recovery, and order-sensitive safety invariants.
- [Monorepo Architecture Overview](overview.md) - System-level map of the independently versioned Deep Agents packages, their public entry points, dependency directions, and the boundaries between the SDK, dcode, ACP, Talon, evals, and partner integrations.
- [dcode Runtime Behavior and Failure Handling](runtime-behavior.md) - How dcode starts a workspace-aware LangGraph server, executes and streams agent turns, handles interrupts and remote conflicts, and resumes persisted threads safely.
- [SDK Construction and Execution](sdk-construction-execution.md) - Trace how create_deep_agent resolves dependencies and policies into a LangChain agent compiled on LangGraph, including middleware, subagents, state, streaming, checkpoints, and interrupts.
- [Source Map and Change Routing](source-map.md) - Route an intended Deep Agents behavior change to its owning package, supported surface, implementation seam, focused tests, and release boundary. Use this as a change-navigation map rather than a package inventory.
