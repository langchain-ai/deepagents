# Files

- [Deep Agents Code Architecture](code-agent.md) - Architecture of dcode's normal terminal-client and local LangGraph-server boundary, workspace-scoped runtimes, streaming, persistence, cleanup, and separate ACP stdio mode.
- [Middleware Stack and Extension Boundaries](middleware-stack.md) - How create_deep_agent assembles and filters ordered middleware stacks for the main agent and subagents. Covers core and tail extension seams, profile exclusions, state boundaries, and context compaction.
- [Monorepo Architecture Overview](overview.md) - System-level map of the independently versioned Deep Agents packages, their public entry points, dependency directions, and the boundaries between the SDK, dcode, ACP, Talon, evals, and partner integrations.
- [dcode Runtime Behavior and Failure Handling](runtime-behavior.md) - How dcode launches and owns a workspace-aware LangGraph runtime, constructs its agent resources, selects bounded workspace runtimes, and surfaces startup and request failures.
- [SDK Construction and Execution](sdk-construction-execution.md) - Trace how create_deep_agent resolves configuration into a LangChain-compiled LangGraph agent and how Deep Agents middleware, backends, state, delegation, persistence, interrupts, and streams participate at execution time.
- [Source Map and Change Routing](source-map.md) - Route a Deep Agents behavior change to its owning package, public surface, implementation seam, focused tests, and release boundary. Use this as a change-navigation map rather than a directory inventory.
