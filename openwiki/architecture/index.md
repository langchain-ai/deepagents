# Files

- [Deep Agents Code Architecture](code-agent.md) - Architecture of dcode's normal terminal-client and local LangGraph-server boundary, workspace-scoped runtimes, streaming, persistence, cleanup, and separate ACP stdio mode.
- [Middleware Stack and Customization Boundaries](middleware-stack.md) - How create_deep_agent assembles and filters the ordered middleware stacks for a main agent and its subagents. Covers profile exclusions, caller insertion and replacement, state boundaries, and the distinction between middleware and ordinary tools.
- [Monorepo Architecture Overview](overview.md) - System-level map of the independently versioned Deep Agents packages, their public entry points, dependency directions, and the boundaries between the SDK, dcode, ACP, Talon, evals, and partner integrations.
- [dcode Runtime Behavior and Failure Handling](runtime-behavior.md) - How dcode launches and owns a workspace-aware LangGraph runtime, constructs its agent resources, selects bounded workspace runtimes, and surfaces startup and request failures.
- [SDK Construction and Execution](sdk-construction-execution.md) - Trace how create_deep_agent resolves its dependencies and policies into a LangChain-compiled LangGraph agent, then how state, streaming, tool calls, checkpoints, and interrupts behave at runtime.
- [Source Map and Change Routing](source-map.md) - Route an intended Deep Agents behavior change to its owning package, supported surface, implementation seam, focused tests, and release boundary. Use this as a change-navigation map rather than a package inventory.
