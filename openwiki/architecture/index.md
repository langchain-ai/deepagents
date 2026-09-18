# Files

- [Deep Agents Code Architecture](code-agent.md) - Architecture of dcode's normal terminal-client and local LangGraph-server boundary, workspace-scoped runtimes, persistence, cleanup, and separate ACP stdio mode.
- [Deep Agents Middleware Stack](middleware-stack.md) - How create_deep_agent assembles, orders, filters, and hands off middleware for main agents and synchronous subagents. Covers profile and caller customization, prompt caching, memory and approval tails, exclusions, context repair, and state boundaries.
- [Monorepo Architecture Overview](overview.md) - System-level map of the independently versioned Deep Agents packages, their public entry points, dependency directions, and the boundaries between the SDK, dcode, ACP, Talon, evals, and partner integrations.
- [dcode Runtime Behavior and Failure Handling](runtime-behavior.md) - How dcode launches a workspace-aware LangGraph server, binds remote threads to durable workspace policy, caches runtime resources, and handles retries, recovery, and cleanup.
- [SDK Construction and Execution](sdk-construction-execution.md) - Trace how create_deep_agent resolves models, profiles, storage, and policies into a LangChain-compiled LangGraph agent, then how its tool loop, state, streaming, and interrupts operate.
- [Source Map and Change Routing](source-map.md) - Route an intended Deep Agents behavior change to its owning package, supported surface, implementation seam, focused tests, and release boundary. Use this as a change-navigation map rather than a package inventory.
