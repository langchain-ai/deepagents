# Files

- [dcode Product Architecture](code-agent.md) - How dcode assembles the Deep Agents SDK into normal CLI and TUI sessions, a server-owned runtime, durable workspace bindings, tools and approvals, and a separate ACP surface.
- [Middleware Stack and Extension Boundaries](middleware-stack.md) - The ordered middleware assembly used by `create_deep_agent()` for main and delegated agents, including profile and caller customization, exclusions, caching, memory, and state boundaries.
- [Monorepo Architecture Overview](overview.md) - System-level map of the independently versioned Deep Agents packages, their public entry points, dependency directions, and the boundaries between the SDK, dcode, ACP, Talon, evals, and partner integrations.
- [dcode Server Runtime Behavior](runtime-behavior.md) - How dcode launches and owns a workspace-aware LangGraph child server, binds threads to trusted workspace policy, caches runtimes, and handles streaming, offload, retries, recovery, and shutdown.
- [SDK Construction and Execution](sdk-construction-execution.md) - Trace how create_deep_agent resolves its dependencies and policies into a LangChain-compiled LangGraph agent, then how state, streaming, tool calls, checkpoints, and interrupts behave at runtime.
- [Source Map and Change Routing](source-map.md) - Route a Deep Agents behavior change from its public contract to the owning assembly or lifecycle boundary, adjacent consumers, and the smallest useful regression test. Use this responsibility map to avoid implementing a fix in the wrong layer.
