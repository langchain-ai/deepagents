# Files

- [Deep Agents Code Architecture](code-agent.md) - Architecture of dcode's terminal and headless clients, local LangGraph server boundary, workspace-scoped runtimes, streaming, persistence, configuration bootstrap, and separate ACP stdio mode.
- [Middleware Stack Assembly](middleware-stack.md) - How create_deep_agent assembles and filters the ordered middleware stacks for a main agent and its subagents. Covers profile exclusions, caller insertion and replacement, state boundaries, and the distinction between middleware and ordinary tools.
- [Repository Architecture Overview](overview.md)
- [Code Runtime and Session Behavior](runtime-behavior.md) - How dcode starts a workspace-aware LangGraph server, routes remote sessions, streams and recovers thread state, and protects server-owned offload persistence.
- [SDK Construction and Agent Execution](sdk-construction-execution.md) - Explains how create_deep_agent resolves models, profiles, backends, subagents, and middleware into a LangChain-built LangGraph agent, and how its state, checkpoints, interrupts, and streams behave.
- [Source Map and Ownership Boundaries](source-map.md) - Route a Deep Agents behavior change from its public surface or runtime entrypoint to the owning package, lifecycle seam, focused tests, examples, and release unit. Use this as a practical change-navigation reference rather than a directory inventory.
