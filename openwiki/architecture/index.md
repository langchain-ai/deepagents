# Files

- [Deep Agents Code Architecture](code-agent.md) - How dcode routes terminal and ACP sessions into workspace-bound Deep Agents graphs, resolves models and MCP tools, and persists session and approval state.
- [Middleware Stack and Ordering](middleware-stack.md) - How create_deep_agent constructs, filters, and executes the ordered middleware stacks for the main agent and subagents. Covers profiles, caller overrides, approvals, tool exclusion, and request-time unsupported-content filtering.
- [System Architecture Overview](overview.md) - Ownership and dependency boundaries in the Deep Agents monorepo, with emphasis on Talon as an experimental local single-event-loop host. Covers Talon bootstrap, host/runtime split, channel adapters, optional scheduling, persistence, and safe extension points.
- [Long-Running Runtime Behavior](runtime-behavior.md) - Control flow and isolation rules for Talon's durable agent turns, per-conversation models, approvals, host delivery, background work, and scheduled execution.
- [SDK Construction and Execution](sdk-construction-execution.md) - Explains how create_deep_agent resolves models, profiles, backends, subagents, and middleware into a LangChain-built LangGraph agent, including state, interrupts, and request-safe multimodal execution.
- [Source Map and Ownership Boundaries](source-map.md) - Practical navigation and ownership map for the experimental Talon local runtime host. Use it to locate public entrypoints, lifecycle seams, state boundaries, channel integrations, and their narrowest regression tests.
