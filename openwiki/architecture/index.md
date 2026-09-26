# Files

- [Deep Agents Code Architecture](code-agent.md) - How dcode routes terminal and ACP sessions into workspace-bound Deep Agents graphs, resolves models and MCP tools, and persists session and approval state.
- [Middleware Stack and Ordering](middleware-stack.md) - How create_deep_agent constructs, filters, and executes the ordered middleware stacks for the main agent and subagents. Covers profiles, caller overrides, approvals, tool exclusion, and request-time unsupported-content filtering.
- [System Architecture Overview](overview.md) - Package ownership and runtime boundaries across the Deep Agents SDK, dcode terminal agent, ACP adapter, Talon host, evaluation suite, and sandbox partners. Explains graph assembly, request lifecycles, persistence, and independent releases.
- [Long-Running Runtime Behavior](runtime-behavior.md) - How Talon executes durable agent turns, applies retries and approval context, refreshes graphs safely, and constrains background and scheduled work.
- [SDK Construction and Execution](sdk-construction-execution.md) - Explains how create_deep_agent resolves models, profiles, backends, subagents, and middleware into a LangChain-built LangGraph agent, including state, interrupts, and request-safe multimodal execution.
- [Source Map and Ownership Boundaries](source-map.md) - Practical change map from Deep Agents public surfaces to their implementation owners, focused tests, package manifests, and independent release units. It highlights lifecycle and safety boundaries that must remain aligned across packages.
