# Files

- [Deep Agents Code Architecture](code-agent.md) - How dcode separates terminal presentation from its local LangGraph execution server, binds durable workspace policy to threads, and manages persistence, tracing, readiness, and process cleanup.
- [SDK Middleware Assembly](middleware-stack.md) - Explains how Deep Agents builds and filters middleware stacks for the main agent and subagents, including replacement, profile exclusion, caching, memory, approval, tool-surface, and content-compatibility rules.
- [Repository Architecture Overview](overview.md) - Architecture and dependency map for the Deep Agents Python monorepo, including independently released packages, published versus local development dependencies, runtime layering, and product ownership.
- [Runtime Behavior and Failure Boundaries](runtime-behavior.md) - Execution-time state, prompt shaping, asynchronous subagent lifecycle, notification delivery, and retry boundaries in Deep Agents. Distinguishes checkpointed state and remote task records from best-effort observation and delivery.
- [SDK Construction and Execution](sdk-construction-execution.md) - How create_deep_agent constructs a Deep Agents LangChain graph, composes middleware and subagent stacks, propagates state, and configures compiled execution.
- [System Source Map](source-map.md) - Task-oriented change routing for dcode startup and serving, SQLite thread ownership, ACP, Talon, and the focused tests that protect each lifecycle contract. Use it to find the contract owner and durable boundary before changing behavior.
