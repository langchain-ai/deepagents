# Files

- [Deep Agents Code Architecture](code-agent.md) - dcode separates terminal presentation from a local agent server. This page documents that ownership boundary and the retained, lazily expanded tool-detail path that keeps long transcripts responsive without changing server execution.
- [SDK Middleware Assembly](middleware-stack.md) - Explains how Deep Agents builds and filters middleware stacks for the main agent and subagents, including replacement, profile exclusion, caching, memory, approval, tool-surface, and content-compatibility rules.
- [Repository Architecture Overview](overview.md) - Package graph, ownership boundaries, Python and release contracts, and the distinction between published dependencies and local development sources in the Deep Agents monorepo.
- [Runtime Behavior and Failure Boundaries](runtime-behavior.md) - Execution-time state, prompt shaping, asynchronous subagent lifecycle, notification delivery, and retry boundaries in Deep Agents. Distinguishes checkpointed state and remote task records from best-effort observation and delivery.
- [SDK Construction and Execution](sdk-construction-execution.md) - How create_deep_agent assembles a Deep Agents LangChain graph and how optional RubricMiddleware adds bounded, rubric-driven review after an ordinary agent completion.
- [System Source Map](source-map.md) - Practical change-routing map for dcode graph assembly and middleware, Talon MCP and runtime seams, QuickJS, and CI/release-note automation. Use it to find the lifecycle owner and closest behavioral test before changing a boundary.
