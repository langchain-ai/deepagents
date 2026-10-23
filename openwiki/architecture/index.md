# Files

- [Deep Agents Code Architecture](code-agent.md) - How dcode dispatches terminal, headless, and ACP execution; separates its client from the agent server; and binds durable threads to server-authorized workspaces.
- [SDK Middleware Assembly](middleware-stack.md) - Documents the exact middleware composition, replacement, exclusion, and tail-safeguard rules used when the Deep Agents SDK compiles main, declarative, forked, and general-purpose agents.
- [Repository Architecture Overview](overview.md) - Architecture and dependency map for the Deep Agents Python monorepo, including independently released packages, published versus local development dependencies, runtime layering, and product ownership.
- [Runtime Behavior and Failure Boundaries](runtime-behavior.md) - Execution-time state, compaction, asynchronous subagent lifecycle, notification delivery, and retry boundaries in Deep Agents. Distinguishes checkpointed state and remote task records from best-effort observation and delivery.
- [SDK Construction and Execution](sdk-construction-execution.md) - How create_deep_agent assembles a Deep Agents LangChain graph, defines its state and tool surface, and separates synchronous delegation from remote Agent Protocol tasks.
- [System Source Map](source-map.md) - Task-oriented starting points for Deep Agents graph construction, dcode CLI and server execution, ACP session bridging, Talon hosting, and evaluation operations. Use it to identify the contract owner, durable state, lifecycle boundary, and narrowest relevant tests before changing behavior.
