# Files

- [Deep Agents Code Architecture](code-agent.md) - How dcode separates terminal presentation from local agent execution, establishes server-authoritative workspace policy and durable sessions, and exposes interactive, headless, ACP, plugin, MCP, and Textual rendering seams.
- [SDK Middleware Assembly](middleware-stack.md) - Explains how Deep Agents builds and filters middleware stacks for the main agent and subagents, including replacement, profile exclusion, caching, memory, approval, tool-surface, and content-compatibility rules.
- [Repository Architecture Overview](overview.md)
- [Runtime Behavior and Failure Boundaries](runtime-behavior.md) - Execution-time state, prompt shaping, asynchronous subagent lifecycle, notification delivery, and retry boundaries in Deep Agents. Distinguishes checkpointed state and remote task records from best-effort observation and delivery.
- [SDK Construction and Execution](sdk-construction-execution.md) - How create_deep_agent constructs a Deep Agents LangChain graph, composes middleware and subagent stacks, propagates state, and configures compiled execution.
- [System Source Map](source-map.md) - Practical change-routing map for Deep Agents graph assembly, dcode client and server, plugins, MCP, Talon, Harbor evaluations, sandbox partners, and delivery automation. Use it to locate the owner of a behavioral contract and its lifecycle boundary before making a change.
