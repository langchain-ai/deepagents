# Files

- [Deep Agents Code Architecture](code-agent.md)
- [Middleware Assembly and Extension Boundaries](middleware-stack.md) - Explains how Deep Agents assembles, customizes, filters, and compiles middleware for main agents and subagents. Covers prompt and tool-surface composition, profile effects, permissions, and ownership boundaries.
- [Repository Architecture Overview](overview.md) - Architecture and dependency map for the Deep Agents Python monorepo, including Talon's invocation-scoped graph runtime, approval snapshots, host-owned transport, and optional progress delivery.
- [Runtime Behavior and Failure Boundaries](runtime-behavior.md) - How Deep Agents keeps graph state recoverable, compacts model context without discarding the raw log, and manages asynchronous subagent lifecycles. Covers pause, cancellation, notification, and retry boundaries across graph and host runtimes.
- [SDK Construction and Execution](sdk-construction-execution.md) - How create_deep_agent assembles and compiles Deep Agents, isolates synchronous task subagents, and coordinates remote asynchronous Agent Protocol tasks.
- [System Source Map](source-map.md) - Task-oriented source map for Deep Agents graph assembly, middleware and storage; dcode terminal and ACP boundaries; and Talon host, runtime, protocol, and channel adapters. Use it to locate the contract owner and focused tests before changing behavior.
