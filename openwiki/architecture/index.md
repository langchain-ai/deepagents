# Files

- [Deep Agents Code Architecture](code-agent.md) - Product architecture for dcode, its terminal client and agent-server runtime, and the CLI-owned version and resilient self-update subsystem.
- [Middleware Assembly and Extension Boundaries](middleware-stack.md) - Explains how Deep Agents assembles, customizes, filters, and compiles middleware for main agents and subagents. Covers prompt and tool-surface composition, profile effects, permissions, and ownership boundaries.
- [Repository Architecture Overview](overview.md) - Architecture and dependency map for the Deep Agents Python monorepo, including published and development-only package relationships, SDK graph assembly, and product hosting boundaries.
- [Runtime Behavior and Failure Boundaries](runtime-behavior.md) - How Deep Agents persists and recovers graph state, reduces model context, and operates durable asynchronous subagent tasks. Covers queued follow-ups, notification opt-out and delivery, polling, cancellation, and model retry boundaries.
- [SDK Construction and Execution](sdk-construction-execution.md) - How create_deep_agent builds the Deep Agents middleware stack, partitions synchronous and remote subagents, and exposes task execution and tracking surfaces.
- [System Source Map](source-map.md) - Task-oriented source map for Deep Agents graph assembly, middleware and storage; dcode terminal and ACP boundaries; and Talon host, runtime, protocol, and channel adapters. Use it to locate the contract owner and focused tests before changing behavior.
