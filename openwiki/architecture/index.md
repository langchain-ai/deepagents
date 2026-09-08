# Files

- [Deep Agents Code Architecture](code-agent.md) - Architecture and lifecycle guide for dcode's normal loopback LangGraph-server runtime and its distinct ACP stdio mode. Covers configuration handoff, workspace-bound graph construction, streaming, persistence, cleanup, and failure boundaries.
- [Middleware Stack & Composition](middleware-stack.md) - How create_deep_agent composes profile-dependent middleware for the main agent and distinct subagent paths. Covers ordering, caller replacement and exclusion rules, model-visible tool filtering, state boundaries, and context modes.
- [Architecture Overview](overview.md) - How Deep Agents layers an opinionated harness over LangChain create_agent and the LangGraph runtime, and how the monorepo ownership boundaries identify the right component for a change.
- [Runtime Behavior & Failure Findings](runtime-behavior.md) - Verified runtime behavior and diagnostic seams for dcode agent execution, remote streaming, retry and recovery, interrupts, and server startup. Separates source-backed operational contracts from focused test findings.
- [SDK Construction & Execution](sdk-construction-execution.md) - How create_deep_agent resolves models, profiles, backends, prompts, middleware, subagents, permissions, and configuration into a LangChain-compiled LangGraph agent.
- [Source Map & Change Boundaries](source-map.md) - Practical ownership map from public surfaces and runtime domains to implementation modules and focused tests across the Deep Agents SDK, dcode, ACP, Talon, evaluations, and partner packages.
