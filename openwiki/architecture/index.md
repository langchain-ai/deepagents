# Files

- [Deep Agents Code (dcode) Architecture](code-agent.md) - Architecture and lifecycle guide for dcode's normal loopback LangGraph-server runtime and its distinct ACP stdio mode. Covers configuration handoff, workspace-bound graph construction, streaming, persistence, cleanup, and failure boundaries.
- [Middleware Stack](middleware-stack.md) - How create_deep_agent assembles, filters, and hands off ordered middleware stacks for the main agent and its subagents. Covers extension placement, protected exclusions, tool visibility, and isolated versus forked delegation boundaries.
- [Architecture Overview](overview.md) - How Deep Agents is layered on LangChain create_agent and the LangGraph runtime, and how the monorepo packages map to responsibilities so you know which layer owns a behavior before changing it.
- [SDK Construction & Execution](sdk-construction-execution.md) - How create_deep_agent resolves models and profiles, assembles prompts, tools, subagents, middleware, and graph configuration, then hands execution to the LangChain and LangGraph agent runtime.
- [Source Map](source-map.md) - Practical ownership and public-entrypoint map for the Deep Agents SDK, dcode, ACP, evaluations, Talon, and partner packages. Use it to find the implementation boundary and focused tests for a change.
