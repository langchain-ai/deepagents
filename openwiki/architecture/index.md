# Files

- [Deep Agents Code Architecture](code-agent.md) - Architecture of the dcode terminal client and LangGraph server, including workspace authority, remote operations, UI compatibility boundaries, configuration, titles, and accounting.
- [Middleware Stack Assembly](middleware-stack.md) - Exact middleware assembly, replacement, and exclusion order for Deep Agents main agents and subagent forms. Covers prompt-producing middleware, tool visibility, filesystem permissions, and human approval.
- [Repository Architecture Overview](overview.md) - Architecture and release map for the Deep Agents Python monorepo, covering the SDK, dcode, ACP, evals, Talon, and optional partner integrations. Explains dependency direction, persistence ownership, and long-running-host lifecycle boundaries.
- [Runtime Behavior and State Boundaries](runtime-behavior.md) - How a Deep Agents graph is assembled, evolves its checkpointed state, executes tools and subagents, and pauses or resumes around human approval. Covers message reduction, middleware-owned state, and the boundaries between parent, inline, forked, compiled, and remote graphs.
- [SDK Construction and Execution](sdk-construction-execution.md) - Explains how create_deep_agent resolves model and profile policy, assembles tools, subagents, approvals, and middleware, then compiles the LangChain and LangGraph execution loop.
- [Source Map and Ownership Boundaries](source-map.md) - A change-oriented map from Deep Agents behavior to its owning package, public entrypoint, runtime domain, and focused regression suite.
