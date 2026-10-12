# Files

- [Deep Agents Code Architecture](code-agent.md) - dcode separates a terminal client from a LangGraph agent server, with durable goal context, skill pinning and trust, classifier-backed approval, and defensive stream rendering at their boundary.
- [Middleware Assembly Stack](middleware-stack.md) - Exact middleware assembly, replacement, and exclusion order for Deep Agents main agents and subagent forms. Covers profile invariants, request-time tool visibility, and declarative, compiled, and remote subagent boundaries.
- [Repository Architecture Overview](overview.md) - Architecture and release map for the Deep Agents Python monorepo, covering the SDK, dcode, ACP, evals, Talon, and optional partner integrations. Explains dependency direction, persistence ownership, and long-running-host lifecycle boundaries.
- [Runtime Behavior, State, and Delegation](runtime-behavior.md) - Runtime contracts for Deep Agents graph construction, checkpointed state, binary blob hydration, synchronous and remote delegation, and recovery from interrupted tool calls. Explains ownership boundaries that must survive graph and middleware changes.
- [SDK Construction and Execution](sdk-construction-execution.md) - Explains how create_deep_agent resolves model and profile policy, assembles tools, subagents, approvals, and middleware, then compiles the LangChain and LangGraph execution loop.
- [Source Map and Ownership Boundaries](source-map.md) - A change-oriented map from Deep Agents behavior to its owning package, public entrypoint, runtime domain, and focused regression suite.
