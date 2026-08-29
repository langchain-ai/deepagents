# Files

- [Deep Agents Code (dcode) Architecture](code-agent.md) - Ownership and lifecycle guide for dcode's loopback client/server runtime and its separate ACP stdio mode. Covers graph construction, streaming, persistence, startup cleanup, configuration, and failure boundaries.
- [Middleware Stack](middleware-stack.md) - How create_deep_agent composes base scaffolding, caller, and profile/tail middleware into the final request-shaping stack, and how middleware differs from plain tools.
- [Architecture Overview](overview.md) - How Deep Agents is layered on LangChain create_agent and the LangGraph runtime, and how the monorepo packages map to responsibilities so you know which layer owns a behavior before changing it.
- [SDK: Construction & Execution (create_deep_agent)](sdk-construction-execution.md) - How create_deep_agent assembles a fully configured deep agent in a single construction pass and how the compiled LangGraph agent executes each turn as a model-call plus tool-call loop until a final response.
- [Source Map](source-map.md) - Practical ownership and entrypoint map for the Deep Agents SDK, dcode, ACP, evaluations, Talon, and partner integrations. Use it to select the right implementation boundary and focused tests before changing behavior.
