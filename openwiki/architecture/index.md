# Files

- [Deep Agents Code (dcode) Architecture](code-agent.md) - How the prebuilt terminal coding agent splits into a terminal client and an agent server, how a request flows between them over a streaming protocol, and how its layered configuration resolves.
- [Middleware Stack](middleware-stack.md) - How create_deep_agent composes base scaffolding, caller, and profile/tail middleware into the final request-shaping stack, and how middleware differs from plain tools.
- [Architecture Overview](overview.md) - How Deep Agents is layered on LangChain create_agent and the LangGraph runtime, and how the monorepo packages map to responsibilities so you know which layer owns a behavior before changing it.
- [SDK: Construction & Execution (create_deep_agent)](sdk-construction-execution.md) - How create_deep_agent assembles a fully configured deep agent in a single construction pass and how the compiled LangGraph agent executes each turn as a model-call plus tool-call loop until a final response.
- [Source Map](source-map.md) - Directory-to-responsibility index across the Deep Agents monorepo, mapping create_deep_agent assembly, SDK middleware/backends/profiles, and the deepagents-code coding agent to the files that own each behavior.
