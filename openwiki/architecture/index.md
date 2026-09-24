# Files

- [Deep Agents Code Architecture](code-agent.md) - How dcode selects interactive, headless, and ACP entrypoints, separates its terminal client from the agent server, assembles agent graphs, and owns configuration and approval state.
- [Middleware Stack Assembly](middleware-stack.md) - How create_deep_agent assembles and filters the ordered middleware stacks for a main agent and its subagents. Covers profile exclusions, caller insertion and replacement, state boundaries, and the distinction between middleware and ordinary tools.
- [Repository Architecture Overview](overview.md) - Package ownership and dependency boundaries across the Deep Agents SDK, dcode, ACP, Talon, evaluation, and partner integrations. Explains the reusable graph assembly seam, dcode's product and ACP assembly, and independently released package baselines.
- [Code Runtime and Session Behavior](runtime-behavior.md) - How dcode launches a workspace-aware LangGraph runtime, streams and recovers durable sessions, and contains failures in retry, offload, and shutdown paths.
- [SDK Construction and Agent Execution](sdk-construction-execution.md) - Explains how create_deep_agent resolves models, profiles, backends, subagents, and middleware into a LangChain-built LangGraph agent, and how its state, checkpoints, interrupts, and streams behave.
- [Repository Source Map](source-map.md) - Practical map of Deep Agents package ownership, executable entrypoints, focused tests, and independent release units. It highlights the Code MCP loader and debug console plus the SDK backend and context-eviction seams.
