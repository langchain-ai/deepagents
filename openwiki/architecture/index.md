# Files

- [dcode Client and Agent Server](code-agent.md) - dcode separates Textual terminal presentation and interaction from loopback agent-server execution and checkpoint authority. This page explains client-owned compatibility, command completion, thread selection, resume, and durable name presentation alongside the server boundary.
- [Middleware Stack and Ordering](middleware-stack.md) - Exact middleware assembly, replacement, and exclusion order for Deep Agents main agents and each synchronous subagent stack. Covers the boundary between tool visibility, filesystem permissions, and human approval.
- [Architecture Overview](overview.md) - How the Deep Agents monorepo separates the reusable SDK from dcode, ACP, Talon, evaluations, and optional provider integrations. Covers dependency direction, state ownership, package versions, and the lifecycle boundaries of the long-running host.
- [Talon Runtime and Host Behavior](runtime-behavior.md) - Control flow and isolation rules for Talon's agent turns, host delivery lifecycle, approvals and OAuth routing, background work, scheduled execution, history, retries, and shutdown.
- [SDK Construction and Execution](sdk-construction-execution.md) - Explains how create_deep_agent resolves model and profile policy, assembles tools, subagents, approvals, and middleware, then compiles the LangChain and LangGraph execution loop.
- [System Source Map](source-map.md) - Change-oriented ownership and focused regression neighborhoods for SDK graph assembly and dcode command dispatch, thread persistence, completion, and Textual presentation.
