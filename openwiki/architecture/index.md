# Files

- [dcode Client and Agent Server](code-agent.md) - How dcode launches its local LangGraph agent server, streams and resumes sessions through the RemoteAgent client, and preserves workspace identity while distinguishing policy drift from runtime rebuilds.
- [Middleware Stack and Ordering](middleware-stack.md) - How create_deep_agent constructs, filters, and executes the ordered middleware stacks for the main agent and subagents. Covers profiles, caller overrides, approvals, tool exclusion, and request-time unsupported-content filtering.
- [Repository Runtime Architecture](overview.md) - Ownership boundaries among the Deep Agents SDK, dcode, ACP, Talon, and evaluation tooling. Explains Talon's host/runtime lifecycle, channel turns, persistence, scheduling, and shutdown behavior.
- [Talon Host Runtime Behavior](runtime-behavior.md) - Control flow and isolation rules for Talon's durable agent turns, host command and delivery lifecycle, models, approvals, background work, and scheduled execution.
- [SDK Construction and Execution](sdk-construction-execution.md) - Explains how create_deep_agent resolves models, profiles, backends, subagents, and middleware into a LangChain-built LangGraph agent, including state, interrupts, and request-safe multimodal execution.
- [System Ownership Map](source-map.md) - Navigation map for the Deep Agents SDK, dcode client and server, Talon host, ACP bridge, partner integrations, and evaluation harness. Use it to find public entrypoints, lifecycle owners, policy boundaries, and focused regression seams.
