# Files

- [Agent Client Protocol](acp.md) - How deepagents-acp adapts a LangGraph graph to an editor-facing ACP server, including sessions, streaming, interrupts, and optional recovery. It also explains dcode ACP launch and the ownership boundary between its in-process ACP graphs and normal loopback server sessions.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. Documents the public action contract, credential and workspace handoff, memory cache lifecycle, and headless tool controls.
- [MCP Integration](mcp.md) - How dcode discovers, trust-gates, connects, and authenticates MCP servers, then adapts, bounds, and reports their tools across primary and delegated agents.
- [Sandbox and Partner Integrations](sandbox-partners.md) - Map dcode's optional remote sandbox providers to the shared backend contract, selection and lifecycle rules, package constraints, and Talon routing. Distinguish those providers from the local QuickJS JavaScript REPL middleware.
- [Talon Runtime Host](talon.md) - Experimental local runtime host for long-running Deep Agents, channel adapters, durable conversation state, scheduling, MCP tools, and optional sandbox execution.
