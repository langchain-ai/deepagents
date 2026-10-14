# Files

- [Agent Client Protocol](acp.md) - How deepagents-acp adapts a LangGraph graph to an editor-facing ACP server, including sessions, streaming, interrupts, and optional recovery. It also explains dcode ACP launch and the ownership boundary between its in-process ACP graphs and normal loopback server sessions.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. Documents the public action contract, credential and workspace handoff, memory cache lifecycle, and headless tool controls.
- [MCP Integration](mcp.md) - How dcode discovers, trust-gates, connects, and authenticates MCP servers, then adapts, bounds, and reports their tools across primary and delegated agents.
- [Sandbox and Partner Integrations](sandbox-partners.md) - Explains the derived-operation guarantees of remote sandbox backends and the QuickJS REPL boundary, including lifecycle, persistence, resource limits, programmatic tool calling, and approval limitations.
- [Talon Runtime Host](talon.md) - Experimental local host for a long-running Deep Agents assistant. It composes the runtime, admitted channel adapters, persistence, scheduling, tools, and per-turn context while defining lifecycle and failure boundaries.
