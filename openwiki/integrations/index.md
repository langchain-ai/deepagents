# Files

- [Agent Client Protocol](acp.md) - How deepagents-acp adapts a LangGraph graph to an editor-facing ACP server, including sessions, streaming, interrupts, and optional recovery. It also explains dcode ACP launch and the ownership boundary between its in-process ACP graphs and normal loopback server sessions.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. Documents the public action contract, credential and workspace handoff, memory cache lifecycle, and headless tool controls.
- [MCP Integration](mcp.md) - How dcode discovers, trust-gates, connects, and authenticates MCP servers, then adapts, bounds, and reports their tools across primary and delegated agents.
- [Sandbox and Partner Integrations](sandbox-partners.md) - How dcode discovers and owns remote sandbox providers, how Talon consumes that surface for filesystem and shell execution, and how the QuickJS partner differs from a remote sandbox.
- [Talon Runtime Integration](talon.md) - Talon is an experimental, local host for long-running Deep Agents, channel adapters, schedules, persistence, and optional dcode sandboxes. This page explains process ownership, state boundaries, lifecycle choices, and operating constraints.
