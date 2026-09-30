# Files

- [Agent Client Protocol Integration](acp.md) - Explains how deepagents-acp projects a LangGraph agent into an ACP stdio server, including session-scoped graph construction, streaming, permissions, cancellation, and optional durable recovery. It also describes dcode's ACP launcher and its separate tool, policy, and checkpoint ownership.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. Documents the public action contract, credential and workspace handoff, memory cache lifecycle, and headless tool controls.
- [MCP Servers, Trust, OAuth, and Tool Execution](mcp.md) - How dcode discovers, trust-gates, connects, and authenticates MCP servers, and how it normalizes, bounds, and reports MCP tool calls across primary and delegated agents.
- [Sandbox Providers, QuickJS, and Execution Boundaries](sandbox-partners.md) - Explains dcode's optional remote sandbox-provider lifecycle and the separate QuickJS JavaScript execution middleware, including subagent dispatch, replay identity, streaming events, and cost ownership.
- [Talon Channels, Models, MCP, and Sandboxes](talon.md) - Operator guide to wiring the experimental Talon host to channels, models, MCP servers, scheduling state, and optional execution sandboxes.
