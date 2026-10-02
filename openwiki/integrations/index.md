# Files

- [Agent Client Protocol Integration](acp.md) - Explains how deepagents-acp projects a LangGraph agent into an ACP stdio server, including session-scoped graph construction, streaming, permissions, cancellation, and optional durable recovery. It also describes dcode's ACP launcher and its separate tool, policy, and checkpoint ownership.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. Documents the public action contract, credential and workspace handoff, memory cache lifecycle, and headless tool controls.
- [MCP Servers, Trust, OAuth, and Tool Execution](mcp.md) - How dcode discovers, trust-gates, connects, and authenticates MCP servers, and how it normalizes, bounds, and reports MCP tool calls across primary and delegated agents.
- [Sandbox Provider Integrations](sandbox-partners.md) - Explains dcode and Talon remote sandbox-provider discovery, provisioning, ownership, and routing, and distinguishes those execution capabilities from host-resident integrations and QuickJS middleware.
- [Talon Runtime Integration](talon.md) - Operator guide to configuring and operating the experimental Talon local runtime, including channels, persistent history and cron, MCP, subagents, and sandboxed execution.
