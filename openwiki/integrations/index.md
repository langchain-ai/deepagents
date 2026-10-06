# Files

- [Agent Client Protocol Integration](acp.md) - How deepagents-acp adapts a LangGraph graph to an editor-facing ACP server, including sessions, streaming, interrupts, and optional recovery. It also explains dcode ACP launch and the ownership boundary between its in-process ACP graphs and normal loopback server sessions.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. Documents the public action contract, credential and workspace handoff, memory cache lifecycle, and headless tool controls.
- [MCP Servers, Trust, OAuth, and Tool Execution](mcp.md) - How dcode discovers, trust-gates, connects, and authenticates MCP servers, and how it normalizes, bounds, and reports MCP tool calls across primary and delegated agents.
- [Sandbox Provider Integrations](sandbox-partners.md) - Install and operate dcode sandbox providers, with their ownership and lifetime rules. Distinguishes provider-backed remote execution from Talon routing and the in-process QuickJS middleware.
- [Talon Runtime Integration](talon.md) - Operator-facing map of the experimental Talon CLI host, channels, durable checkpoints and history, MCP, sandbox execution, schedules, and per-assistant state.
