# Files

- [Agent Client Protocol Integration](acp.md) - How deepagents-acp projects LangGraph agents into editor-facing ACP sessions, including durable recovery, permission handling, and hidden local context. It also explains why dcode ACP is an in-process stdio server rather than dcode's normal RemoteAgent path.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. Documents the public action contract, runner execution boundary, package-release relationship, memory cache lifecycle, and CI coverage.
- [MCP Integration](mcp.md) - Configure and operate dcode MCP servers, including optional provider-native deferred MCP-tool search with safe fallback, and Talon's separate managed MCP and authorization path.
- [Sandbox and Partner Integrations](sandbox-partners.md) - How dcode selects and owns optional remote sandbox providers, and how the separate QuickJS package provides in-process JavaScript middleware with durable, authenticated REPL snapshots.
- [Talon Runtime Integration](talon.md) - Talon is an experimental local host for Deep Agents channels and schedules. It assembles request-scoped graphs, MCP capabilities, approvals, persistence, and delivery while making the limits of its single-operator security model explicit.
