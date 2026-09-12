# Files

- [Agent Client Protocol Integration](acp.md) - Run a reusable Deep Agents graph or dcode's prebuilt coding agent from an ACP-capable editor over stdio. Covers session creation and recovery, working-directory validation, streamed turns, approvals, and the MCP ownership boundary.
- [GitHub Action Integration](github-action.md) - Run a bounded, non-interactive dcode task from a GitHub Actions job. Covers the composite action inputs, credential and workspace boundaries, memory cache lifecycle, tool integrations, and headless approval behavior.
- [MCP Integration](mcp.md) - How dcode and Talon validate, authorize, load, expose, and refresh Model Context Protocol servers. Covers their intentionally separate configuration, trust, credential, and session boundaries.
- [Sandbox and Partner Integrations](sandbox-partners.md) - How dcode discovers, provisions, and owns sandbox providers; how provider adapters meet the deepagents shell and filesystem contract; and the operational boundaries of supported partner packages.
- [Talon Runtime Host](talon.md) - Talon is an experimental single-assistant host that connects Deep Agents to messaging channels, persistent conversations, MCP tools, background subagents, and cron delivery. This page explains its lifecycle, interruption and recovery semantics, approval boundaries, and operational limits.
