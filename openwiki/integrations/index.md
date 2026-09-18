# Files

- [Agent Client Protocol Integration](acp.md) - Connect a reusable Deep Agents graph or dcode's prebuilt coding agent to an ACP-capable editor over stdio. This guide covers session construction, streaming and replay projection, selectors, permissions, durable recovery, and dcode startup.
- [GitHub Action Integration](github-action.md) - Run a bounded, non-interactive dcode task from a GitHub Actions job. Covers the composite action inputs, credential and workspace boundaries, memory cache lifecycle, tool integrations, and headless approval behavior.
- [Model Context Protocol Integration](mcp.md) - How dcode and Talon discover, validate, authorize, expose, refresh, and manage Model Context Protocol servers. The two runtimes deliberately keep configuration trust, credentials, and tool lifetimes separate.
- [Sandbox and Partner Integrations](sandbox-partners.md) - How dcode discovers, provisions, and owns sandbox providers; how provider adapters meet the deepagents shell and filesystem contract; and the release, dependency, security, and test boundaries of partner packages.
- [Talon Runtime Host](talon.md) - Experimental single-assistant runtime host that connects Deep Agents to messaging channels, persistent conversation state, cron jobs, MCP, and background subagents. Covers Talon lifecycle, routing, failure behavior, and operational limits.
