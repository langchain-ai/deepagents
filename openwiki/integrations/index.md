# Files

- [Agent Client Protocol Integration](acp.md) - Run a reusable Deep Agents graph or dcode's coding agent from an ACP-capable editor over stdio. Covers ACP session lifecycle, streamed update projection, permissions, durable recovery, and dcode-specific startup.
- [GitHub Action Integration](github-action.md) - Run one bounded, headless dcode task from a GitHub Actions job. Covers action inputs, credentials, cached memory, skills, tool controls, outputs, and focused regression tests.
- [MCP Integration](mcp.md) - How dcode and Talon discover, validate, authorize, expose, refresh, and manage Model Context Protocol servers. Explains their distinct configuration, trust, credential, and runtime-lifetime boundaries.
- [Sandbox and Partner Integrations](sandbox-partners.md) - How dcode discovers, provisions, and owns sandbox providers; how provider adapters meet the deepagents shell and filesystem contract; and the operational boundaries of supported partner packages.
- [Talon Runtime Host](talon.md) - Talon is an experimental single-assistant host that connects a Deep Agents runtime to messaging channels, durable conversation history, scheduled work, and background subagents. This page covers lifecycle, conversation identity, approvals, operational limits, persistence, and security boundaries.
