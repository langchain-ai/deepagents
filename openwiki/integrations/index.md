# Files

- [Agent Client Protocol Integration](acp.md) - Run a reusable Deep Agents graph or dcode's prebuilt coding agent from an ACP-capable editor over stdio. Covers session creation and recovery, working-directory validation, streamed turns, approvals, and the MCP ownership boundary.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. Documents the root composite action’s public input and output contract, command translation, state cache lifecycle, and headless security controls.
- [MCP Integration](mcp.md) - How dcode and Talon discover, validate, authorize, expose, refresh, and manage Model Context Protocol servers. Explains their distinct configuration, trust, credential, and runtime-lifetime boundaries.
- [Sandbox and Partner Integrations](sandbox-partners.md) - How dcode discovers, provisions, and owns sandbox providers; how provider adapters meet the deepagents shell and filesystem contract; and the operational boundaries of supported partner packages.
- [Talon Runtime Host](talon.md) - Experimental local host for one Deep Agents assistant, covering channel turns, runtime lifecycle, MCP, durable state, cron, media, and observability. It explains the operational and security limits that matter when running Talon locally.
