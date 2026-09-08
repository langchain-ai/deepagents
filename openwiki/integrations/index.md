# Files

- [Agent Client Protocol Integration](acp.md) - Run a reusable Deep Agents graph or dcode's prebuilt coding agent from an ACP-capable editor over stdio. Covers sessions, streamed turns, approvals, persistence, MCP boundaries, and dcode startup and cleanup.
- [Model Context Protocol Integration](mcp.md) - How dcode and Talon configure, authorize, discover, expose, refresh, and clean up Model Context Protocol servers. Separates configuration trust and credential handling from tool discovery and runtime ownership.
- [Sandbox & Partner Integrations](sandbox-partners.md) - The deepagents sandbox backend contract and dcode lifecycle for remote execution, including provider discovery, cleanup, and the Daytona, Modal, Runloop, Vercel, and QuickJS integration boundaries.
- [Talon Local Runtime Host](talon.md) - Talon is an experimental single-operator local host for a Deep Agents assistant, messaging channels, durable conversation history, and persistent cron work. It documents boot, interruption, approval, media, MCP, observability, and shutdown behavior.
