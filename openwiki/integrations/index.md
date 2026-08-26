# Files

- [ACP (Agent Client Protocol) Integration](acp.md) - How Deep Agents run inside ACP-capable editors like Zed, covering the deepagents-acp server that bridges a Deep Agent to the Agent Client Protocol and the prebuilt dcode coding agent exposed with `dcode --acp`.
- [MCP Integration](mcp.md) - How dcode and talon discover, configure, authenticate, and load Model Context Protocol (MCP) servers into the agent's tool surface, including config precedence, trust gating, OAuth flows, and approval interaction.
- [Sandbox & Partner Integrations](sandbox-partners.md) - How deepagents sandbox backends route file and shell operations through a provider's execute primitive, the isolation role sandboxes play, and what each partner package (Daytona, Modal, Runloop, Vercel, QuickJS) provides.
- [Talon: Local Runtime Host](talon.md) - Talon is the experimental, single-operator local host that owns channel adapters, a persistent cron scheduler, and the Deep Agents runtime in one asyncio event loop, with graceful shutdown and per-conversation serialization.
