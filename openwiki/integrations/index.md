# Files

- [ACP (Agent Client Protocol) Integration](acp.md) - Run a generic Deep Agent or the dcode coding agent in an ACP-capable editor over stdio. This guide covers the server bridge, session and model lifecycle, MCP boundary, and the distinct protocol and dcode approval behaviors.
- [MCP Integration](mcp.md) - How dcode and Talon discover, trust-filter, authenticate, load, and expose Model Context Protocol (MCP) tools. Covers configuration precedence, project-MCP trust boundaries, OAuth login, and UI-agnostic login resolution.
- [Sandbox & Partner Integrations](sandbox-partners.md) - How deepagents sandbox backends route file and shell operations through a provider's execute primitive, the isolation role sandboxes play, and what each partner package (Daytona, Modal, Runloop, Vercel, QuickJS) provides.
- [Talon: Local Runtime Host](talon.md) - Talon is the experimental, single-operator local host that owns channel adapters, a persistent cron scheduler, and the Deep Agents runtime in one asyncio event loop, with graceful shutdown and per-conversation serialization.
