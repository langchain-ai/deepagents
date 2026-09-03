# Files

- [ACP Integration](acp.md) - Run a reusable Deep Agents graph or the prebuilt dcode coding agent from an ACP-capable editor over stdio. Covers session lifecycle, streamed output, permission interrupts, persistence, and dcode-specific operational boundaries.
- [MCP Integration](mcp.md) - How dcode and Talon discover, trust-filter, authenticate, load, and expose Model Context Protocol (MCP) tools. Covers configuration precedence, project-MCP trust boundaries, OAuth login, and UI-agnostic login resolution.
- [Sandbox & Partner Integrations](sandbox-partners.md) - How deepagents sandbox backends route file and shell operations through a provider's execute primitive, the isolation role sandboxes play, and what each partner package (Daytona, Modal, Runloop, Vercel, QuickJS) provides.
- [Talon: Local Runtime Host](talon.md) - Talon is an experimental local host that connects a Deep Agents runtime to messaging channels and persistent scheduled work. It coordinates conversation interruption and recovery, channel approvals, media handling, observability, and ordered shutdown for one operator-managed assistant.
