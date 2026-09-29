# Files

- [Agent Client Protocol Integration](acp.md) - Explains how deepagents-acp projects a LangGraph agent into an ACP stdio server, including session-scoped graph construction, streaming, permissions, cancellation, and optional durable recovery. It also describes dcode's ACP launcher and its separate tool, policy, and checkpoint ownership.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. Documents the public action contract, credential and workspace handoff, memory cache lifecycle, and headless tool controls.
- [MCP Servers, Trust, and OAuth](mcp.md) - How dcode discovers and trust-gates MCP servers, loads transports and tools, persists OAuth credentials, coordinates refresh, and connects CLI and TUI login interactions.
- [Sandbox Providers and Execution Boundaries](sandbox-partners.md) - Explains how dcode selects and owns optional remote sandbox providers, why a server sandbox belongs to only one workspace, and how Talon reuses the same provider lifecycle while retaining selected control-plane paths on the host.
- [Talon Channels, Models, MCP, and Sandboxes](talon.md) - Integration-facing operating guide for Talon's channel adapters, per-chat models, MCP and OAuth lifecycle, media, tracing, optional sandboxes, and experimental security boundary.
