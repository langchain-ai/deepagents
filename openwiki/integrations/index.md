# Files

- [Agent Client Protocol Integration](acp.md) - Explains how deepagents-acp projects a LangGraph agent into an ACP stdio server, including session-scoped graph construction, streaming, permissions, cancellation, and optional durable recovery. It also describes dcode's ACP launcher and its separate tool, policy, and checkpoint ownership.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. Documents the public action contract, credential and workspace handoff, memory cache lifecycle, and headless tool controls.
- [MCP Servers, Trust, and OAuth](mcp.md) - How dcode discovers and trust-gates MCP servers, loads transports and tools, persists OAuth credentials, coordinates refresh, and connects CLI and TUI login interactions.
- [Sandbox and Partner Backends](sandbox-partners.md) - Maps the Deep Agents sandbox contract to dcode provider lifecycle and the Daytona, Modal, Runloop, and Vercel partner adapters. Distinguishes remote shell environments from the separate QuickJS in-process JavaScript execution capability.
- [Talon Runtime Integration](talon.md) - Talon is an experimental local host for long-running Deep Agents. It connects channel adapters, graph execution, approvals, MCP, persistent conversation history, scheduled work, and background subagents.
