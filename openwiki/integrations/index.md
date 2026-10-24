# Files

- [Agent Client Protocol Integration](acp.md) - How deepagents-acp adapts a LangGraph graph to an editor-facing ACP server, including sessions, streaming, interrupts, and recovery. It distinguishes dcode's local ACP server from its normal RemoteAgent transport.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. Documents the public action contract, runner execution boundary, package-release relationship, memory cache lifecycle, and CI coverage.
- [MCP Integration](mcp.md) - How dcode discovers, validates, loads, and cleans up Model Context Protocol servers, with explicit-config startup errors, isolated server failures, tool adaptation, and OAuth lifecycle. It also contrasts Talon's separate operator-selected MCP model.
- [Sandbox and Partner Integrations](sandbox-partners.md) - How dcode discovers optional sandbox providers, acquires or attaches sandboxes, runs setup, and assigns cleanup ownership. Covers the independently distributed partner adapters and Daytona's bounded command execution behavior.
- [Talon Runtime Integration](talon.md) - Talon is an experimental local host that routes admitted channel and scheduled work into Deep Agents. It coordinates request-scoped capabilities, graph replacement, approvals, persistence, progress delivery, and lifecycle cleanup.
