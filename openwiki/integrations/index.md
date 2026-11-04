# Files

- [Agent Client Protocol Integration](acp.md) - How deepagents-acp projects LangGraph agents into editor-facing ACP sessions, including durable recovery, permission handling, and hidden local context. It also explains why dcode ACP is an in-process stdio server rather than dcode's normal RemoteAgent path.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. Documents the root action compatibility contract, dcode execution boundary, memory cache lifecycle, release alignment, and CI coverage.
- [MCP Integration](mcp.md) - Configure, trust, authenticate, and operate MCP servers in dcode. This page covers layered configuration, OAuth token lifecycle, command and TUI controls, failure isolation, and the distinct session behavior of local and headless/server execution.
- [Sandbox and Partner Integrations](sandbox-partners.md) - How dcode selects and owns optional remote sandbox providers, and how the separate QuickJS partner supplies request-aware, durable in-process JavaScript REPL middleware.
- [Talon Runtime Integration](talon.md) - How Talon turns admitted channel messages into scoped Deep Agents requests, including lifecycle, progress delivery, MCP reloads, sandbox routing, and observability. Describes the experimental security limits at the host-to-runtime boundary.
