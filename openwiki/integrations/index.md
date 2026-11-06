# Files

- [Agent Client Protocol Integration](acp.md) - How deepagents-acp projects LangGraph agents into editor-facing ACP sessions, including durable recovery, permission handling, and hidden local context. It also explains the versioned dcode ACP surface and its in-process stdio boundary.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. Documents the root action compatibility contract, dcode execution boundary, memory cache lifecycle, versioned CLI assumptions, and CI coverage.
- [MCP Integration](mcp.md) - Configure, trust, authenticate, and operate MCP servers in dcode. This page covers layered configuration, OAuth token lifecycle, command and TUI controls, failure isolation, and the distinct session behavior of local and headless/server execution.
- [Sandbox and Partner Integrations](sandbox-partners.md) - How dcode selects and owns optional remote sandbox providers, and how the separate QuickJS partner supplies request-aware, durable in-process JavaScript REPL middleware.
- [Talon Runtime Integration](talon.md) - Talon's MCP adapter boundary preserves MCP schemas for OpenAI Responses model binding, normalizes tool arguments, scopes OAuth authorization to a tool call, and turns MCP protocol failures into correlated tool results.
