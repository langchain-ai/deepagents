# Files

- [Agent Client Protocol Integration](acp.md) - How deepagents-acp adapts a LangGraph graph to an editor-facing ACP server, including sessions, streaming, interrupts, and optional recovery. It also explains dcode ACP launch and how its in-process ACP graphs differ from remote Agent Protocol delegation.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. Documents the public action contract, runner execution boundary, package-release relationship, memory cache lifecycle, and CI coverage.
- [MCP Integration and OAuth](mcp.md) - How dcode and Talon discover, configure, load, reload, and expose MCP tools, including trust boundaries, OAuth credential lifecycle, and failure handling.
- [Sandbox and Partner Integrations](sandbox-partners.md) - How dcode discovers sandbox providers, acquires or attaches sandboxes, runs setup, and assigns cleanup ownership. Covers the optional provider packages and the contract for third-party extensions.
- [Talon Runtime Integration](talon.md) - Talon is an experimental local host that routes admitted channel and scheduled work into Deep Agents. It coordinates request-scoped capabilities, graph replacement, approvals, persistence, progress delivery, and lifecycle cleanup.
