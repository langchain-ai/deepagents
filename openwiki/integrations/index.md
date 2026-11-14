# Files

- [Agent Client Protocol Integration](acp.md) - How deepagents-acp projects LangGraph agents into editor-facing ACP sessions, including streaming, recovery, and permission decisions. It also describes dcode's in-process ACP server, resolved approval modes, MCP startup boundaries, and session ownership fencing.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. Documents the root action contract, CLI distribution and runtime assumptions, memory lifecycle, security handoff, and direct workflow tests.
- [MCP Integration](mcp.md) - Configure, trust, load, authenticate, and operate MCP servers in dcode and Talon. Covers configuration provenance, remote and stdio safety, OAuth credential lifecycle, unattended authorization, and reload behavior.
- [Sandbox and Partner Integrations](sandbox-partners.md) - How dcode selects and owns optional remote sandbox providers, and how the separately packaged QuickJS middleware embeds a JavaScript REPL. Covers QuickJS persistence, direct filesystem and tool bridges, approval boundaries, and partner package contracts.
- [Talon Runtime Integration](talon.md) - Talon composes channel-facing host ownership with a replaceable Deep Agents graph. It documents graph assembly, approval and capability snapshots, delegation modes, hooks and MCP seams, and lifecycle boundaries.
