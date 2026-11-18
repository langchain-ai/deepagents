# Files

- [Agent Client Protocol Integration](acp.md) - How deepagents-acp exposes a LangGraph agent through ACP, and how dcode builds, owns, and serves its in-process ACP sessions. Covers durable sessions, approval translation, MCP loading, and focused verification.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. This guide covers the action contract, its CLI distribution boundary, state lifecycle, safeguards, verification, and release controls.
- [MCP Integration](mcp.md) - Configure, trust, load, authenticate, and operate MCP servers in dcode and Talon. Covers configuration provenance, remote and stdio safety, OAuth credential lifecycle, call deadlines, and reload behavior.
- [Sandbox and Partner Integrations](sandbox-partners.md) - How dcode acquires remote sandbox backends and how the separately packaged QuickJS middleware persists JavaScript state and bridges to agent capabilities. Explains the authorization and containment limits of filesystem, tool, and subagent bridges.
- [Talon Runtime Integration](talon.md) - Talon composes a channel-owning host with a replaceable Deep Agents runtime. This page covers per-invocation hooks and approvals, MCP and tool attachment, fresh subagents, delegation, and lifecycle boundaries.
