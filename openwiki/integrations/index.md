# Files

- [Agent Client Protocol Integration](acp.md) - How deepagents-acp exposes a LangGraph agent through ACP and how dcode builds, owns, and serves durable ACP sessions. Covers factory boundaries, stdio startup, streaming projection, approvals, persistence, and operational compatibility.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. This guide covers the composite action contract, published CLI boundary, state lifecycle, safeguards, verification, and release safety.
- [MCP Integration](mcp.md) - Configure, trust, load, authenticate, and operate MCP servers in dcode and Talon. Covers configuration provenance, remote and stdio safety, OAuth credential lifecycle, call deadlines, and reload behavior.
- [Sandbox and Partner Integrations](sandbox-partners.md) - How dcode acquires remote sandbox backends and how the separately packaged QuickJS middleware persists JavaScript state and bridges to agent capabilities. Explains the authorization and containment limits of filesystem, tool, and subagent bridges.
- [Talon Runtime Integration](talon.md) - Talon composes a channel-owning host with a replaceable Deep Agents runtime. This page covers per-invocation hooks and approvals, MCP and tool attachment, fresh subagents, delegation, and lifecycle boundaries.
