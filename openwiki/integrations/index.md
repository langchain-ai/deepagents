# Files

- [Agent Client Protocol Integration](acp.md) - How deepagents-acp exposes a LangGraph graph through ACP and how dcode builds, owns, and serves durable ACP sessions. Covers graph selection, stream projection, approvals, persistence fencing, and operational compatibility.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. This guide covers the composite action contract, published CLI boundary, state lifecycle, safeguards, verification, and release safety.
- [MCP Integration](mcp.md) - Configure, trust, load, authenticate, and operate MCP servers in dcode and Talon. Covers configuration provenance, remote and stdio safety, OAuth credential lifecycle, call deadlines, and reload behavior.
- [Sandbox and Partner Integrations](sandbox-partners.md) - QuickJS embeds a persistent, in-process JavaScript REPL in Deep Agents, with checkpointed snapshots and explicit host-capability bridges. This page explains lifecycle, authority boundaries, resource limits, subagent and cost transport, and the tests that protect them.
- [Talon Runtime Integration](talon.md) - Talon composes a channel-owning host with a replaceable Deep Agents runtime. It documents graph assembly, per-invocation authority, process-local delegation, diagnostics, lifecycle, and compatibility with dcode and the SDK.
