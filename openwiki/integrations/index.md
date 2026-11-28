# Files

- [Agent Client Protocol Integration](acp.md)
- [GitHub Action Integration](github-action.md) - Run one bounded, headless dcode task from a GitHub Actions job. Covers the composite action contract, its published deepagents-code runtime boundary, safety controls, state lifecycle, and verification.
- [MCP Integration](mcp.md) - Configure, trust, load, authenticate, and operate MCP servers in dcode and Talon. Covers configuration provenance, remote and stdio safety, OAuth credential lifecycle, call deadlines, and reload behavior.
- [Sandbox and Partner Integrations](sandbox-partners.md) - Optional partner packages provide remote sandbox backends or an in-process QuickJS interpreter for Deep Agents. This page distinguishes provider selection from a complete security boundary and documents QuickJS lifecycle, authority bridges, persistence, and cost transport.
- [Talon Runtime Integration](talon.md) - Talon is an experimental local host that composes channel adapters, schedules, a Deep Agents runtime, media preparation, scoped tools, and model-specific reasoning controls. This page explains request ownership, lifecycle, local-host limitations, graph reload, delegation, and host-mediated file delivery.
