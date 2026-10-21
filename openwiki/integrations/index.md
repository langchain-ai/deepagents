# Files

- [Agent Client Protocol Integration](acp.md) - How deepagents-acp adapts a LangGraph graph to an editor-facing ACP server, including sessions, streaming, interrupts, and optional recovery. It also explains dcode ACP launch and how its in-process ACP graphs differ from remote Agent Protocol delegation.
- [GitHub Action Integration](github-action.md) - Run one bounded, non-interactive dcode task from a GitHub Actions job. Documents the public action contract, credential and workspace handoff, memory cache lifecycle, and headless tool controls.
- [MCP Configuration, Tools, and OAuth](mcp.md) - How dcode discovers and trust-gates MCP configuration, connects and exposes MCP tools, and manages OAuth login, credential persistence, reloads, and failures.
- [Sandbox and Partner Integrations](sandbox-partners.md) - How dcode discovers and owns remote sandbox providers, how Talon consumes that surface for filesystem and shell execution, and how the QuickJS partner differs from a remote sandbox.
- [Talon Runtime Integration](talon.md) - Talon is an experimental local host that turns admitted channel input into Deep Agents turns, managing media, approvals, progress, history, schedules, and background-result follow-up. This page explains ownership boundaries, request flow, and operating constraints.
