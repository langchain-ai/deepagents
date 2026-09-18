# Files

- [dcode Sessions, Cost, and Local State](cost-and-sessions.md) - Explains dcode thread identity and resume behavior, the boundary between local checkpoint state and server operations, and the best-effort cost and cache diagnostics for long-running coding sessions.
- [Development, CI, and Releases](development.md) - Package-scoped uv and Make workflows, CI routing, lock integrity, and independently versioned release operations for the Deep Agents monorepo. Covers release-please guardrails and recovery when a publish does not complete.
- [Security Boundaries and Operational Safeguards](security.md) - Trust boundaries and operating safeguards for Deep Agents tools, dcode workspaces and MCP, GitHub Actions and CI secrets, and Talon's experimental runtime. Distinguishes approval and mediation controls from actual process or tenant isolation.
