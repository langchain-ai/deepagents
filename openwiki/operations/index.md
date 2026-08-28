# Files

- [Cost Tracking, Sessions & Runtime Stats (dcode)](cost-and-sessions.md) - How dcode estimates and persists per-thread model cost, resolves pricing from genai-prices plus bundled and user overrides, accumulates session usage stats, and persists resumable thread state in SQLite.
- [Development & Build Operations](development.md) - Practical development and CI-parity operations for independently versioned packages in the Deep Agents monorepo. Covers package-local uv and Make workflows, repository-wide checks, hooks, release fan-out, and the release-please lifecycle.
- [Security & Threat Model](security.md) - Consolidated trust and threat-model boundaries across the deepagents SDK, the deepagents-code (dcode) coding agent, and the Talon runtime, explaining where enforcement actually happens and where it does not.
