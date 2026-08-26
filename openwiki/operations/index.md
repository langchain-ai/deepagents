# Files

- [Cost Tracking, Sessions & Runtime Stats (dcode)](cost-and-sessions.md) - How dcode estimates and persists per-thread model cost, resolves pricing from genai-prices plus bundled and user overrides, accumulates session usage stats, and persists resumable thread state in SQLite.
- [Development & Build Operations](development.md) - How to develop, build, lint, and release packages in the Deep Agents monorepo — the uv-only per-package workflow, repo-wide Makefile fan-out targets, pre-commit hooks, coding-agent conventions, and release-please independent versioning.
- [Security & Threat Model](security.md) - Consolidated trust and threat-model boundaries across the deepagents SDK, the deepagents-code (dcode) coding agent, and the Talon runtime, explaining where enforcement actually happens and where it does not.
