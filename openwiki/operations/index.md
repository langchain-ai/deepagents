# Files

- [Cost Tracking, Sessions & Runtime Stats](cost-and-sessions.md) - How dcode produces display-only model-cost estimates, prevents streamed usage revisions and replay from inflating request statistics, and persists resumable thread state through LangGraph SQLite checkpoints.
- [Development & Build Operations](development.md) - Practical development and CI-parity operations for independently versioned packages in the Deep Agents monorepo. Covers package-local uv and Make workflows, repository-wide checks, hooks, release fan-out, and the release-please lifecycle.
- [Security & Threat Model](security.md) - Consolidated trust and threat-model boundaries across the deepagents SDK, the deepagents-code (dcode) coding agent, and the Talon runtime, explaining where enforcement actually happens and where it does not.
