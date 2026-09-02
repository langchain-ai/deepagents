# Files

- [Cost Tracking, Sessions & Runtime Stats](cost-and-sessions.md) - How dcode produces display-only model-cost estimates, prevents streamed usage revisions and replay from inflating request statistics, and persists resumable thread state through LangGraph SQLite checkpoints.
- [Development & Build Operations](development.md) - Package-local development, validation, lockfile maintenance, and release operations for the independently versioned Python packages in this monorepo. Use this guide to choose the correct Makefile entrypoint and avoid accidental release fan-out.
- [Security & Threat Model](security.md) - Consolidated trust and threat-model boundaries across the deepagents SDK, the deepagents-code (dcode) coding agent, and the Talon runtime, explaining where enforcement actually happens and where it does not.
