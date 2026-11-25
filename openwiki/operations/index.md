# Files

- [CI Gates and Labels](ci-labeling.md) - Operational map of merge-facing CI, acknowledgement labels, release publication, release notes, and workflow trust boundaries. Explains what blocks a merge or release, where labels apply, and how credentials are scoped.
- [dcode Costs, Traces, and Session Operations](cost-and-sessions.md) - How dcode persists best-effort model-cost estimates and transports JavaScript subagent receipts into the owning thread checkpoint. Covers receipt ownership, cancellation and resume safety, and focused regression tests.
- [dcode Installation and Update Lifecycle](dcode-install-and-update.md) - How dcode discovers, configures, and safely applies application updates and optional dependency changes. Covers configuration resolution, cached release discovery, installation safeguards, startup behavior, and diagnostic introspection.
- [Development and Release Operations](development.md) - Package-local development, generated lockfile propagation, release-please baselines, curated notes, and recovery-safe PyPI and GitHub release operations.
- [Security and Trust Boundaries](security.md) - Operational security guidance for Talon's experimental local-host authority, delegation, MCP and session bridges, channel admission, and CI credential scope. It distinguishes convenience controls from authorization and containment boundaries.
