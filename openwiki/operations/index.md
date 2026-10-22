# Files

- [CI and Pull-Request Labeling](ci-labeling.md) - How the main CI workflow selects and gates work, including curated-notes inherited-result diagnostics, and how the privileged pull-request labeler safely applies configuration-driven metadata.
- [dcode Sessions, Resume, and Cost Operations](cost-and-sessions.md) - Operate dcode's persistent SQLite threads, interactive and headless resume safeguards, checkpoint migration, and estimated-cost accounting. Covers ownership fencing, selector behavior, pricing overrides, diagnostics, and failure handling.
- [Development, Dependencies, and Releases](development.md) - Package-scoped uv development, dependency and lockfile maintenance, local-source consumer closure, and the release-please publication process for Deep Agents packages.
- [Security Boundaries and Operational Safety](security.md) - Operational trust boundaries for privileged PR labeling and Deep Agents delegation, including untrusted repository input, remote child authority, approvals, credentials, and safe failure behavior.
