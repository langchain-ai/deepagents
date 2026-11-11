# Files

- [CI Gates and Labels](ci-labeling.md) - CI topology, required-gate semantics, unconditional lockfile verification, text hygiene, trusted curated-release handling, and narrowly scoped PR-label exceptions. Also covers labeling and release fan-out controls that affect merge operations.
- [dcode Costs, Traces, and Session Operations](cost-and-sessions.md) - Operate dcode's durable thread checkpoints and estimated cost accounting while keeping transient UI telemetry distinct. Covers server-side offload settlement, trace redaction and lifecycle, ownership fencing, and focused verification.
- [dcode Installation and Update Lifecycle](dcode-install-and-update.md) - How dcode checks, announces, and applies self-updates, including install-method safeguards, metadata caching, locking, Python migrations, and optional dependency changes. Covers safe deferral, recovery, and editable-install behavior for updates, extras, and arbitrary packages.
- [Development and Release Operations](development.md) - Package-local development contracts, Python and dependency compatibility, lockfile regeneration, release baselines, and validation for cross-package changes.
- [Security and Trust Boundaries](security.md) - Operational guidance for execution authority, filesystem policy, dcode server startup, MCP/OAuth safety, and Talon channel and unattended-run controls.
