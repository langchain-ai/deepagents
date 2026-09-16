# Partner package guidance

Follow the repository-wide rules in the root [`AGENTS.md`](../../AGENTS.md), including [Warnings are errors](../../AGENTS.md#warnings-are-errors) — the heading each partner `pyproject.toml` cites by name. Each partner package is independently versioned and owns its environment, `pyproject.toml`, `Makefile`, and tests.

## Adding a partner package

Wire a new partner into all relevant repository surfaces:

- Area options in `.github/ISSUE_TEMPLATE/bug-report.yml`, `feature-request.yml`, and `privileged.yml`
- Dependabot in `.github/dependabot.yml`
- Scope-to-label and path rules in `.github/scripts/labeling/pr-labeler-config.json`
- Issue labels in `.github/workflows/auto-label-by-package.yml`
- Change detection and jobs in `.github/workflows/ci.yml`
- Allowed scopes in `.github/workflows/pr_lint.yml`, mirrored byte-identically into the `SCOPES_RE` lists in `.githooks/pre-push` and `.github/workflows/branch_name_check.yml`. The `branch-scopes-sync` pre-commit hook fails until all three agree.
- Package setup and inputs in `.github/workflows/release.yml`
- Release detection in `.github/workflows/release-please.yml`
- Package entries in `release-please-config.json` and `.release-please-manifest.json`
- The managed-packages table in `.github/RELEASING.md`
- Sandbox options and credential checks in `.github/workflows/harbor.yml` when the partner is sandbox-backed
- Matrix entries and per-partner secret gating in `.github/workflows/integration_tests.yml`
- The distribution-to-path map in `.github/scripts/release/build_release_notes.py`
- The package list in `.github/workflows/raise_langchain_minimums.yml`
- The credential inventory in `.github/SECRETS.md`

For a first release, set the manifest baseline to `0.0.0`. See [Adding a release-please-managed package](../../.github/RELEASING.md#adding-a-release-please-managed-package) for why, and for the check that blocks a wrong baseline.
