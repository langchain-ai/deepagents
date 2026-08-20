# Partner package guidance

Follow the repository-wide rules in the root `AGENTS.md`. Each partner package is independently versioned and owns its environment, `pyproject.toml`, `Makefile`, and tests.

## Adding a partner package

Wire a new partner into all relevant repository surfaces:

- Area options in `.github/ISSUE_TEMPLATE/bug-report.yml`, `feature-request.yml`, and `privileged.yml`
- Dependabot in `.github/dependabot.yml`
- Scope-to-label and path rules in `.github/scripts/labeling/pr-labeler-config.json`
- Issue labels in `.github/workflows/auto-label-by-package.yml`
- Change detection and jobs in `.github/workflows/ci.yml`
- Allowed scopes in `.github/workflows/pr_lint.yml`
- Package setup and inputs in `.github/workflows/release.yml`
- Release detection in `.github/workflows/release-please.yml`
- Package entries in `release-please-config.json` and `.release-please-manifest.json`
- The managed-packages table in `.github/RELEASING.md`
- Sandbox options and credential checks in `.github/workflows/harbor.yml` when the partner is sandbox-backed

For a first release at `0.0.1`, set the manifest baseline to `0.0.0`.
