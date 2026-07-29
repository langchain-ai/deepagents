# `.github` layout

Quick map of CI/automation files in this folder.

## Top level

| Path | Purpose |
| --- | --- |
| `workflows/` | GitHub Actions workflows (entrypoints and reusable callers) |
| `actions/` | Local composite actions consumed by workflows |
| `scripts/` | Helper scripts invoked by workflows, plus their tests |
| `ISSUE_TEMPLATE/` | Issue forms |
| `PULL_REQUEST_TEMPLATE.md` | Default PR body template |
| `CODEOWNERS` | Review routing for paths in this tree |
| `dependabot.yml` | Dependabot update groups |
| `RELEASING.md` | Release-please / publish process |
| `SECRETS.md` | Non-`GITHUB_TOKEN` CI credential inventory (names and scopes only) |
| `images/` | Static assets referenced by workflows or docs |

Package-level CI conventions and partner onboarding checklists live in root [`AGENTS.md`](../AGENTS.md).

## Workflows (`workflows/`)

- **Entry workflows** (no leading underscore) run on events such as `pull_request`, `push`, `schedule`, or `workflow_dispatch`.
- **Reusable workflows** are named `_*.yml` (for example `_lint.yml`, `_test.yml`, `_eval.yml`) and are called from entry workflows via `workflow_call`.
- Prefer extending an existing reusable workflow over pasting setup/checkout/`uv` boilerplate into a new entry file.

Credential placement rules are in [`SECRETS.md`](./SECRETS.md). Release wiring is in [`RELEASING.md`](./RELEASING.md).

## Local composite actions (`actions/`)

Reusable steps shared by multiple workflows. Today this is mainly `actions/uv_setup` (Python + pinned `uv` with caching). Add a new composite action here only when two or more workflows need the same multi-step setup.

## Helper scripts (`scripts/`)

Production helpers are nested by domain:

```text
scripts/
├── checks/      # repo integrity / sync checkers
├── evals/       # eval/harbor matrix and aggregation
├── labeling/    # PR/issue labeling and triage automation
├── release/     # release-please guards, notes, pin checks
└── tests/       # tests for the helpers above (and some workflow contracts)
```

### Placement rules

1. **Put new helpers in an existing domain folder** when they clearly belong there.
2. **Add a domain folder** only for a sustained new area (not a one-off script). Keep the name short and topic-style like the neighbors.
3. Prefer plain modules invoked with `python .github/scripts/<domain>/<script>.py` (or `node …`) from workflow steps. Avoid inventing a package install story under `.github/`.
4. Keep secrets out of scripts; read them from the environment the workflow injects.

### Tests (`scripts/tests/`)

Tests mirror top-level layout:

```text
scripts/<domain>/<name>.py
scripts/tests/<domain>/test_<name>.py
```

Special cases:

| Location | What goes there |
| --- | --- |
| `scripts/tests/workflows/` | Contract tests over workflow/action YAML (job graphs, options matrices, secret scoping, root `action.yml`) — not a production `scripts/workflows/` tree |
| `scripts/tests/conftest.py` | Shared pytest path setup so domain helpers import without packaging |

`conftest.py` puts each domain directory (`checks`, `evals`, `labeling`, `release`) on `sys.path`. New domains must be added there if their tests import modules by bare filename the same way.

## Related docs

- [`RELEASING.md`](./RELEASING.md) — version branches, release-please, fan-out guards, publishing
- [`SECRETS.md`](./SECRETS.md) — secret/variable names and environment scopes
- [`../AGENTS.md`](../AGENTS.md) — monorepo conventions, PR title scopes, "adding a new partner to CI"
