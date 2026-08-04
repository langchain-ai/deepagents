# Development

Starting point for working in the Deep Agents monorepo. For how the code is structured at runtime, see [`ARCHITECTURE.md`](./ARCHITECTURE.md).

> [!IMPORTANT]
> Before opening a pull request, read the [LangChain contributing guide](https://docs.langchain.com/oss/python/contributing/overview). External PRs must link to an issue or discussion that a maintainer has approved, and the contributor must be assigned to it before the PR is opened.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) — manages interpreters, virtual environments, and dependencies. Do not use `pip`, `poetry`, or `conda`.
- `make` — task runner. Every package's `Makefile` is the source of truth for its commands; run `make help` in any package directory to list targets.

`uv` provisions the right Python interpreter automatically, so there is no global Python version to install or pin.

## Quickstart

Pick the package you are changing, install its dependencies, and use its `Makefile` for the normal edit-test-lint loop:

```bash
uv tool install pre-commit
pre-commit install --install-hooks
cd libs/deepagents
uv sync --all-groups
make test
make lint
```

Use `make help` inside any package to see its supported targets. To run a repo-wide check, move to `libs/` and use the fan-out targets, for example `make lint` or `make lock-check`.

## Repository layout

This is a monorepo of independently versioned packages under `libs/`:

```txt
libs/
├── deepagents/     # Core SDK — create_deep_agent, middleware, backends
├── acp/            # Agent Client Protocol integration
├── cli/            # Deployment CLI (init / dev / deploy)
├── evals/          # Evaluation suite and Harbor integration
├── code/           # Prebuilt coding agent for interactive and headless use
├── talon/          # Local runtime host for long-running agents
└── partners/       # Provider/sandbox integrations
    ├── daytona/
    ├── modal/
    ├── vercel/
    ├── runloop/
    └── quickjs/
```

Each package has its own `pyproject.toml`, `Makefile`, and `README.md`. There is no root `pyproject.toml`; you work inside the package you are changing. Local package dependencies are editable, so changes in one package are visible to sibling packages that depend on it during development.

## Setup

Work inside the package you are changing. `uv` creates and manages the virtual environment for you — no manual `activate` needed.

```bash
cd libs/deepagents
uv sync --all-groups      # install the package + all dependency groups
```

Prefer the package's `make` targets for standard workflows; use `uv run ...` for direct one-off commands.

## Common commands

Run these from inside a package directory (e.g. `libs/deepagents`). They are consistent across the core SDK packages (`deepagents`, `code`); run `make help` to see what a given package supports:

| Command | What it does |
| --- | --- |
| `make help` | List the package's available targets |
| `make test` | Run unit tests (no network; coverage output in packages that enable it) |
| `make test TEST_FILE=tests/unit_tests/test_foo.py` | Run a single test file |
| `make integration_test` | Run integration tests (network allowed) |
| `make lint` | Run `ruff` checks + `ty` type checking |
| `make format` | Auto-format and apply safe `ruff` fixes |
| `make type` | Run the `ty` type checker only |
| `make coverage` | Run the package's explicit coverage target, usually including XML output |

You can also run a specific test directly:

```bash
uv run --group test pytest tests/unit_tests/test_specific.py
```

### Repo-wide commands

Run these from `libs/` to fan out across packages:

| Command | What it does |
| --- | --- |
| `make lint` | Lint every package |
| `make format` | Format every package |
| `make lock` | Update all lockfiles |
| `make lock-check` | Verify all lockfiles are up to date |
| `make lock-bump DEP=<pkg>` | Bump one dependency across all lockfiles |

## Pre-commit hooks

The repo uses [`pre-commit`](https://pre-commit.com/) for formatting, linting, lockfile checks, and Conventional Commit message validation:

```bash
uv tool install pre-commit   # or: pipx install pre-commit
pre-commit install --install-hooks
```

The hooks run `make format lint` for changed packages and validate commit messages, so most CI lint failures are caught before you push.

### Branch-name pre-push hook

The `pre-push` stage also runs a branch-name check (`.githooks/pre-push`, registered in `.pre-commit-config.yaml`) that rejects pushes of branches that don't follow the `<github-username>/<scope>/<short-description>` convention (e.g. `mdrxy/cli/startup-cmd-flag`). Because it runs through pre-commit, `pre-commit install --install-hooks` enables it — no separate `core.hooksPath` wiring, which would shadow the other installed hooks.

**If you installed the hooks before this check was added, re-run the install command.** `pre-commit` writes one hook file per type at install time, so an existing checkout has no `.git/hooks/pre-push` and gets no enforcement until you re-run:

```bash
pre-commit install --install-hooks
```

The hook resolves your GitHub login from `git config github.user`, falling back to `gh api user` and then the local part of `user.email`. The fallbacks are best-effort — setting it explicitly is the reliable option, and required if your commit email is a `users.noreply.github.com` or `first.last@` address that doesn't match your login:

```bash
git config github.user <your-github-login>
```

Protected branches (`main`, `master`, `vX.Y`), automation branches (`release-please--*`, `dependabot/*`, `copilot/*`) and release branches (`alpha/*`, `beta/*`, `rc/*`, `dev/*`) are always allowed, and pushing one needs no resolvable login at all — the hook only looks your username up when the branch it is checking is supposed to carry one.

The hook is a local convenience and can be skipped with `git push --no-verify` or `SKIP=branch-name git push`. Two cases it cannot catch, both consequences of running through pre-commit rather than as a raw git hook: pushing several refs at once validates only one of them, and pushing a branch that carries no new commits runs no hooks at all. `.github/workflows/branch_name_check.yml` covers both, as a non-blocking warning on the PR head branch — note that CI deliberately does not check the username segment against the PR author, so on that one point it is looser than the local hook.

## Contributing conventions

The full conventions live in [`AGENTS.md`](../AGENTS.md) at the repo root. The points most likely to trip up a first PR:

- **Conventional Commits with a mandatory scope.** Titles look like `type(scope): description` — e.g. `fix(cli): resolve type hinting issue`. Allowed types and scopes are defined in `.github/workflows/pr_lint.yml`.
- **Branch naming:** `<github-username>/<scope>/<short-description>` (e.g. `mdrxy/docs/architecture-guide`).
- **Tests required.** Every feature or bugfix needs unit tests under `tests/unit_tests/` (no network); integration tests go in `tests/integration_tests/`.
- **Stable public interfaces.** Avoid breaking exported signatures; add new parameters as keyword-only with defaults.
- **PRs from external contributors must link an approved issue/discussion** (see the contributing guide linked above), and the PR description fills in the repository template.

CI runs a number of gates beyond tests — Conventional Commit linting, lockfile freshness, version/extras consistency, and SDK-pin checks among them. Running `make format lint` in the package you changed and `make lock-check` from `libs/` clears the most common ones.
