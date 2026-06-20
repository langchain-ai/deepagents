# Development

A single starting point for working in the Deep Agents monorepo: how to set up,
which commands to run, and where the rules live. For how the code is structured
at runtime, see [`ARCHITECTURE.md`](./ARCHITECTURE.md).

> [!IMPORTANT]
> Before opening a pull request, read the
> [LangChain contributing guide](https://docs.langchain.com/oss/python/contributing/overview).
> External PRs must link to an issue or discussion that a maintainer has
> approved, and the contributor must be assigned to it before the PR is opened.

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) — manages interpreters, virtual
  environments, and dependencies. Do not use `pip`, `poetry`, or `conda`.
- `make` — task runner. Every package's `Makefile` is the source of truth for
  its commands; run `make help` in any package directory to list targets.

`uv` provisions the right Python interpreter automatically, so there is no
global Python version to install or pin.

## Repository layout

This is a monorepo of independently versioned packages under `libs/`:

```txt
libs/
├── deepagents/     # Core SDK — create_deep_agent, middleware, backends
├── cli/            # Deployment CLI (init / dev / deploy)
├── code/           # Coding agent with an interactive Textual TUI
├── acp/            # Agent Client Protocol integration
├── evals/          # Evaluation suite and Harbor integration
├── talon/          # Local runtime host for long-running agents (channels, cron) — experimental
└── partners/       # Provider/sandbox integrations
    ├── daytona/
    ├── modal/
    ├── vercel/
    ├── runloop/
    └── quickjs/
```

Each package has its own `pyproject.toml`, `uv.lock`, `Makefile`, and
`README.md`. There is no root `pyproject.toml`; you work inside the package you
are changing. Local cross-package dependencies are wired as editable installs
via `[tool.uv.sources]`, so a change in `libs/deepagents` is picked up by
`libs/cli` without reinstalling.

## Setup

Work inside the package you are changing. `uv` creates and manages the virtual
environment for you — no manual `activate` needed.

```bash
cd libs/deepagents
uv sync --all-groups      # install the package + all dependency groups
```

Then run commands with `uv run ...` or via the package's `make` targets.

## Python versions

Runtime support is declared per package via `requires-python` in each
`pyproject.toml`. For dependency **locking**, the monorepo pins one interpreter
per package: `acp` resolves against Python 3.14 and every other package against
3.12. That mapping lives in the `python_version` function in
[`libs/Makefile`](./libs/Makefile) and is the source of truth for `make lock`,
`make lock-check`, and `make lock-bump`.

## Common commands

Run these from inside a package directory (e.g. `libs/deepagents`). They are
consistent across the core SDK packages (`deepagents`, `code`); smaller packages
(`cli`, `acp`) expose a subset — run `make help` to see what a given package
supports:

| Command | What it does |
| --- | --- |
| `make help` | List the package's available targets |
| `make test` | Run unit tests (no network) with coverage |
| `make test TEST_FILE=tests/unit_tests/test_foo.py` | Run a single test file |
| `make integration_test` | Run integration tests (network allowed) |
| `make lint` | Run `ruff` checks + `ty` type checking |
| `make format` | Auto-format and apply safe `ruff` fixes |
| `make type` | Run the `ty` type checker only |
| `make coverage` | Unit tests with a coverage report |

Benchmarks are package-specific: `make bench` (CodSpeed-instrumented) is defined
only in the benched packages (`deepagents`, `code`, and `quickjs`). See the
Benchmarks section of the root [`AGENTS.md`](./AGENTS.md) for the full workflow.

You can also run a specific test directly:

```bash
uv run --group test pytest tests/unit_tests/test_specific.py
```

> [!NOTE]
> Do not add `@pytest.mark.asyncio` to async tests — every package sets
> `asyncio_mode = "auto"`, so they are discovered automatically.

### Repo-wide commands

Run these from the repository root to fan out across packages:

| Command | What it does |
| --- | --- |
| `make -C libs lint` | Lint every package |
| `make -C libs format` | Format every package |
| `make -C libs lock` | Update all lockfiles |
| `make -C libs lock-check` | Verify all lockfiles are up to date |
| `make -C libs lock-bump DEP=<pkg>` | Bump one dependency across all lockfiles |

## Pre-commit hooks

The repo uses [`pre-commit`](https://pre-commit.com/) for formatting, linting,
lockfile checks, and Conventional Commit message validation:

```bash
uv tool install pre-commit   # or: pipx install pre-commit
pre-commit install --install-hooks
```

The hooks run `make format lint` for changed packages and validate commit
messages, so most CI lint failures are caught before you push.

## Contributing conventions

The full conventions live in [`AGENTS.md`](./AGENTS.md) at the repo root. The
points most likely to trip up a first PR:

- **Conventional Commits with a mandatory scope.** Titles look like
  `type(scope): description` — e.g. `fix(cli): resolve type hinting issue`.
  Allowed types and scopes are defined in `.github/workflows/pr_lint.yml`.
- **Branch naming:** `<github-username>/<scope>/<short-description>`
  (e.g. `mdrxy/docs/architecture-guide`).
- **Tests required.** Every feature or bugfix needs unit tests under
  `tests/unit_tests/` (no network); integration tests go in
  `tests/integration_tests/`.
- **Stable public interfaces.** Avoid breaking exported signatures; add new
  parameters as keyword-only with defaults.
- **PRs must link an approved issue/discussion** (see the contributing guide
  linked above), and the PR description fills in the repository template.

CI runs a number of gates beyond tests — Conventional Commit linting,
lockfile freshness, version/extras consistency, and SDK-pin checks among them.
Running `make format lint` and `make -C libs lock-check` locally before pushing
clears the most common ones.
