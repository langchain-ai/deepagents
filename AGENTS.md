# Global development guidelines for the Deep Agents monorepo

This document provides context to understand the Deep Agents Python project and assist with development.

For environment setup, pre-commit installation, and the standard edit-test-lint loop, see [`libs/DEVELOPMENT.md`](libs/DEVELOPMENT.md). The rest of this document covers conventions and architecture reference.

## Project architecture and context

### Monorepo structure

This is a Python monorepo with multiple independently versioned packages:

```txt
deepagents/
├── libs/
│   ├── deepagents/  # SDK
│   ├── cli/         # CLI tool
│   ├── acp/         # Agent Context Protocol support
│   ├── evals/       # Evaluation suite and Harbor integration
│   └── partners/    # Integration packages
│       └── daytona/
│       └── ...
├── .github/         # CI/CD workflows and templates
└── README.md        # Information about Deep Agents
```

### Development tools & commands

- `uv` – Package installer and resolver (replaces pip/poetry)
- `make` – Task runner. Look at the `Makefile` for available commands and usage patterns.
- `ruff` – Linter and formatter
- `ty` – Static type checking

Local development uses editable installs: `[tool.uv.sources]`

```bash
# Run unit tests (no network)
make test

# Run specific test file
uv run --group test pytest tests/unit_tests/test_specific.py
```

```bash
# Lint code
make lint

# Format code
make format
```

#### Environment and dependency management

Use `uv` for all environment and dependency operations in this monorepo. Do not invoke `pip`, `poetry`, or `conda` directly.

- Let `uv` manage the interpreter and virtual environments — `uv sync` and `uv run` operate without manual `source .venv/bin/activate`. Do not create ad-hoc virtual environments outside the package directory.
- Each package targets its own supported Python range via its `pyproject.toml`; do not pin a global Python version. If you need an interpreter explicitly, defer to the package's `requires-python` rather than assuming system Python.
- Install dependencies explicitly through `uv sync` (optionally `--group <name>` / `--all-groups`); never let them install implicitly.
- Don't mix environments within a session, and don't add new dependencies unless strictly required — when you do, justify them (recent releases/commits, adoption).

#### Suppressing ruff lint rules

Prefer inline `# noqa: RULE` over `[tool.ruff.lint.per-file-ignores]` for individual exceptions. `per-file-ignores` silences a rule for the *entire* file — If you add it for one violation, all future violations of that rule in the same file are silently ignored. Inline `# noqa` is precise to the line, self-documenting, and keeps the safety net intact for the rest of the file. Add comments to justify silencing. If you can't make a good justification for the ignore, it is probably code smell and should be re-evaluated.

Reserve `per-file-ignores` for **categorical policy** that applies to a whole class of files (e.g., `"tests/**" = ["D1", "S101"]` — tests don't need docstrings, `assert` is expected). These are not exceptions; they are different rules for a different context.

```toml
# GOOD – categorical policy in pyproject.toml
[tool.ruff.lint.per-file-ignores]
"tests/**" = ["D1", "S101"]

# BAD – single-line exception buried in pyproject.toml
"deepagents_cli/agent.py" = ["PLR2004"]
```

```python
# GOOD – precise, self-documenting inline suppression
timeout = 30  # noqa: PLR2004  # default HTTP timeout, not arbitrary
```

#### PR and commit titles

Follow Conventional Commits. See `.github/workflows/pr_lint.yml` for allowed types and scopes. All titles must include a scope with no exceptions.

- Start the text after `type(scope):` with a lowercase letter, unless the first word is a proper noun (e.g. `Azure`, `GitHub`, `OpenAI`) or a named entity (class, function, method, parameter, or variable name).
- Wrap named entities in backticks so they render as code. Proper nouns are left unadorned.
- Keep titles short and descriptive — save detail for the body.
- Do not include Linear issue-closing markers such as `[closes DCD-52]` in PR titles. Put issue references and closing metadata in the PR description instead.
- For version-branch sync PRs, use a title like `chore(repo): sync main into vX.Y`. Do not use `release` as the scope; PR title lint reserves `release` for the type and disallows it as a scope.

Examples:

```txt
feat(sdk): add new chat completion feature
fix(cli): resolve type hinting issue
chore(evals): update infrastructure dependencies
test(cli): missing unit tests for `_git`
feat(cli): `--startup-cmd` flag
style(cli): strip trailing annotations from `ask_user` questions
```

See [PR labeling and linting](#pr-labeling-and-linting) for more info.

#### Branch naming

Branches should be prefixed `<github-username>/<scope>/<short-description>`:

- `<github-username>` — the author's GitHub login (e.g. `mdrxy`).
- `<scope>` — the same scope used in the Conventional Commit title (`sdk`, `cli`, `code`, `evals`, `acp`, partner name, `infra`, `docs`).
- `<short-description>` — kebab-case, brief, no trailing slash.

Examples:

```txt
mdrxy/sdk/concrete-toolruntime-middleware-tools
mdrxy/code/help-screen-drift-test
mdrxy/cli/startup-cmd-flag
```

#### PR descriptions

Do not add a `# Summary` or `## Release note` heading. Use an opening block ("frontmatter") in this order:

```md
Closes #123

A high-level, plain-English summary of the user-visible change.

---

<rest of PR body>
```

- The issue or PR relationship line is optional. Use the appropriate keyword, such as `Fixes`, `Closes`, `Resolves`, `Supersedes`, `Depends on`, or `Related`. Only `Closes`, `Fixes`, and `Resolves` auto-close the referenced GitHub issue on merge.
- For net new features or behavior-changing bugfixes, include the high-level user-facing summary in this opening block. Write it in release-note-ready plain English without a label or heading. Omit it for chores, refactors, or test-only changes.
- Explain the *why* in the rest of the body: the motivation and why this solution is the right one. Limit prose.
- Write for readers who may be unfamiliar with this area of the codebase. Avoid insider shorthand and prefer language that is friendly to public viewers — this aids interpretability.
- Do **not** cite line numbers; they go stale as soon as the file changes.
- Rarely include full file paths or filenames. Reference the affected symbol, class, or subsystem by name instead.
- Wrap class, function, method, parameter, and variable names in backticks.
- Do not include a dedicated "Test plan" or "Testing" section unless the PR is large or the changes are highly consequential. When one is warranted, keep it collapsed with GitHub's `<details>` and `<summary>` elements:

  ```html
  <details>
  <summary>Test plan</summary>

  - Describe the verification performed.

  </details>
  ```
- Call out areas of the change that require careful review.

## Core development principles

### Maintain stable public interfaces

CRITICAL: Always attempt to preserve function signatures, argument positions, and names for exported/public methods. Do not make breaking changes.

You should warn the developer for any function signature changes, regardless of whether they look breaking or not.

**Before making ANY changes to public APIs:**

- Check if the function/class is exported in `__init__.py`
- Look for existing usage patterns in tests and examples
- Use keyword-only arguments for new parameters: `*, new_param: str = "default"`
- Mark experimental features clearly with docstring warnings (using MkDocs Material admonitions, like `!!! warning`)

Ask: "Would this change break someone's code if they used it last week?"

### Code quality standards

All Python code MUST include type hints and return types.

```python title="Example"
def filter_unknown_users(users: list[str], known_users: set[str]) -> list[str]:
    """Single line description of the function.

    Any additional context about the function can go here.

    Args:
        users: List of user identifiers to filter.
        known_users: Set of known/valid user identifiers.

    Returns:
        List of users that are not in the `known_users` set.
    """
```

- Use descriptive, self-explanatory variable names.
- Follow existing patterns in the codebase you're modifying
- Attempt to break up complex functions (>20 lines) into smaller, focused functions where it makes sense
- Avoid using the `any` type
- Prefer single word variable names where possible

### Testing requirements

Every new feature or bugfix MUST be covered by unit tests.

- Unit tests: `tests/unit_tests/` (no network calls allowed)
- Integration tests: `tests/integration_tests/` (network calls permitted)
- We use `pytest` as the testing framework; if in doubt, check other existing tests for examples.
- Do NOT add `@pytest.mark.asyncio` to async tests — every package sets `asyncio_mode = "auto"` in `pyproject.toml`, so pytest-asyncio discovers them automatically.
- The testing file structure should mirror the source code structure.
- Avoid mocks as much as possible
- Test actual implementation, do not duplicate logic into tests

Ensure the following:

- Does the test suite fail if your new logic is broken?
- Edge cases and error conditions are tested
- Tests are deterministic (no flaky tests)

### Security and risk assessment

- No `eval()`, `exec()`, or `pickle` on user-controlled input
- Proper exception handling (no bare `except:`) and use a `msg` variable for error messages
- Remove unreachable/commented code before committing
- Race conditions or resource leaks (file handles, sockets, threads).
- Ensure proper resource cleanup (file handles, connections)

### Documentation standards

Use Google-style docstrings with Args section for all public functions.

```python title="Example"
def send_email(to: str, msg: str, *, priority: str = "normal") -> bool:
    """Send an email to a recipient with specified priority.

    Any additional context about the function can go here.

    Args:
        to: The email address of the recipient.
        msg: The message body to send.
        priority: Email priority level.

    Returns:
        `True` if email was sent successfully, `False` otherwise.

    Raises:
        InvalidEmailError: If the email address format is invalid.
        SMTPConnectionError: If unable to connect to email server.
    """
```

- Types go in function signatures, NOT in docstrings
  - If a default is present, DO NOT repeat it in the docstring unless there is post-processing or it is set conditionally.
- Focus on "why" rather than "what" in descriptions
- Document all parameters, return values, and exceptions
- Keep descriptions concise but clear
- Ensure American English spelling (e.g., "behavior", not "behaviour")
- Do NOT use Sphinx-style double backtick formatting (` ``code`` `). Use single backticks (`code`) for inline code references in docstrings and comments.

#### Model references in docs and examples

Always use the latest generally available models when referencing LLMs in docstrings, examples, and default values. Outdated model names signal stale code and confuse users. Before writing or updating model references, look up the current model IDs from each provider's official docs (Anthropic, OpenAI, Google). Do not rely on memorized model names — they go stale quickly.

## Package-specific guidance

### Deep Agents SDK (`libs/deepagents/`)

For SDK questions about `create_deep_agent`, middleware, tools, subagents, or agent construction, start in:

- `libs/deepagents/deepagents/graph.py`
  - `create_deep_agent` is the public construction entry point.
  - It builds the Deep Agents middleware stack and delegates to `langchain.agents.create_agent(...)`.
  - The final call currently happens near the end of `create_deep_agent`, followed by `.with_config(...)` for Deep Agents metadata and recursion config.
- `libs/deepagents/deepagents/middleware/`
  - Built-in Deep Agents middleware lives here.
  - `subagents.py` handles subagent middleware and nested `create_agent` use.
  - `filesystem.py`, `skills.py`, `memory.py`, `permissions.py`, and `summarization.py` are feature-specific middleware modules.
- `libs/deepagents/tests/`
  - Unit tests for SDK behavior.

If investigating LangChain `create_agent` internals, Deep Agents usually delegates into LangChain rather than owning the graph node assembly itself. Resolve the installed dependency source directly rather than searching the whole repo.

### Search hygiene

Avoid broad repo-level `glob` / `grep` for normal SDK work. This repo contains package `.venv`s, hidden worktrees, generated metadata, and scratch files that make broad searches noisy.

Prefer targeted paths:

- SDK source: `libs/deepagents/deepagents`
- SDK tests: `libs/deepagents/tests`
- Deep Agents Code/TUI package: `libs/code` (terminal coding agent)
- CLI deploy package: `libs/cli`
- ACP package: `libs/acp`

Avoid searching these unless explicitly needed:

- `.venv/`
- `.worktrees/`
- `.claude/worktrees/`
- `deepagents.egg-info/`
- benchmark result JSONs and scratch scripts at repo root

For dependency internals, first locate the dependency file from the package environment, then read that exact file instead of grepping all `site-packages`.

### Deep Agents Code (`libs/code/`)

The `deepagents-code` package ships the interactive terminal coding agent, launched via the `dcode` console command (`dcode` is the short alias for `deepagents-code`).

See `libs/code/AGENTS.md` for package-specific guidance — Textual, startup performance, slash commands, model providers, SDK pin, help-screen drift.

### Deep Agents CLI (`libs/cli/`)

As of `deepagents-cli==0.1.0` this package contains only the deployment subcommands — `init`, `dev`, and `deploy`. The interactive Textual REPL moved to `libs/code/` (`deepagents-code`); see [Deep Agents Code](#deep-agents-code-libscode) above for Textual/widget/slash-command guidance.

#### Surface

- Entry points: `deepagents` and `deepagents-cli` console scripts → `deepagents_cli.cli_main`.
- Subcommands: `init` (scaffold project), `dev` (`langgraph dev` against a bundled project), `deploy` (`langgraph deploy` to LangGraph Platform).
- Bare `deepagents` invocations print a deprecation notice pointing at `deepagents-code` and exit non-zero.

#### Layout

- `deepagents_cli/main.py` — argparse wiring + `cli_main` dispatch.
- `deepagents_cli/deploy/` — the entire deploy/dev/init pipeline (`commands.py`, `bundler.py`, `config.py`, `templates.py`, `context_hub.py`, `frontend_dist/`).
- `deepagents_cli/config.py` — slim `_load_dotenv` helper used by deploy/dev.
- `deepagents_cli/model_config.py` — slim `resolve_env_var` helper for the `DEEPAGENTS_CLI_` env-var prefix.
- `deepagents_cli/_version.py` — `__version__` (managed by release-please).

Everything else (REPL widgets, Textual app, MCP, skills, sandbox bootstrap, agent picker, slash commands, splash tips, help-screen drift test, model-provider drift test, SDK-pin check) lived under `libs/cli/` before 0.1.0 and now lives under `libs/code/`.

### Evals (`libs/evals/`)

**Vendored data files:**

`libs/evals/tests/evals/tau2_airline/data/` contains vendored data from the upstream [tau-bench](https://github.com/sierra-research/tau-bench) project. These files must stay byte-identical to upstream. Pre-commit hooks (`end-of-file-fixer`, `trailing-whitespace`, `fix-smartquotes`, `fix-spaces`) are excluded from this directory in `.pre-commit-config.yaml`. Do not remove those exclusions or reformat files in this directory.

### Benchmarks

Each package's `Makefile` defines `bench` (walltime) and `bench-memory` (heap) targets that are the **single source of truth for the bench invocation** — both local runs and the reusable CI workflow (`.github/workflows/_benchmark.yml`) call these targets. To change how benchmarks are invoked, edit the Makefile; CI inherits the change automatically.

**Run locally:**

```bash
# Single package (same target CI invokes):
make -C libs/deepagents bench
make -C libs/cli bench

# All benched packages in one go:
make -C libs bench-all

# Existing `benchmark` target (no CodSpeed instrumentation, faster, suitable
# for ad-hoc local tuning with pytest-benchmark):
make -C libs/deepagents benchmark
```

The `bench` target adds `--codspeed`; the older `benchmark` target stays around for plain `pytest-benchmark` runs that don't need walltime profiling. `bench-memory` runs the `memory_benchmark`-marked subset and is gated in CI behind `has-memory-benchmarks: true` on the workflow input — currently set by `libs/partners/quickjs`.

**Dashboard:** https://codspeed.io/langchain-ai/deepagents — separate views per package via the upper-left selector. PR comments with performance reports are posted by the CodSpeed GitHub App when it is enabled for the repository (independent of this workflow's configuration).

**Regression thresholds:** currently 10% global, managed in the CodSpeed dashboard. Tighten per-benchmark thresholds for benches whose noise floor is well below 10% (e.g., the `create_deep_agent` construction benches in `libs/deepagents/tests/benchmarks/`) — wide thresholds will mask real regressions in tight code.

**Nightly full sweep:** `.github/workflows/_benchmark_nightly.yml` runs every benched package on a daily cron without path gating, so baselines for unchanged packages don't drift. Use `workflow_dispatch` on that workflow for an ad-hoc full sweep before bumping `pytest-codspeed` or the `CodSpeedHQ/action` SHA.

## CI/CD infrastructure

### Release process

Releases use **release-please** automation. When conventional commits land on `main`, release-please creates/updates a release PR with version bumps and CHANGELOG entries. Merging the release PR triggers `.github/workflows/release.yml` via `.github/workflows/release-please.yml`.

The release pipeline: build → unit tests against built package → publish to Test PyPI → publish to PyPI (trusted publishing/OIDC) → create GitHub release.

See `.github/RELEASING.md` for the full workflow (version bumping, pre-releases, troubleshooting failed releases, and label management).

#### Overriding a merged commit's changelog entry

See [Overriding a Merged Commit's Changelog Entry](.github/RELEASING.md#overriding-a-merged-commits-changelog-entry) in `RELEASING.md` for the workflow (when to use it, the block format, and the squash-merge caveats).

#### Reverting a merged-but-unreleased PR

See [Reverting a Merged-but-Unreleased PR](.github/RELEASING.md#reverting-a-merged-but-unreleased-pr) in `RELEASING.md` when a PR has landed on `main` but its `release(<component>): X.Y.Z` PR has not yet shipped. Covers the quiet path (override to `chore` + `chore` revert, so the entry never appears in the changelog) and the `revert:` audit-trail path.

#### Developing a new version line

See [Developing a new version line](.github/RELEASING.md#developing-a-new-version-line) in `RELEASING.md` before creating a version branch (e.g. staging `0.7` while `main` stays `0.6.x`, or maintaining `0.6.x` after `main` moves on). Branches must be named `vX.Y` to match the protection ruleset (CI-passing PRs required like `main`, but `v[0-9].*` additionally allows merge commits — `main` stays squash-only); release-please only runs on `main`; keep a staging branch current by opening forward-merge PRs from `main` (a merge commit, not squash), reserving cherry-pick for when the branch deliberately diverges; and the cutover is an admin merge-commit to `main` that preserves individual commits (don't squash) so the changelog stays itemized.

### PR labeling and linting

**Title linting** (`.github/workflows/pr_lint.yml`) – Enforces Conventional Commits format with required scope on PR titles

**Release-please parse check** (`.github/workflows/release_please_parse_check.yml`) – Runs `@conventional-commits/parser` on the would-be squash-merge message (`<title> (#<num>)\n\n<body>`) at PR time. Fails the check and posts a sticky comment with a paste-ready `BEGIN_COMMIT_OVERRIDE` block when the parser would reject the body, preventing silent changelog drops. Mirrors release-please's `preprocessCommitMessage` and `splitMessages` so per-sub-message parse failures are caught the same way release-please catches them. The parser is exact-pinned (not a semver range) and must stay in lock-step with `release-please/package.json`.

**Auto-labeling:**

- `.github/workflows/pr_labeler.yml` – Unified PR labeler (size, file, title, external/internal, contributor tier)
- `.github/workflows/pr_labeler_backfill.yml` – Manual backfill of PR labels on open PRs
- `.github/workflows/auto-label-by-package.yml` – Issue labeling by package
- `.github/workflows/tag-external-issues.yml` – Issue external/internal classification and contributor tier labeling

### Adding a new partner to CI

When adding a new partner package, update these files:

- `.github/ISSUE_TEMPLATE/bug-report.yml` – Add to Area checkbox options
- `.github/ISSUE_TEMPLATE/feature-request.yml` – Add to Area checkbox options
- `.github/ISSUE_TEMPLATE/privileged.yml` – Add to Area checkbox options
- `.github/dependabot.yml` – Add dependency update directory
- `.github/scripts/pr-labeler-config.json` – Add scope-to-label mapping and file rule
- `.github/workflows/auto-label-by-package.yml` – Add package label mapping
- `.github/workflows/ci.yml` – Add to change detection and lint/test jobs
- `.github/workflows/pr_lint.yml` – Add to allowed scopes
- `.github/workflows/release.yml` – Add to `package` input options and `setup` job mapping
- `.github/workflows/release-please.yml` – Add release detection output and trigger job
- `release-please-config.json` – Add package entry under `packages`
- `.release-please-manifest.json` – Add the latest-released baseline; for a new package whose first release should be `0.0.1`, use `0.0.0`
- `.github/RELEASING.md` – Add to Managed Packages table
- `.github/workflows/harbor.yml` – Add sandbox option and credential check (sandbox-backed partners only)

### GitHub Actions & Workflows

This repository require actions to be pinned to a full-length commit SHA. Attempting to use a tag will fail. Use the `gh` cli to query. Verify tags are not annotated tag objects (which would need dereferencing).

## Additional resources

- **Documentation:** https://docs.langchain.com/oss/python/deepagents/overview and source at https://github.com/langchain-ai/docs or `../docs/`. Prefer the local install and use file search tools for best results. If needed, use the docs MCP server as defined in `.mcp.json` for programmatic access.
- **Contributing Guide:** [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview)

<!-- OPENWIKI:START -->

## OpenWiki

This repository uses OpenWiki for recurring code documentation. Start with `openwiki/quickstart.md`, then follow its links to architecture, workflows, domain concepts, operations, integrations, testing guidance, and source maps.

The scheduled OpenWiki GitHub Actions workflow refreshes the repository wiki. Do not hand-edit generated OpenWiki pages unless explicitly asked; prefer updating source code/docs and letting OpenWiki regenerate.

<!-- OPENWIKI:END -->
