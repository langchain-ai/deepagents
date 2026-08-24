# Global development guidelines for the Deep Agents monorepo

This file contains repository-wide rules for agents and contributors. Use these sources for details instead of duplicating them here:

- [`libs/DEVELOPMENT.md`](libs/DEVELOPMENT.md) — repository layout, setup, commands, pre-commit, the edit-test-lint loop, testing, and benchmarks.
- [`libs/ARCHITECTURE.md`](libs/ARCHITECTURE.md) — runtime structure and SDK starting points.
- [`.github/RELEASING.md`](.github/RELEASING.md) — releases, version lines, fan-out, CI labels, and troubleshooting.
- [LangChain contributing guide](https://docs.langchain.com/oss/python/contributing/overview) — general contribution policy.

<corridor>

## Corridor security analysis

When Corridor's `analyzePlan` tool is available, create a plan and use the tool to analyze it before generating or modifying code. Apply the resulting security guidance before writing code.

</corridor>

## Development workflow

Work inside the package you are changing; [`libs/DEVELOPMENT.md`](libs/DEVELOPMENT.md) covers environment setup (`uv`, `make`) and the edit-test-lint loop.

Do not add dependencies unless required. When adding one, justify its maintenance, adoption, and release activity.

### Suppressing ruff rules

Use inline `# noqa: RULE` with a justification for individual exceptions. Reserve `[tool.ruff.lint.per-file-ignores]` for categorical policies that apply to a whole class of files, such as tests not requiring docstrings. Do not hide a single violation with a file-wide ignore. If you cannot justify a suppression, the code is probably the problem. See [`libs/DEVELOPMENT.md`](libs/DEVELOPMENT.md#suppressing-ruff-rules) for worked examples.

## PR conventions

### Titles and scope

Follow Conventional Commits and include a scope. Allowed types and scopes are defined in `.github/workflows/pr_lint.yml`.

- Start the text after `type(scope):` with a lowercase letter unless it begins with a proper noun or named code entity.
- Wrap class, function, method, parameter, and variable names in backticks.
- Do not put Linear issue-closing markers in titles; put issue relationships in the PR body.
- For version-branch syncs, use `chore(repo): sync main into vX.Y`; `release` is a type, not a scope.
- Keep each bump-worthy PR to one releasable component. Put cross-package dependency or lockfile churn in a separate `chore(deps):` PR. See [multi-component fan-out](.github/RELEASING.md#multi-component-fan-out) and [lockfile churn fan-out](.github/RELEASING.md#lockfile-churn-fan-out).

### Branch naming

Name branches `<github-username>/<scope>/<short-description>`, where the description is brief kebab-case. Use the same scope as the PR title, except documentation-only branches may use the branch-only `docs` scope.

### PR bodies

Follow [the PR template](.github/PULL_REQUEST_TEMPLATE.md).

- An issue relationship line is optional; only `Closes`, `Fixes`, and `Resolves` auto-close issues.
- For features and behavior-changing fixes, place one plain-English user-visible summary above `---`. It is the release note; do not add a release-note heading or repeat it below the divider. Omit it for chores, refactors, and test-only changes.
- Below `---`, explain why the change is needed and why the approach is appropriate. Keep prose concise and public-reader friendly.
- Do not cite line numbers. Prefer symbols or subsystems over full paths, and format code entities with backticks.
- Add a collapsed test plan only for large or consequential changes. Call out areas needing careful review.

## Core development principles

### Public interfaces

Preserve exported function signatures, argument positions, and names. Before changing a public API:

- Check exports in `__init__.py` and usage in tests and examples.
- Add new parameters as keyword-only with defaults.
- Mark experimental features with MkDocs Material docstring warnings.
- Warn the developer about any signature change, even if it appears compatible.

### Code and documentation

- Add type hints and return types to Python code. Avoid `Any`; use a precise type and follow local patterns.
- Use Google-style docstrings for public functions ([template](libs/DEVELOPMENT.md#docstrings)). Put types in signatures, not docstrings; do not repeat defaults unless post-processing or conditional behavior changes them.
- Document public parameters, return values, and exceptions concisely, focusing on why rather than restating code.
- Use American English and single backticks for inline code; do not use Sphinx-style double backticks.
- Use descriptive variable names. Prefer a single word when it reads clearly.
- Keep functions under about 20 lines. Split a longer one into focused helpers.
- Build error text in a `msg` variable and raise with it, matching the surrounding code.
- Remove unreachable or commented-out code before committing.

When adding or updating model names in docs, examples, or defaults, verify the latest generally available IDs in the provider's official documentation. Do not rely on remembered model names.

### Testing

Every feature or bugfix needs unit coverage.

- Put network-free tests in `tests/unit_tests/` and networked tests in `tests/integration_tests/`.
- Do not add `@pytest.mark.asyncio`; packages use `asyncio_mode = "auto"`.
- Test behavior rather than duplicating implementation logic. Cover edge cases and keep tests deterministic.

#### Warnings are errors

All packages treat unaccepted pytest warnings as errors. Fix actionable warnings before adding filters. See [`libs/DEVELOPMENT.md`](libs/DEVELOPMENT.md#warnings-fail-the-suite) for how a stray warning surfaces and for the `bypass-warnings-check` label.

Keep this heading stable: package `pyproject.toml` comments cite it by name.

- Scope an expected warning to the test with `@pytest.mark.filterwarnings`; reserve package-level entries for categorical or third-party warnings and justify them.
- Prefer `default::` to `ignore::` for warnings such as `PytestUnhandledThreadExceptionWarning` and `PytestUnraisableExceptionWarning`, so failures remain visible.
- Warning filter message fields in ini files are unescaped regexes. Escape literal metacharacters and stop message prefixes before warning-text colons.
- Keep a message-scoped prefix narrow enough that it cannot swallow an adjacent warning.
- A filter may be version-specific, such as one that only fires on Python 3.14. A filter that matches nothing is harmless.

### Security and resources

Do not use `eval()`, `exec()`, or `pickle` on user-controlled input. Avoid bare `except:` blocks, clean up files, connections, sockets, and threads, and check changes for leaks or races.

## Repository routing

### SDK and dependencies

For SDK architecture and common starting points, use [`libs/ARCHITECTURE.md`](libs/ARCHITECTURE.md). Deep Agents delegates graph assembly to LangChain's `create_agent`; when investigating dependency internals, locate and read the installed dependency source directly.

### Search hygiene

Avoid broad repository searches during normal SDK work. Target these paths:

- SDK source and tests: `libs/deepagents/deepagents`, `libs/deepagents/tests`
- Coding agent: `libs/code`
- ACP: `libs/acp`
- Talon: `libs/talon`
- Evals: `libs/evals`
- Partner packages: `libs/partners/<partner>`

Exclude package `.venv` directories, hidden worktrees, `deepagents.egg-info`, generated metadata, benchmark results, and scratch files unless needed. For dependency internals, find the exact environment file instead of searching all of `site-packages`.

### Scoped guides

- [`libs/code/AGENTS.md`](libs/code/AGENTS.md) — Textual, startup performance, slash commands, providers, and the SDK pin.
- [`libs/evals/AGENTS.md`](libs/evals/AGENTS.md) — eval commands, reports, and Harbor integration.
- [`libs/partners/AGENTS.md`](libs/partners/AGENTS.md) — partner-package CI and release wiring.
- [`libs/code/DEVELOPMENT.md`](libs/code/DEVELOPMENT.md) — coding-agent setup and local development.
- [`.github/LAYOUT.md`](.github/LAYOUT.md) — map of CI workflows, composite actions, and labeling.

`deepagents-code` is the terminal coding agent launched by `dcode`.
### Benchmarks

Benchmarks live in `deepagents`, `code`, and `partners/quickjs`; other packages have no `bench` target. Use the package's `bench` and `bench-memory` Make targets rather than invoking pytest directly — they are the source of truth for local and CI invocation. See [`libs/DEVELOPMENT.md`](libs/DEVELOPMENT.md#benchmarks) for commands, thresholds, dashboards, and the nightly sweep.

## CI and releases

Use [`.github/RELEASING.md`](.github/RELEASING.md) for release-please behavior, version branches, changelog overrides, reverts, and release troubleshooting, including the CI guardrails that gate a release. Use [`.github/LAYOUT.md`](.github/LAYOUT.md) to find a workflow by what it does. Workflow files are authoritative for linting and labeling behavior.

Pin GitHub Actions to full-length commit SHAs; a tag reference is rejected. Verify whether a tag is annotated and dereference it before using its commit. Use the `gh` CLI to resolve one.

## Additional resources

- [Deep Agents documentation](https://docs.langchain.com/oss/python/deepagents/overview) — source lives in the `langchain-ai/docs` repo; a local checkout supports file search, and the docs MCP server is configured in `.mcp.json`

<!-- OPENWIKI:START -->

## OpenWiki

This repository has a generated `openwiki/` evidence index. It is optional just-in-time context, not required startup reading.

- Treat source code and tests as authoritative. A brief's unknowns and review items are verification gaps, not automatic requirements.
- Prefer the narrowest quiet validation that proves the changed behavior. Preserve complete failure output.

The scheduled OpenWiki GitHub Actions workflow refreshes the repository wiki. Do not hand-edit generated OpenWiki pages unless explicitly asked; prefer updating source code/docs and letting OpenWiki regenerate.

<!-- OPENWIKI:END -->
