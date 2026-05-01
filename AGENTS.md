# Global development guidelines for the Deep Agents monorepo

This document provides context to understand the Deep Agents Python project and assist with development.

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

#### PR descriptions

The description *is* the summary — do not add a `# Summary` header.

- When the PR closes an issue, lead with the closing keyword on its own line at the very top, followed by a horizontal rule and then the body:

  ```txt
  Closes #123

  ---

  <rest of description>
  ```

  Only `Closes`, `Fixes`, and `Resolves` auto-close the referenced issue on merge. `Related:` or similar labels are informational and do not close anything.

- Explain the *why*: the motivation and why this solution is the right one. Limit prose.
- Write for readers who may be unfamiliar with this area of the codebase. Avoid insider shorthand and prefer language that is friendly to public viewers — this aids interpretability.
- Do **not** cite line numbers; they go stale as soon as the file changes.
- Rarely include full file paths or filenames. Reference the affected symbol, class, or subsystem by name instead.
- Wrap class, function, method, parameter, and variable names in backticks.
- Skip dedicated "Test plan" or "Testing" sections in most cases. Mention tests only when coverage is non-obvious, risky, or otherwise notable.
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

### Deep Agents CLI (`libs/cli/`)

#### Textual (terminal UI framework)

`deepagents-cli` uses [Textual](https://textual.textualize.io/).

**Key Textual resources:**

- **Guide:** https://textual.textualize.io/guide/
- **Widget gallery:** https://textual.textualize.io/widget_gallery/
- **CSS reference:** https://textual.textualize.io/styles/
- **API reference:** https://textual.textualize.io/api/

**Styled text in widgets:**

Prefer Textual's `Content` (`textual.content`) over Rich's `Text` for widget rendering. `Content` is immutable (like `str`) and integrates natively with Textual's rendering pipeline. Rich `Text` is still correct for code that renders via Rich's `Console.print()` (e.g., `non_interactive.py`, `main.py`).

IMPORTANT: `Content` requires **Textual's** `Style` (`textual.style.Style`) for rendering, not Rich's `Style` (`rich.style.Style`). Mixing Rich `Style` objects into `Content` spans will cause `TypeError` during widget rendering. String styles (`"bold cyan"`, `"dim"`) work for non-link styling. For links, use `TStyle(link=url)`.

**Never use f-string interpolation in Rich markup** (e.g., `f"[bold]{var}[/bold]"`). If `var` contains square brackets, the markup breaks or throws. Use `Content` methods instead:

- `Content.from_markup("[bold]$var[/bold]", var=value)` — for inline markup templates. `$var` substitution auto-escapes dynamic content. **Use when the variable is external/user-controlled** (tool args, file paths, user messages, diff content, error messages from exceptions).
- `Content.styled(text, "bold")` — single style applied to plain text. No markup parsing. Use for static strings or when the variable is internal/trusted (glyphs, ints, enum-like status values). Avoid `Content.styled(f"..{var}..", style)` when `var` is user-controlled — while `styled` doesn't parse markup, the f-string pattern is fragile and inconsistent with the `from_markup` convention.
- `Content.assemble("prefix: ", (text, "bold"), " ", other_content)` — for composing pre-built `Content` objects, `(text, style)` tuples, and plain strings. Plain strings are treated as plain text (no markup parsing). Use for structural composition, especially when parts use `TStyle(link=url)`.
- `content.join(parts)` — like `str.join()` for `Content` objects.

**Decision rule:** if the value could ever come from outside the codebase (user input, tool output, API responses, file contents), use `from_markup` with `$var`. If it's a hardcoded string, glyph, or computed int, `styled` is fine.

**`App.notify()` defaults to `markup=True`:** Textual's `App.notify(message)` parses the message string as Rich markup by default. Any dynamic content (exception messages, file paths, user input, command strings) containing brackets `[]`, ANSI escape codes, or `=` will cause a `MarkupError` crash in Textual's Toast renderer. Always pass `markup=False` when the message contains f-string interpolated variables. Hardcoded string literals are safe with the default.

**Rich `console.print()` and number highlighting:**

`console.print()` defaults to `highlight=True`, which runs `ReprHighlighter` and auto-applies bold + cyan to any detected numbers. This visually overrides subtle styles like `dim` (bold cancels dim in most terminals). Pass `highlight=False` on any `console.print()` call where the content contains numbers and consistent dim/subtle styling matters.

**Textual patterns used in this codebase:**

- **Workers** (`@work` decorator) for async operations - see [Workers guide](https://textual.textualize.io/guide/workers/)
- **Message passing** for widget communication - see [Events guide](https://textual.textualize.io/guide/events/)
- **Reactive attributes** for state management - see [Reactivity guide](https://textual.textualize.io/guide/reactivity/)

**Testing Textual apps:**

- Use `textual.pilot` for async UI testing - see [Testing guide](https://textual.textualize.io/guide/testing/)
- Snapshot testing available for visual regression - see repo `notes/snapshot_testing.md`

#### SDK dependency pin

The CLI pins an exact `deepagents==X.Y.Z` version in `libs/cli/pyproject.toml`. When developing CLI features that depend on new SDK functionality, bump this pin as part of the same PR. A CI check verifies the pin matches the current SDK version at release time (unless bypassed with `dangerous-skip-sdk-pin-check`).

#### Startup performance

The CLI must stay fast to launch. Never import heavy packages (e.g., `deepagents`, LangChain, LangGraph) at module level or in the argument-parsing path. These imports pull in large dependency trees and add seconds to every invocation, including trivial commands like `deepagents -v`.

- Keep top-level imports in `main.py` and other entry-point modules minimal.
- Defer heavy imports to the point where they are actually needed (inside functions/methods).
- To read another package's version without importing it, use `importlib.metadata.version("package-name")`.
- Feature-gate checks on the startup hot path (before background workers fire) must be lightweight — env var lookups, small file reads. Never pull in expensive modules just to decide whether to skip a feature.
- When adding logic that already exists elsewhere (e.g., editable-install detection), import the existing cached implementation rather than duplicating it.
- Features that run shell commands silently must be opt-in, never default-enabled. Gate behind an explicit env var or config key.
- Background workers that spawn subprocesses must set a timeout to avoid blocking indefinitely.

#### CLI help screen

The `deepagents --help` screen is hand-maintained in `ui.show_help()`, separate from the argparse definitions in `main.parse_args()`. When adding a new CLI flag, update **both** files. A drift-detection test (`test_args.TestHelpScreenDrift`) fails if a flag is registered in argparse but missing from the help screen.

#### Splash screen tips

When adding a user-facing CLI feature (new slash command, keybinding, workflow), add a corresponding tip to the `_TIPS` list in `libs/cli/deepagents_cli/widgets/welcome.py`. Tips are shown randomly on startup to help users discover features. Keep tips short and action-oriented (e.g., `"Press ctrl+x to compose prompts in your external editor"`).

#### Slash commands

Slash commands are defined as `SlashCommand` entries in the `COMMANDS` tuple in `libs/cli/deepagents_cli/command_registry.py`. Each entry declares the command name, description, `bypass_tier` (queue-bypass classification), optional `hidden_keywords` for fuzzy matching, and optional `aliases`. Bypass-tier frozensets and the `SLASH_COMMANDS` autocomplete list are derived automatically — no other file should hard-code command metadata.

To add a new slash command: (1) add a `SlashCommand` entry to `COMMANDS`, (2) set the appropriate `bypass_tier`, (3) add a handler branch in `_handle_command` in `app.py`, (4) run `make lint && make test` — the drift test will catch any mismatch.

#### Adding a new model provider

The CLI supports LangChain-based chat model providers as optional dependencies. To add a new provider, update these files (all entries alphabetically sorted):

1. `libs/cli/deepagents_cli/model_config.py` — add `"provider_name": "ENV_VAR_NAME"` to `PROVIDER_API_KEY_ENV`
2. `libs/cli/pyproject.toml` — add `provider = ["langchain-provider>=X.Y.Z,<N.0.0"]` to `[project.optional-dependencies]` and include it in the `all-providers` composite extra
3. `libs/cli/tests/unit_tests/test_model_config.py` — add `assert PROVIDER_API_KEY_ENV["provider_name"] == "ENV_VAR_NAME"` to `TestProviderApiKeyEnv.test_contains_major_providers`

**Not required** unless the provider's models have a distinctive name prefix (like `gpt-*`, `claude*`, `gemini*`):

- `detect_provider()` in `config.py` — only needed for auto-detection from bare model names
- `Settings.has_*` property in `config.py` — only needed if referenced by `detect_provider()` fallback logic

Model discovery, credential checking, and UI integration are automatic once `PROVIDER_API_KEY_ENV` is populated and the `langchain-*` package is installed.

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
make bench-all

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
- `.release-please-manifest.json` – Add initial version entry
- `.github/RELEASING.md` – Add to Managed Packages table
- `.github/workflows/harbor.yml` – Add sandbox option and credential check (sandbox-backed partners only)

### GitHub Actions & Workflows

This repository require actions to be pinned to a full-length commit SHA. Attempting to use a tag will fail. Use the `gh` cli to query. Verify tags are not annotated tag objects (which would need dereferencing).

## Additional resources

- **Documentation:** https://docs.langchain.com/oss/python/deepagents/overview and source at https://github.com/langchain-ai/docs or `../docs/`. Prefer the local install and use file search tools for best results. If needed, use the docs MCP server as defined in `.mcp.json` for programmatic access.
- **Contributing Guide:** [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview)
