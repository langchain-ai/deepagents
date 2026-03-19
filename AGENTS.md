# Global development guidelines for the Deep Agents monorepo

This document provides context to understand the Deep Agents Python project and assist with development.

## Project architecture and context

### Monorepo structure

This is a Python monorepo with multiple independently versioned packages that use `uv`.

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

- `uv` – Fast Python package installer and resolver (replaces pip/poetry)
- `make` – Task runner for common development commands. Feel free to look at the `Makefile` for available commands and usage patterns.
- `ruff` – Fast Python linter and formatter
- `ty` – Static type checking
- Do NOT use Sphinx-style double backtick formatting (` ``code`` `). Use single backticks (`code`) for inline code references in docstrings and comments.

#### Suppressing ruff lint rules

Prefer inline `# noqa: RULE` over `[tool.ruff.lint.per-file-ignores]` for individual exceptions. `per-file-ignores` silences a rule for the *entire* file — If you add it for one violation, all future violations of that rule in the same file are silently ignored. Inline `# noqa` is precise to the line, self-documenting, and keeps the safety net intact for the rest of the file.

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

- `pytest` – Testing framework

This monorepo uses `uv` for dependency management. Local development uses editable installs: `[tool.uv.sources]`

Each package in `libs/` has its own `pyproject.toml` and `uv.lock`.

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

#### Key config files

- pyproject.toml: Main workspace configuration with dependency groups
- uv.lock: Locked dependencies for reproducible builds
- Makefile: Development tasks

#### Commit standards

Suggest PR titles that follow Conventional Commits format. Refer to .github/workflows/pr_lint for allowed types and scopes. Note that all commit/PR titles should be in lowercase with the exception of proper nouns/named entities. All PR titles should include a scope with no exceptions. For example:

```txt
feat(sdk): add new chat completion feature
fix(cli): resolve type hinting issue
chore(evals): update infrastructure dependencies
```

- Do NOT use Sphinx-style double backtick formatting (` ``code`` `). Use single backticks (`code`) for inline code references in docstrings and comments.

#### Pull request guidelines

- Always add a disclaimer to the PR description mentioning how AI agents are involved with the contribution.
- Describe the "why" of the changes, why the proposed solution is the right one. Limit prose.
- Highlight areas of the proposed changes that require careful review.

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

## Package-specific guidance

### Deep Agents CLI (`libs/cli/`)

`deepagents-cli` uses [Textual](https://textual.textualize.io/) for its terminal UI framework.

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

**Rich `console.print()` and number highlighting:**

`console.print()` defaults to `highlight=True`, which runs `ReprHighlighter` and auto-applies bold + cyan to any detected numbers. This visually overrides subtle styles like `dim` (bold cancels dim in most terminals). Pass `highlight=False` on any `console.print()` call where the content contains numbers and consistent dim/subtle styling matters.

**Textual patterns used in this codebase:**

- **Workers** (`@work` decorator) for async operations - see [Workers guide](https://textual.textualize.io/guide/workers/)
- **Message passing** for widget communication - see [Events guide](https://textual.textualize.io/guide/events/)
- **Reactive attributes** for state management - see [Reactivity guide](https://textual.textualize.io/guide/reactivity/)

**SDK dependency pin:**

The CLI pins an exact `deepagents==X.Y.Z` version in `libs/cli/pyproject.toml`. When developing CLI features that depend on new SDK functionality, bump this pin as part of the same PR. A CI check verifies the pin matches the current SDK version at release time (unless bypassed with `dangerous-skip-sdk-pin-check`).

**Startup performance:**

The CLI must stay fast to launch. Never import heavy packages (e.g., `deepagents`, LangChain, LangGraph) at module level or in the argument-parsing path. These imports pull in large dependency trees and add seconds to every invocation, including trivial commands like `deepagents -v`.

- Keep top-level imports in `main.py` and other entry-point modules minimal.
- Defer heavy imports to the point where they are actually needed (inside functions/methods).
- To read another package's version without importing it, use `importlib.metadata.version("package-name")`.

**CLI help screen:**

The `deepagents --help` screen is hand-maintained in `ui.show_help()`, separate from the argparse definitions in `main.parse_args()`. When adding a new CLI flag, update **both** files. A drift-detection test (`test_args.TestHelpScreenDrift`) fails if a flag is registered in argparse but missing from the help screen.

**Splash screen tips:**

When adding a user-facing CLI feature (new slash command, keybinding, workflow), add a corresponding tip to the `_TIPS` list in `libs/cli/deepagents_cli/widgets/welcome.py`. Tips are shown randomly on startup to help users discover features. Keep tips short and action-oriented (e.g., `"Press ctrl+x to compose prompts in your external editor"`).

**Slash commands:**

Slash commands are defined as `SlashCommand` entries in the `COMMANDS` tuple in `libs/cli/deepagents_cli/command_registry.py`. Each entry declares the command name, description, `bypass_tier` (queue-bypass classification), optional `hidden_keywords` for fuzzy matching, and optional `aliases`. Bypass-tier frozensets and the `SLASH_COMMANDS` autocomplete list are derived automatically — no other file should hard-code command metadata.

To add a new slash command: (1) add a `SlashCommand` entry to `COMMANDS` (keep alphabetical order), (2) set the appropriate `bypass_tier`, (3) add a handler branch in `_handle_command` in `app.py`, (4) run `make lint && make test` — the drift test will catch any mismatch.

**Adding a new model provider:**

The CLI supports LangChain-based chat model providers as optional dependencies. To add a new provider, update these files (all entries alphabetically sorted):

1. `libs/cli/deepagents_cli/model_config.py` — add `"provider_name": "ENV_VAR_NAME"` to `PROVIDER_API_KEY_ENV`
2. `libs/cli/pyproject.toml` — add `provider = ["langchain-provider>=X.Y.Z,<N.0.0"]` to `[project.optional-dependencies]` and include it in the `all-providers` composite extra
3. `libs/cli/tests/unit_tests/test_model_config.py` — add `assert PROVIDER_API_KEY_ENV["provider_name"] == "ENV_VAR_NAME"` to `TestProviderApiKeyEnv.test_contains_major_providers`

**Not required** unless the provider's models have a distinctive name prefix (like `gpt-*`, `claude*`, `gemini*`):

- `detect_provider()` in `config.py` — only needed for auto-detection from bare model names
- `Settings.has_*` property in `config.py` — only needed if referenced by `detect_provider()` fallback logic

Model discovery, credential checking, and UI integration are automatic once `PROVIDER_API_KEY_ENV` is populated and the `langchain-*` package is installed.

**Building chat/streaming interfaces:**

- Blog post: [Anatomy of a Textual User Interface](https://textual.textualize.io/blog/2024/09/15/anatomy-of-a-textual-user-interface/) - demonstrates building an AI chat interface with streaming responses

**Testing Textual apps:**

- Use `textual.pilot` for async UI testing - see [Testing guide](https://textual.textualize.io/guide/testing/)
- Snapshot testing available for visual regression - see repo `notes/snapshot_testing.md`

### Evals (`libs/evals/`)

**Vendored data files:**

`libs/evals/tests/evals/tau2_airline/data/` contains vendored data from the upstream [tau-bench](https://github.com/sierra-research/tau-bench) project. These files must stay byte-identical to upstream. Pre-commit hooks (`end-of-file-fixer`, `trailing-whitespace`, `fix-smartquotes`, `fix-spaces`) are excluded from this directory in `.pre-commit-config.yaml`. Do not remove those exclusions or reformat files in this directory.

## Additional resources

- **Documentation:** https://docs.langchain.com/oss/python/deepagents/overview and source at https://github.com/langchain-ai/docs or `../docs/`. Prefer the local install and use file search tools for best results. If needed, use the docs MCP server as defined in `.mcp.json` for programmatic access.
- **Contributing Guide:** [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview)
- **CLI Release Process:** See `.github/RELEASING.md` for the full CLI release workflow (release-please, version bumping, troubleshooting failed releases, and label management).

- Do NOT use Sphinx-style double backtick formatting (` ``code`` `). Use single backticks (`code`) for inline code references in docstrings and comments.
