# `libs/code` agent guide

`deepagents-code` is the interactive coding agent — the Textual REPL, headless `-x` mode, MCP integration, skills, sandbox bootstrap, and slash-command surface. Forked from `deepagents-cli` at the 0.1.0 split.

For monorepo-wide conventions (commit titles, lint, testing, docs, CI, benchmarks), see the root `AGENTS.md`.

## Textual (terminal UI framework)

`deepagents-code` uses [Textual](https://textual.textualize.io/).

**Key Textual resources:**

- **Guide:** https://textual.textualize.io/guide/
- **Widget gallery:** https://textual.textualize.io/widget_gallery/
- **CSS reference:** https://textual.textualize.io/styles/
- **API reference:** https://textual.textualize.io/api/

### Styled text in widgets

Prefer Textual's `Content` (`textual.content`) over Rich's `Text` for widget rendering. `Content` is immutable (like `str`) and integrates natively with Textual's rendering pipeline. Rich `Text` is still correct for code that renders via Rich's `Console.print()` (e.g., `non_interactive.py`, `main.py`).

IMPORTANT: `Content` requires **Textual's** `Style` (`textual.style.Style`) for rendering, not Rich's `Style` (`rich.style.Style`). Mixing Rich `Style` objects into `Content` spans will cause `TypeError` during widget rendering. String styles (`"bold cyan"`, `"dim"`) work for non-link styling. For links, use `TStyle(link=url)`.

**Never use f-string interpolation in Rich markup** (e.g., `f"[bold]{var}[/bold]"`). If `var` contains square brackets, the markup breaks or throws. Use `Content` methods instead:

- `Content.from_markup("[bold]$var[/bold]", var=value)` — for inline markup templates. `$var` substitution auto-escapes dynamic content. **Use when the variable is external/user-controlled** (tool args, file paths, user messages, diff content, error messages from exceptions).
- `Content.styled(text, "bold")` — single style applied to plain text. No markup parsing. Use for static strings or when the variable is internal/trusted (glyphs, ints, enum-like status values). Avoid `Content.styled(f"..{var}..", style)` when `var` is user-controlled — while `styled` doesn't parse markup, the f-string pattern is fragile and inconsistent with the `from_markup` convention.
- `Content.assemble("prefix: ", (text, "bold"), " ", other_content)` — for composing pre-built `Content` objects, `(text, style)` tuples, and plain strings. Plain strings are treated as plain text (no markup parsing). Use for structural composition, especially when parts use `TStyle(link=url)`.
- `content.join(parts)` — like `str.join()` for `Content` objects.

**Decision rule:** if the value could ever come from outside the codebase (user input, tool output, API responses, file contents), use `from_markup` with `$var`. If it's a hardcoded string, glyph, or computed int, `styled` is fine.

### `App.notify()` defaults to `markup=True`

Textual's `App.notify(message)` parses the message string as Rich markup by default. Any dynamic content (exception messages, file paths, user input, command strings) containing brackets `[]`, ANSI escape codes, or `=` will cause a `MarkupError` crash in Textual's Toast renderer. Always pass `markup=False` when the message contains f-string interpolated variables. Hardcoded string literals are safe with the default.

### Rich `console.print()` and number highlighting

`console.print()` defaults to `highlight=True`, which runs `ReprHighlighter` and auto-applies bold + cyan to any detected numbers. This visually overrides subtle styles like `dim` (bold cancels dim in most terminals). Pass `highlight=False` on any `console.print()` call where the content contains numbers and consistent dim/subtle styling matters.

### Textual patterns used in this codebase

- **Workers** (`@work` decorator) for async operations - see [Workers guide](https://textual.textualize.io/guide/workers/)
- **Message passing** for widget communication - see [Events guide](https://textual.textualize.io/guide/events/)
- **Reactive attributes** for state management - see [Reactivity guide](https://textual.textualize.io/guide/reactivity/)

### Testing Textual apps

- Use `textual.pilot` for async UI testing - see [Testing guide](https://textual.textualize.io/guide/testing/)
- Snapshot testing available for visual regression - see repo `notes/snapshot_testing.md`

## SDK dependency pin

`deepagents-code` pins an exact `deepagents==X.Y.Z` version in `pyproject.toml`. When developing features that depend on new SDK functionality, bump this pin as part of the same PR. A CI check verifies the pin matches the current SDK version at release time (unless bypassed with `dangerous-skip-sdk-pin-check`).

## Startup performance

`deepagents-code` must stay fast to launch. Never import heavy packages (e.g., `deepagents`, LangChain, LangGraph) at module level or in the argument-parsing path. These imports pull in large dependency trees and add seconds to every invocation, including trivial commands like `deepagents-code -v`.

- Keep top-level imports in `main.py` and other entry-point modules minimal.
- Defer heavy imports to the point where they are actually needed (inside functions/methods).
- To read another package's version without importing it, use `importlib.metadata.version("package-name")`.
- Feature-gate checks on the startup hot path (before background workers fire) must be lightweight — env var lookups, small file reads. Never pull in expensive modules just to decide whether to skip a feature.
- When adding logic that already exists elsewhere (e.g., editable-install detection), import the existing cached implementation rather than duplicating it.
- Features that run shell commands silently must be opt-in, never default-enabled. Gate behind an explicit env var or config key.
- Background workers that spawn subprocesses must set a timeout to avoid blocking indefinitely.

## CLI help screen

The `deepagents-code --help` screen is hand-maintained in `ui.show_help()`, separate from the argparse definitions in `main.parse_args()`. When adding a new CLI flag, update **both** files. A drift-detection test (`test_args.TestHelpScreenDrift`) fails if a flag is registered in argparse but missing from the help screen.

## Splash screen tips

When adding a user-facing CLI feature (new slash command, keybinding, workflow), add a corresponding tip to the `_TIPS` list in `deepagents_code/widgets/welcome.py`. Tips are shown randomly on startup to help users discover features. Keep tips short and action-oriented (e.g., `"Press ctrl+x to compose prompts in your external editor"`).

## Slash commands

Slash commands are defined as `SlashCommand` entries in the `COMMANDS` tuple in `deepagents_code/command_registry.py`. Each entry declares the command name, description, `bypass_tier` (queue-bypass classification), optional `hidden_keywords` for fuzzy matching, and optional `aliases`. Bypass-tier frozensets and the `SLASH_COMMANDS` autocomplete list are derived automatically — no other file should hard-code command metadata.

To add a new slash command: (1) add a `SlashCommand` entry to `COMMANDS`, (2) set the appropriate `bypass_tier`, (3) add a handler branch in `_handle_command` in `app.py`, (4) run `make lint && make test` — the drift test will catch any mismatch.

## Adding a new model provider

`deepagents-code` supports LangChain-based chat model providers as optional dependencies. To add a new provider, update these files (all entries alphabetically sorted):

1. `deepagents_code/model_config.py` — add `"provider_name": "ENV_VAR_NAME"` to `PROVIDER_API_KEY_ENV`
2. `pyproject.toml` — add `provider = ["langchain-provider>=X.Y.Z,<N.0.0"]` to `[project.optional-dependencies]` and include it in the `all-providers` composite extra
3. `tests/unit_tests/test_model_config.py` — add `assert PROVIDER_API_KEY_ENV["provider_name"] == "ENV_VAR_NAME"` to `TestProviderApiKeyEnv.test_contains_major_providers`

**Not required** unless the provider's models have a distinctive name prefix (like `gpt-*`, `claude*`, `gemini*`):

- `detect_provider()` in `config.py` — only needed for auto-detection from bare model names
- `Settings.has_*` property in `config.py` — only needed if referenced by `detect_provider()` fallback logic

Model discovery, credential checking, and UI integration are automatic once `PROVIDER_API_KEY_ENV` is populated and the `langchain-*` package is installed.
