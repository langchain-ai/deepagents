# `libs/code` agent guide

`deepagents-code` is the interactive coding agent — the Textual REPL, headless `-x` mode, MCP integration, skills, sandbox bootstrap, and slash-command surface. Forked from `deepagents-cli` at the 0.1.0 split.

For monorepo-wide conventions (commit titles, lint, testing, docs, CI, benchmarks), see the root `AGENTS.md`. For a high-level map of the package (client/server processes, request lifecycle, module map), see `ARCHITECTURE.md`.

## Textual (terminal UI framework)

`deepagents-code` uses [Textual](https://textual.textualize.io/).

**Key Textual resources:**

- **Guide:** <https://textual.textualize.io/guide/>
- **Widget gallery:** <https://textual.textualize.io/widget_gallery/>
- **CSS reference:** <https://textual.textualize.io/styles/>
- **API reference:** <https://textual.textualize.io/api/>

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

### Glyphs and spinners — reuse, don't redefine

Charset-dependent characters and animations have **single sources of truth**. Reuse them instead of hand-rolling new ones — copies drift, and (more subtly) a hardcoded Unicode literal won't degrade to ASCII on terminals that need it.

- **Glyphs** (checkmarks, arrows, ellipsis, cursor, box-drawing, branch icon, etc.): pull from `get_glyphs()` (`config.Glyphs`). Each glyph has a Unicode and an ASCII variant; `get_glyphs()` returns the right set for the active terminal. Never inline `"✓"`, `"…"`, `"›"`, and friends.
- **Animated spinners**: reuse the `Spinner` class in `tui/widgets/loading.py`, which wraps `get_glyphs().spinner_frames` (braille on Unicode, `(-) (\) (|) (/)` on ASCII) and exposes `next_frame()`/`current_frame()`. Do **not** define your own frame tuples or interval constants for a spinner — drive a `Spinner` on a `set_interval`. The status-bar reconnect indicator (`tui/widgets/status.py`) is the reference example; it ticks at the same 0.1s cadence as `LoadingWidget`. All connection/queue progress lives in the status bar — the welcome banner deliberately stays out of it, so there are currently no bespoke spinners to carve an exception for.

### Textual patterns used in this codebase

- **Workers** (`@work` decorator) for async operations - see [Workers guide](https://textual.textualize.io/guide/workers/)
- **Message passing** for widget communication - see [Events guide](https://textual.textualize.io/guide/events/)
- **Reactive attributes** for state management - see [Reactivity guide](https://textual.textualize.io/guide/reactivity/)

### UI component organization

Apply these rules to new UI; do not treat them as a mandate to refactor existing code.

- Put root abstractions under `tui/screens/` and `tui/modals/`, and reusable components under `tui/widgets/`.
- Give a large component its own `snake_case/` directory. Export its PascalCase root class from `__init__.py` (prefer implementing it there); give large subcomponents nested directories with the same pattern, while small subcomponents stay as sibling modules.
- Keep single-use constants, helpers, business logic, and UI logic beside the component that owns them. If a helper gains a second caller, consider hoisting it to their nearest shared directory; by the third caller, extract it to a shared utility module.
- Keep components focused; UI component modules should rarely exceed 200 lines.
- Co-locate a screen's `.tcss` file with its root component and set `CSS_PATH` relative to that module. Its styles may target sibling and small nested components, but a large nested component should generally own its own stylesheet.
- Widgets cannot use `CSS_PATH`; put intrinsic, auto-scoped defaults in `DEFAULT_CSS`. The mounting screen owns the widget's sizing and placement.
- Children must not import parent components. They may import shared utilities and data models; send events up and pass state/data down.
- A `ModalScreen` root inherits Textual's default translucent backdrop (`background: $background 60%`, which degrades to transparent under ansi themes) so it dims and composites the content underneath. Don't override it with `background: transparent` unless the modal is deliberately a non-dimming popover (as some selector overlays are); doing so by accident removes the dim.

### Testing Textual apps

- Use `textual.pilot` for async UI testing - see [Testing guide](https://textual.textualize.io/guide/testing/)
- Snapshot testing available for visual regression - see repo `notes/snapshot_testing.md`
- For modal flows, test the real interaction path with keypresses when possible. Unit tests that call action methods or resume handlers directly can miss focus and modal-stack bugs.
- Do not open another modal or refocus the base chat input directly inside a modal dismiss callback. Preserve the non-blocking `push_screen(..., callback)` flow and schedule follow-up UI work with `call_after_refresh` so the dismissing modal fully unwinds first.
- Be cautious replacing `push_screen(..., callback)` with an awaited modal result inside slash-command handlers; awaiting can block the Textual message pump and break keyboard navigation in the active modal.

### Typing and test doubles

When fixing `ty` diagnostics, do not mechanically replace `# type: ignore[...]` with `cast(...)`. First try to improve the actual type shape: narrower annotations, typed futures/callbacks, covariant read-only types such as `Mapping`/`Sequence`, local mock variables, or `monkeypatch.setattr(...)`. Treat `cast("Any", ...)` as a last resort.

For Textual tests that intentionally replace concrete app methods with `MagicMock` or `AsyncMock`, prefer `monkeypatch.setattr(...)` or one small documented dynamic helper over repeated `cast("Any", app)` expressions. Assert on local mock variables instead of re-reading mocked methods from the concrete object when possible.

Casts are acceptable when the type violation is the point of the test (for example, passing a wrong runtime type to exercise defensive validation) or when a third-party overload is narrower than verified runtime behavior. In those cases, keep the cast narrowly scoped and add a short comment explaining why it is intentional.

## Input surface nomenclature

The REPL has **many** text-entry surfaces, so "the input" / "the input box" is ambiguous. Use these precise terms in code, comments, commit messages, and when talking to an agent. Each maps to a concrete class so there is a single, greppable referent.

- **Chat input** (a.k.a. the composer): the primary prompt field at the bottom of the REPL where the user types messages to the agent. The widget is `ChatInput` (`tui/widgets/chat_input.py`), a container whose bordered box has id `#input-box`. Its editable field is the `ChatTextArea` with id `#chat-input`. When an unqualified "the input box" is used, it means **this** — but prefer "chat input" to stay unambiguous.
- **Inline prompt**: a multi-line `TextArea` rendered inline in the message flow (not the chat input, not inside a modal). Base class `InlinePromptTextArea` (`tui/widgets/_inline_prompt.py`), subclassed by `AskUserTextArea` (free-text answers to agent `ask_user` questions, `ask_user.py`) and `GoalReviewTextArea` (editing goal text, `goal_review.py`).
- **Modal field**: a single-line Textual `Input` inside a modal screen. Refer to it by its owner, e.g. the *auth key field* (`auth.py`), *MCP login field* (`#ml-input`, `mcp_login.py`), or the *rejection reason field* (`approval.py`).
- **Filter input**: the single-line `Input` that filters an `OptionList`/`Select` in a picker — the *model-selector filter* (`model_selector.py`), *thread-selector filter* (`thread_selector.py`), and *MCP viewer filter* (`#mcp-filter`, `mcp_viewer.py`). A specialized modal field; call it a "filter input" when the point is filtering.

Rule of thumb: say **chat input** for the main composer and name any other surface by its owner (`<owner> field` / `<owner> filter`). Reserve bare "input box" for the chat input only.

## SDK dependency pin

`deepagents-code` pins an exact `deepagents==X.Y.Z` version in `pyproject.toml`. When developing features that depend on new SDK functionality, bump this pin as part of the same PR. A CI check verifies the pin is not older than the current SDK version at release time (unless bypassed with `dangerous-skip-sdk-pin-check`); pins ahead of the workspace SDK are allowed for intentional prerelease coordination.

For a persistent editable `dcode-dev` install that stays separate from a released install, see [Local dev installs](./DEVELOPMENT.md#local-dev-installs) in the development guide.

## Startup performance

`deepagents-code` must stay fast to launch. Never import heavy packages (e.g., `deepagents`, LangChain, LangGraph) at module level or in the argument-parsing path. These imports pull in large dependency trees and add seconds to every invocation, including trivial commands like `deepagents-code -v`.

- Keep top-level imports in `main.py` and other entry-point modules minimal.
- Defer heavy imports to the point where they are actually needed (inside functions/methods).
- To read another package's version without importing it, use `importlib.metadata.version("package-name")`.
- Feature-gate checks on the startup hot path (before background workers fire) must be lightweight — env var lookups, small file reads. Never pull in expensive modules just to decide whether to skip a feature.
- When adding logic that already exists elsewhere (e.g., editable-install detection), import the existing cached implementation rather than duplicating it.
- Features that run shell commands must never do so *silently* by default. Either gate them behind an explicit env var or config key (opt-in), or — if default-enabled — announce the action before running it and offer a documented opt-out. Auto-update is the sole default-enabled exception: it shows a one-time migration notice and skips the first install when enabled only by the opt-out default, prints the upgrade it is about to perform, is overridable via `DEEPAGENTS_CODE_AUTO_UPDATE` / `[update].auto_update`, and is always disabled for editable installs.
- Background workers that spawn subprocesses must set a timeout to avoid blocking indefinitely.

## Logging

Debug logging is configured **once**, on the `deepagents_code` package logger, by the `configure_debug_logging` call in `deepagents_code/__init__.py`. Child module loggers (`logging.getLogger(__name__)`) reach the shared debug file via propagation.

- Do **not** add per-module `configure_debug_logging(logger)` calls. They are redundant now that the package logger is configured at import, and they reintroduce the duplicate-handler problem the single-config approach solves.
- Every module should create its logger with `logging.getLogger(__name__)` so it stays inside the `deepagents_code.*` hierarchy and inherits the package handler. Don't set `logger.propagate = False` or attach your own handlers.
- The **file** handler only attaches when `DEEPAGENTS_CODE_DEBUG` is truthy; that path is a single env-var read, so it's safe on the startup hot path. See `DEVELOPMENT.md` for the `DEEPAGENTS_CODE_DEBUG` / `DEEPAGENTS_CODE_DEBUG_FILE` / `DEEPAGENTS_CODE_LOG_LEVEL` env vars.
- Separately, an **always-on in-memory ring buffer** (`_debug_buffer.install_log_buffer`, called from `__init__.py` right before `configure_debug_logging`) attaches unconditionally at import so the in-app Debug Console (`Ctrl+\`) can tail recent `deepagents_code.*` records without file logging. It captures `INFO` and above by default (or `DEEPAGENTS_CODE_LOG_LEVEL`), and may lower the package logger to `INFO` for the process lifetime. It never spills to the terminal (a handler is always found, so `lastResort` is skipped) and is bounded (a `deque` of 1000 records *per log level*), so it's cheap enough to keep on. Because it installs *before* `configure_debug_logging`, warnings that helper emits at startup are captured and visible in the console.

## CLI help screen

The `deepagents-code --help` screen is hand-maintained in `ui.show_help()`, separate from the argparse definitions in `main.parse_args()`. When adding a new CLI flag, update **both** files. A drift-detection test (`test_args.TestHelpScreenDrift`) fails if a flag is registered in argparse but missing from the help screen.

## Splash screen tips

When adding a user-facing CLI feature (new slash command, keybinding, workflow), add a corresponding entry to the `_TIPS` dict in `deepagents_code/tui/widgets/startup_tip.py`, mapping the tip text to a relative selection weight (higher = shown more often). One tip is chosen at random above the input on startup to help users discover features. Keep tips short and action-oriented (e.g., `"Press ctrl+x to compose prompts in your external editor"`).

## Slash commands

Slash commands are defined as `SlashCommand` entries in the `COMMANDS` tuple in `deepagents_code/command_registry.py`. Each entry declares the command name, description, `bypass_tier` (queue-bypass classification), optional `hidden_keywords` for fuzzy matching, and optional `aliases`. Bypass-tier frozensets and the `SLASH_COMMANDS` autocomplete list are derived automatically — no other file should hard-code command metadata.

To add a new slash command: (1) add a `SlashCommand` entry to `COMMANDS`, (2) set the appropriate `bypass_tier`, (3) add a handler branch in `_handle_command` in `app.py`, (4) run `make commands-catalog` if you changed command names, aliases, descriptions, visibility, or hidden-command metadata, then (5) run `make lint && make test` — the drift test will catch any mismatch.

## Adding a new model provider

`deepagents-code` supports LangChain-based chat model providers as optional dependencies. To add a new provider, update these files (all entries alphabetically sorted):

1. `deepagents_code/model_config.py` — add `"provider_name": "ENV_VAR_NAME"` to `PROVIDER_API_KEY_ENV`
2. `deepagents_code/model_config.py` — if the provider reads a *dedicated* endpoint env var, add `"provider_name": ("CANONICAL_BASE_URL", "ALTERNATE", ...)` to `PROVIDER_BASE_URL_ENV` (see guidelines below); omit the provider entirely when it has no provider-specific endpoint variable
3. `deepagents_code/model_config.py` — add `"provider_name": "max_retries"` (or the provider's actual retry-count constructor kwarg) to `RETRY_PARAM_BY_PROVIDER`. dcode's model-node middleware owns the retry budget, so this entry is the disable-list that forces the provider's own SDK retry loop to zero; omitting it lets the SDK retries multiply the middleware's attempts. Verify the kwarg name against the integration's constructor — leave the provider out only when it exposes no integer retry-count kwarg
4. `pyproject.toml` — add `provider = ["langchain-provider>=X.Y.Z,<N.0.0"]` to `[project.optional-dependencies]` and include it in the `all-providers` composite extra
5. `deepagents_code/tui/widgets/auth.py` — add `"provider_name": "Display Name"` to `PROVIDER_DISPLAY_NAMES` so the `/auth` UI shows a branded label instead of the title-cased key, and (if the provider issues API keys from a self-serve page) `"provider_name": "https://…"` to `PROVIDER_API_KEY_URLS` so the prompt links straight to the key page
6. `tests/unit_tests/test_model_config.py` — add `assert PROVIDER_API_KEY_ENV["provider_name"] == "ENV_VAR_NAME"` to `TestProviderApiKeyEnv.test_contains_major_providers`, and pin any `PROVIDER_BASE_URL_ENV` entry with a matching assertion

### `PROVIDER_BASE_URL_ENV` guidelines

`PROVIDER_BASE_URL_ENV` pairs a provider with the endpoint env var(s) its LangChain integration and SDK read, so a stored `/auth` endpoint resolves and an inherited gateway URL cannot leak. Before adding an entry:

- **Verify against source — never infer from naming.** Read both the `langchain-<provider>` chat model (look for `from_env` / `get_from_dict_or_env` / `secret_from_env`, or a `Field` default on the `base_url`/`endpoint` alias) and the underlying vendor SDK, and record the exact env var name each reads. The integration and the SDK often read different names (e.g. `GROQ_API_BASE` vs `GROQ_BASE_URL`).
- **Canonical name first.** Element `[0]` is written by `apply_stored_credentials` and read by `get_base_url`; every other name the integration or SDK may read goes in the tuple so it is cleared too. By convention the SDK's `*_BASE_URL`-style name is canonical and the integration's `*_API_BASE`/`*_API_URL` name is the alternate.
- **Never list another provider's shared var.** OpenAI-compatible providers (e.g. `deepseek`, `openrouter`, `together`, `xai`, `baseten`) sit on the `openai` SDK, whose only endpoint var is the shared `OPENAI_BASE_URL`. Listing it under those providers would clobber the user's real OpenAI endpoint when their credential is written or cleared — list only the provider's own var (e.g. `DEEPSEEK_API_BASE`), or nothing.
- **Omit providers with no dedicated var.** When the endpoint is a hardcoded default plus a constructor arg (`baseten`), an `api_base` arg resolved per-provider inside the library (`litellm`), or derived from the region (`google_vertexai`), leave the provider out. A `/auth` endpoint still resolves through `get_base_url`'s stored-credential step and reaches the model as the `base_url` kwarg.

**Not required** unless the provider's models have a distinctive name prefix (like `gpt-*`, `claude*`, `gemini*`):

- `detect_provider()` in `config.py` — only needed for auto-detection from bare model names
- `Settings.has_*` property in `config.py` — only needed if referenced by `detect_provider()` fallback logic

Model discovery, credential checking, and UI integration are automatic once `PROVIDER_API_KEY_ENV` is populated and the `langchain-*` package is installed.
