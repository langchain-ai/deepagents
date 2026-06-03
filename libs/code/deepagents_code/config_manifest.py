"""Canonical manifest and resolver for every user-tunable scalar config option.

This module is the single source of truth for the configuration *surface*: the
set of options, their types, typed defaults, env-var names, and `config.toml`
locations. The typed defaults for config-file-only options (notably the
`[interpreter]` section) live here as module constants, and `Settings` derives
its dataclass defaults from them — so a default is defined in exactly one place.

`resolve_scalar` is the shared resolution engine used both by the runtime
(`Settings.from_environment`) and by the `config` CLI command, so introspection
can never drift from what the app actually reads. Resolution precedence mirrors
the loaders: a `DEEPAGENTS_CODE_`-prefixed env var beats the canonical name,
env beats `config.toml`, and the typed default is the final fallback. Malformed
values are logged and fall back rather than raising, so a bad config never
blocks startup.

Structured, user-defined config — `[models.providers.*]`, `[themes.*]`,
`[threads].columns` — is *not* a flat option and is parsed by dedicated typed
loaders elsewhere; the manifest only references those sections for discovery.

Import discipline: the module top level stays stdlib + `_env_vars` only (both
light) so it is safe to import from `config.py` at class-definition time without
pulling the heavy `model_config`/agent runtime onto the startup fast path.
Anything needing `model_config` (provider credentials, the config path, env-var
prefix resolution) is imported lazily inside functions.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from deepagents_code import _env_vars
from deepagents_code._env_vars import is_env_truthy

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


# --- Canonical typed defaults ----------------------------------------------
# These are the single source of truth for `[interpreter]` defaults. The
# `Settings` dataclass references them so the default is defined once.

INTERPRETER_ENABLE_DEFAULT = False
INTERPRETER_TIMEOUT_SECONDS_DEFAULT = 5.0
INTERPRETER_MEMORY_LIMIT_MB_DEFAULT = 64
INTERPRETER_MAX_PTC_CALLS_DEFAULT = 256
INTERPRETER_MAX_RESULT_CHARS_DEFAULT = 4000
INTERPRETER_PTC_DEFAULT: str | bool | list[str] = False
INTERPRETER_PTC_ACKNOWLEDGE_UNSAFE_DEFAULT = False


class OptionKind(Enum):
    """How an option's raw env/TOML value is coerced to a typed value.

    The first five kinds are resolved generically by `resolve_scalar`. The
    `*_DELEGATE` kinds defer to a bespoke parser (their semantics — colon-split
    Path resolution, comma + `recommended`/`all` sentinels, the PTC allowlist —
    do not compress into a generic coercion). `STRUCTURED` marks user-defined
    tables that the scalar resolver only passes through for display.
    """

    BOOL = "bool"
    """Truthy env values (`1`/`true`/`yes`/`on`); falls back to the default."""
    BOOL_PRESENCE = "bool_presence"
    """Any non-empty env value enables the flag (e.g. debug injectors)."""
    INT = "int"
    FLOAT = "float"
    STR = "str"
    SHELL_LIST_DELEGATE = "shell_list"
    """Delegates to `config.parse_shell_allow_list`."""
    SKILLS_DIRS_DELEGATE = "skills_dirs"
    """Delegates to `config._parse_extra_skills_dirs`."""
    PTC_DELEGATE = "ptc"
    """Delegates to `config._parse_interpreter_ptc`."""
    STRUCTURED = "structured"
    """User-defined table parsed by a dedicated loader; not scalar-coerced."""


_KIND_TYPE_LABEL: dict[OptionKind, str] = {
    OptionKind.BOOL: "bool",
    OptionKind.BOOL_PRESENCE: "bool",
    OptionKind.INT: "int",
    OptionKind.FLOAT: "float",
    OptionKind.STR: "str",
    OptionKind.SHELL_LIST_DELEGATE: "list[str]",
    OptionKind.SKILLS_DIRS_DELEGATE: "list[path]",
    OptionKind.PTC_DELEGATE: "str | list[str]",
    OptionKind.STRUCTURED: "table",
}


@dataclass(frozen=True)
class ConfigOption:
    """One user-tunable configuration option and where it can be set.

    Args:
        key: Canonical dotted identifier used by `config get` and as the
            stable display key (e.g. `interpreter.memory_limit_mb`).
        group: Human-readable grouping for `config list`/`config show`.
        summary: One-line description of what the option controls.
        kind: How env/TOML values are coerced to a typed value.
        default: Typed default value, or `None` when there is no static default.
        env_var: Primary environment variable name the loader reads, or `None`
            for config-file-only options. For provider credentials this is the
            canonical name; the `DEEPAGENTS_CODE_` prefix override is applied
            dynamically at resolution time.
        toml_keys: Section/key path within `config.toml`, or `None`.
        cli_flag: Representative CLI flag that sets the option, or `None`.
        secret: When `True`, `config show` reports only set/not-set, never the
            raw value.
        settings_field: Name of the `Settings` attribute this option backs, or
            `None` when the option is read elsewhere (inline) or is descriptive.
    """

    key: str
    group: str
    summary: str
    kind: OptionKind
    default: Any = None
    env_var: str | None = None
    toml_keys: tuple[str, ...] | None = None
    cli_flag: str | None = None
    secret: bool = False
    settings_field: str | None = None

    @property
    def type(self) -> str:
        """Human-readable type label derived from `kind`."""
        return _KIND_TYPE_LABEL[self.kind]

    @property
    def toml_path(self) -> str | None:
        """Render `toml_keys` as a `[section].key` display string."""
        if not self.toml_keys:
            return None
        *sections, leaf = self.toml_keys
        if not sections:
            return leaf
        return f"[{'.'.join(sections)}].{leaf}"


# --- Resolution -------------------------------------------------------------

_INVALID = object()
"""Sentinel: a raw value failed coercion and the next layer should be tried."""


def load_config_toml() -> dict[str, Any]:
    """Load `~/.deepagents/config.toml`.

    Returns:
        The parsed config mapping, or `{}` when the file is absent or invalid.
    """
    import tomllib

    from deepagents_code.model_config import DEFAULT_CONFIG_PATH

    try:
        with DEFAULT_CONFIG_PATH.open("rb") as f:
            return tomllib.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, tomllib.TOMLDecodeError):
        logger.warning("Could not read config from %s", DEFAULT_CONFIG_PATH)
        return {}


def _toml_lookup(data: dict[str, Any], keys: tuple[str, ...]) -> tuple[bool, Any]:
    """Navigate nested `keys` in `data`.

    Returns:
        `(found, value)`, where `found` is `False` if any key was missing.
    """
    node: Any = data
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return False, None
        node = node[key]
    return True, node


def _coerce_env(option: ConfigOption, raw: str, name: str) -> object:
    """Coerce a raw environment-variable string by the option's kind.

    Returns:
        The typed value, or `_INVALID` when the raw value cannot be coerced.
    """
    kind = option.kind
    if kind is OptionKind.BOOL:
        return is_env_truthy(name, default=bool(option.default))
    if kind is OptionKind.BOOL_PRESENCE:
        return bool(raw)
    if kind is OptionKind.STR:
        return raw
    if kind is OptionKind.INT:
        try:
            return int(raw.strip())
        except ValueError:
            logger.warning("Ignoring %s=%r (expected int)", name, raw)
            return _INVALID
    if kind is OptionKind.FLOAT:
        try:
            return float(raw.strip())
        except ValueError:
            logger.warning("Ignoring %s=%r (expected number)", name, raw)
            return _INVALID
    if kind is OptionKind.SHELL_LIST_DELEGATE:
        from deepagents_code.config import parse_shell_allow_list

        try:
            return parse_shell_allow_list(raw)
        except ValueError:
            logger.warning("Ignoring invalid %s", name)
            return _INVALID
    if kind is OptionKind.SKILLS_DIRS_DELEGATE:
        from deepagents_code.config import _parse_extra_skills_dirs

        return _parse_extra_skills_dirs(raw, None)
    # PTC has no env var; STRUCTURED is not env-backed.
    return raw


def _coerce_toml(option: ConfigOption, raw: object) -> object:
    """Coerce a raw TOML value by the option's kind, logging on mismatch.

    Returns:
        The typed value, or `_INVALID` when the raw value has the wrong shape.
    """
    kind = option.kind
    label = option.toml_path or option.key

    if kind in {OptionKind.BOOL, OptionKind.BOOL_PRESENCE}:
        if isinstance(raw, bool):
            return raw
    elif kind is OptionKind.INT:
        if isinstance(raw, int) and not isinstance(raw, bool):
            return raw
    elif kind is OptionKind.FLOAT:
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return float(raw)
    elif kind is OptionKind.STR:
        if isinstance(raw, str):
            return raw
    elif kind is OptionKind.SKILLS_DIRS_DELEGATE:
        if isinstance(raw, list):
            from deepagents_code.config import _parse_extra_skills_dirs

            return _parse_extra_skills_dirs(None, raw)
    elif kind is OptionKind.PTC_DELEGATE:
        from deepagents_code.config import _parse_interpreter_ptc

        try:
            return _parse_interpreter_ptc(raw)
        except ValueError as exc:
            logger.warning("Ignoring %s in config.toml: %s", label, exc)
            return _INVALID
    elif kind is OptionKind.STRUCTURED:
        # Passed through verbatim for display; parsed by a dedicated loader.
        return raw
    else:  # SHELL_LIST_DELEGATE is env-only and never read from TOML.
        return raw

    logger.warning(
        "Ignoring %s=%r in config.toml (expected %s)", label, raw, option.type
    )
    return _INVALID


def resolve_scalar(
    option: ConfigOption, *, toml_data: dict[str, Any]
) -> tuple[Any, str]:
    """Resolve an option against the environment then `config.toml`.

    Args:
        option: The option to resolve.
        toml_data: Parsed `config.toml` mapping (see `load_config_toml`).

    Returns:
        `(value, source)`, where `source` is `env (<name>)`, `config.toml`, or
        `default`. Malformed env/TOML values are logged and skipped so the next
        layer (or the typed default) applies.
    """
    if option.env_var:
        from deepagents_code.model_config import resolved_env_var_name

        name = resolved_env_var_name(option.env_var)
        if name in os.environ:
            value = _coerce_env(option, os.environ[name], name)
            if value is not _INVALID:
                return value, f"env ({name})"

    if option.toml_keys:
        found, raw = _toml_lookup(toml_data, option.toml_keys)
        if found:
            value = _coerce_toml(option, raw)
            if value is not _INVALID:
                return value, "config.toml"

    return option.default, "default"


def resolve_interpreter_kwargs(
    *, toml_data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve the `[interpreter]` options into `Settings` constructor kwargs.

    Only the interpreter group is resolved through the manifest. Credentials,
    the shell allow-list, and the LangSmith project keep their dedicated
    loaders in `config.py` (their empty-string-to-`None` and reload semantics
    do not fit the generic resolver), so this stays scoped to the section whose
    defaults this module owns.

    Args:
        toml_data: Parsed `config.toml`; loaded automatically when omitted.

    Returns:
        Mapping of `Settings` field name to resolved value for the interpreter
        section, suitable for splatting into `Settings(...)`.
    """
    data = load_config_toml() if toml_data is None else toml_data
    resolved: dict[str, Any] = {}
    for option in get_config_options():
        if option.group != "Interpreter" or option.settings_field is None:
            continue
        value, _ = resolve_scalar(option, toml_data=data)
        resolved[option.settings_field] = value
    return resolved


# --- Option definitions -----------------------------------------------------

# Search credentials that are not provider API keys live outside
# `PROVIDER_API_KEY_ENV`, so they are declared explicitly.
_EXTRA_CREDENTIAL_ENV: dict[str, str] = {
    "tavily": "TAVILY_API_KEY",
}

_SECRET_NAME_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "APIKEY")

# Credentials that back a `Settings` field, keyed by canonical env var.
_CREDENTIAL_SETTINGS_FIELD: dict[str, str] = {
    "OPENAI_API_KEY": "openai_api_key",
    "ANTHROPIC_API_KEY": "anthropic_api_key",
    "GOOGLE_API_KEY": "google_api_key",
    "NVIDIA_API_KEY": "nvidia_api_key",
    "TAVILY_API_KEY": "tavily_api_key",
    "GOOGLE_CLOUD_PROJECT": "google_cloud_project",
}


def _is_secret_env(name: str) -> bool:
    """Return whether a credential env var name carries secret material."""
    return any(marker in name for marker in _SECRET_NAME_MARKERS)


def _credential_options() -> tuple[ConfigOption, ...]:
    """Build credential options from the canonical provider/key registries.

    Generating these from `PROVIDER_API_KEY_ENV` (rather than hand-listing
    them) guarantees every provider the app knows how to authenticate has a
    manifest entry, so new providers can never silently miss the config
    surface.

    Returns:
        One credential `ConfigOption` per known provider/key env var.
    """
    from deepagents_code.model_config import PROVIDER_API_KEY_ENV

    options: list[ConfigOption] = []
    seen: set[str] = set()
    sources = {**PROVIDER_API_KEY_ENV, **_EXTRA_CREDENTIAL_ENV}
    for name, env_var in sorted(sources.items()):
        if env_var in seen:
            continue
        seen.add(env_var)
        secret = _is_secret_env(env_var)
        summary = (
            f"Credential for the {name} provider."
            if secret
            else f"Project/identifier for the {name} provider."
        )
        options.append(
            ConfigOption(
                key=f"credentials.{name}",
                group="Credentials",
                summary=summary,
                kind=OptionKind.STR,
                env_var=env_var,
                secret=secret,
                settings_field=_CREDENTIAL_SETTINGS_FIELD.get(env_var),
            )
        )
    return tuple(options)


# Options with a static (non-credential) definition, grouped by domain. The
# drift test asserts every `DEEPAGENTS_CODE_*` constant in `_env_vars` appears
# here (or in `NON_OPTION_ENV_VARS`).
_STATIC_OPTIONS: tuple[ConfigOption, ...] = (
    # --- Display / UI ---------------------------------------------------
    ConfigOption(
        key="display.charset",
        group="Display",
        summary="Glyph set for the TUI ('unicode', 'ascii', or 'auto').",
        kind=OptionKind.STR,
        default="auto",
        env_var="UI_CHARSET_MODE",
    ),
    ConfigOption(
        key="display.theme",
        group="Display",
        summary="Force the CLI to launch with a specific theme name.",
        kind=OptionKind.STR,
        env_var=_env_vars.THEME,
    ),
    ConfigOption(
        key="display.show_header",
        group="Display",
        summary="Show Textual's native header bar at the top of the TUI.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.SHOW_HEADER,
    ),
    ConfigOption(
        key="display.kitty_keyboard",
        group="Display",
        summary="Override kitty-keyboard detection (1 forces on, 0 forces off).",
        kind=OptionKind.BOOL,
        env_var=_env_vars.KITTY_KEYBOARD,
    ),
    ConfigOption(
        key="display.hide_cwd",
        group="Display",
        summary="Hide local path displays in the footer and startup splash.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.HIDE_CWD,
    ),
    ConfigOption(
        key="display.hide_git_branch",
        group="Display",
        summary="Hide the current git branch in the TUI footer.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.HIDE_GIT_BRANCH,
    ),
    ConfigOption(
        key="display.hide_langsmith_tracing",
        group="Display",
        summary="Hide LangSmith tracing info in the startup splash.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.HIDE_LANGSMITH_TRACING,
    ),
    ConfigOption(
        key="display.hide_splash_tips",
        group="Display",
        summary="Hide rotating tips in the startup splash.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.HIDE_SPLASH_TIPS,
    ),
    ConfigOption(
        key="display.hide_splash_version",
        group="Display",
        summary="Hide version and local-install details in the splash screen.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.HIDE_SPLASH_VERSION,
    ),
    ConfigOption(
        key="display.no_terminal_escape",
        group="Display",
        summary="Disable all terminal escape/control sequence output.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.NO_TERMINAL_ESCAPE,
    ),
    # --- Models / Tracing ----------------------------------------------
    ConfigOption(
        key="models.default",
        group="Models",
        summary="Default model spec ('provider:model') used at launch.",
        kind=OptionKind.STR,
        toml_keys=("models", "default"),
        cli_flag="--set-default-model",
    ),
    ConfigOption(
        key="models.recent",
        group="Models",
        summary="Most recently switched-to model (managed by the app).",
        kind=OptionKind.STR,
        toml_keys=("models", "recent"),
    ),
    ConfigOption(
        key="tracing.langsmith_project",
        group="Models",
        summary="LangSmith project name for deepagents agent traces.",
        kind=OptionKind.STR,
        env_var=_env_vars.LANGSMITH_PROJECT,
        settings_field="deepagents_langchain_project",
    ),
    ConfigOption(
        key="tracing.user_id",
        group="Models",
        summary="User identifier attached to LangSmith trace metadata.",
        kind=OptionKind.STR,
        env_var=_env_vars.USER_ID,
    ),
    # --- Tools / Features ----------------------------------------------
    ConfigOption(
        key="shell.allow_list",
        group="Tools",
        summary=(
            "Shell commands allowed without approval (comma-separated, or "
            "'recommended'/'all')."
        ),
        kind=OptionKind.SHELL_LIST_DELEGATE,
        env_var=_env_vars.SHELL_ALLOW_LIST,
        cli_flag="--shell-allow-list",
        settings_field="shell_allow_list",
    ),
    ConfigOption(
        key="skills.extra_allowed_dirs",
        group="Tools",
        summary=(
            "Extra directories added to the skill symlink containment "
            "allowlist (env is colon-separated)."
        ),
        kind=OptionKind.SKILLS_DIRS_DELEGATE,
        env_var=_env_vars.EXTRA_SKILLS_DIRS,
        toml_keys=("skills", "extra_allowed_dirs"),
        settings_field="extra_skills_dirs",
    ),
    ConfigOption(
        key="models.ollama_discovery",
        group="Tools",
        summary="Toggle Ollama model and profile discovery probes.",
        kind=OptionKind.BOOL,
        default=True,
        env_var=_env_vars.OLLAMA_DISCOVERY,
    ),
    ConfigOption(
        key="events.external_socket",
        group="Tools",
        summary="Enable the local Unix-socket external event listener (experimental).",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.EXTERNAL_EVENT_SOCKET,
    ),
    ConfigOption(
        key="events.external_socket_path",
        group="Tools",
        summary="Override the default Unix-socket path for the event listener.",
        kind=OptionKind.STR,
        env_var=_env_vars.EXTERNAL_EVENT_SOCKET_PATH,
    ),
    # --- Interpreter (config.toml-only; defaults owned by this module) --
    ConfigOption(
        key="interpreter.enable_interpreter",
        group="Interpreter",
        summary="Wire the QuickJS REPL middleware into the main agent (local only).",
        kind=OptionKind.BOOL,
        default=INTERPRETER_ENABLE_DEFAULT,
        toml_keys=("interpreter", "enable_interpreter"),
        cli_flag="--enable-interpreter",
        settings_field="enable_interpreter",
    ),
    ConfigOption(
        key="interpreter.timeout_seconds",
        group="Interpreter",
        summary="Per-call wall-clock timeout for the QuickJS REPL.",
        kind=OptionKind.FLOAT,
        default=INTERPRETER_TIMEOUT_SECONDS_DEFAULT,
        toml_keys=("interpreter", "timeout_seconds"),
        settings_field="interpreter_timeout_seconds",
    ),
    ConfigOption(
        key="interpreter.memory_limit_mb",
        group="Interpreter",
        summary="QuickJS heap memory cap (MB) shared across a session.",
        kind=OptionKind.INT,
        default=INTERPRETER_MEMORY_LIMIT_MB_DEFAULT,
        toml_keys=("interpreter", "memory_limit_mb"),
        settings_field="interpreter_memory_limit_mb",
    ),
    ConfigOption(
        key="interpreter.max_ptc_calls",
        group="Interpreter",
        summary="Maximum tools.* host-bridge invocations per js_eval call.",
        kind=OptionKind.INT,
        default=INTERPRETER_MAX_PTC_CALLS_DEFAULT,
        toml_keys=("interpreter", "max_ptc_calls"),
        settings_field="interpreter_max_ptc_calls",
    ),
    ConfigOption(
        key="interpreter.max_result_chars",
        group="Interpreter",
        summary="Cap (chars) on js_eval result and stdout before truncation.",
        kind=OptionKind.INT,
        default=INTERPRETER_MAX_RESULT_CHARS_DEFAULT,
        toml_keys=("interpreter", "max_result_chars"),
        settings_field="interpreter_max_result_chars",
    ),
    ConfigOption(
        key="interpreter.ptc",
        group="Interpreter",
        summary="Programmatic tool-calling allowlist ('safe', 'all', or names).",
        kind=OptionKind.PTC_DELEGATE,
        default=INTERPRETER_PTC_DEFAULT,
        toml_keys=("interpreter", "ptc"),
        cli_flag="--interpreter-tools",
        settings_field="interpreter_ptc",
    ),
    ConfigOption(
        key="interpreter.ptc_acknowledge_unsafe",
        group="Interpreter",
        summary="Acknowledge exposing every tool when interpreter.ptc='all'.",
        kind=OptionKind.BOOL,
        default=INTERPRETER_PTC_ACKNOWLEDGE_UNSAFE_DEFAULT,
        toml_keys=("interpreter", "ptc_acknowledge_unsafe"),
        settings_field="interpreter_ptc_acknowledge_unsafe",
    ),
    # --- Threads (config.toml-only; structured column table excepted) ---
    ConfigOption(
        key="threads.relative_time",
        group="Threads",
        summary="Show thread timestamps as relative time.",
        kind=OptionKind.BOOL,
        default=False,
        toml_keys=("threads", "relative_time"),
        cli_flag="--relative",
    ),
    ConfigOption(
        key="threads.sort_order",
        group="Threads",
        summary="Default thread sort key ('updated_at' or 'created_at').",
        kind=OptionKind.STR,
        default="updated_at",
        toml_keys=("threads", "sort_order"),
        cli_flag="--sort",
    ),
    ConfigOption(
        key="threads.columns",
        group="Threads",
        summary="Per-column visibility for the threads list.",
        kind=OptionKind.STRUCTURED,
        toml_keys=("threads", "columns"),
    ),
    # --- Warnings (config.toml-only) -----------------------------------
    ConfigOption(
        key="warnings.suppress",
        group="Warnings",
        summary="Warning keys to suppress (e.g. 'ripgrep').",
        kind=OptionKind.STRUCTURED,
        toml_keys=("warnings", "suppress"),
    ),
    # --- Updates --------------------------------------------------------
    ConfigOption(
        key="update.auto_update",
        group="Updates",
        summary="Enable automatic app updates.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.AUTO_UPDATE,
        cli_flag="--set-auto-update",
    ),
    ConfigOption(
        key="update.no_update_check",
        group="Updates",
        summary="Disable automatic update checking.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.NO_UPDATE_CHECK,
    ),
    # --- Debug / Development -------------------------------------------
    ConfigOption(
        key="debug.enabled",
        group="Debug",
        summary="Enable verbose debug logging and preserve the server log.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.DEBUG,
    ),
    ConfigOption(
        key="debug.file",
        group="Debug",
        summary="Path for the debug log file.",
        kind=OptionKind.STR,
        default="/tmp/deepagents_debug.log",  # noqa: S108  # documents the app default, not a write target
        env_var=_env_vars.DEBUG_FILE,
    ),
    ConfigOption(
        key="debug.onboarding",
        group="Debug",
        summary="Force the onboarding flow to open on every interactive startup.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.DEBUG_ONBOARDING,
    ),
    ConfigOption(
        key="debug.notifications",
        group="Debug",
        summary="Inject sample missing-dependency notifications at launch.",
        kind=OptionKind.BOOL_PRESENCE,
        default=False,
        env_var=_env_vars.DEBUG_NOTIFICATIONS,
    ),
    ConfigOption(
        key="debug.update",
        group="Debug",
        summary="Inject a sample update notification and open the update modal.",
        kind=OptionKind.BOOL_PRESENCE,
        default=False,
        env_var=_env_vars.DEBUG_UPDATE,
    ),
    ConfigOption(
        key="debug.mcp_project_trust",
        group="Debug",
        summary="Force the project MCP approval prompt for manual UI testing.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.DEBUG_MCP_PROJECT_TRUST,
    ),
    ConfigOption(
        key="debug.override_startup_subheader",
        group="Debug",
        summary="Override the startup splash subheader text.",
        kind=OptionKind.STR,
        env_var=_env_vars.DANGEROUSLY_OVERRIDE_STARTUP_SUBHEADER,
    ),
)


# Env-var constants in `_env_vars` that are not standalone options: prefixes
# and aggregates the manifest deliberately does not enumerate.
NON_OPTION_ENV_VARS: frozenset[str] = frozenset({_env_vars.SERVER_ENV_PREFIX})
"""`_env_vars` constants intentionally excluded from the option catalog."""


@lru_cache(maxsize=1)
def get_config_options() -> tuple[ConfigOption, ...]:
    """Return every option, credentials-first then by domain group.

    Cached: provider credentials are generated once from `PROVIDER_API_KEY_ENV`
    on first call (which lazily imports `model_config`).
    """
    return _credential_options() + _STATIC_OPTIONS


def get_option(key: str) -> ConfigOption | None:
    """Return the manifest entry for `key`, or `None` when unknown."""
    return _options_by_key().get(key)


def option_keys() -> tuple[str, ...]:
    """Return every manifest key in definition order."""
    return tuple(opt.key for opt in get_config_options())


@lru_cache(maxsize=1)
def _options_by_key() -> dict[str, ConfigOption]:
    return {opt.key: opt for opt in get_config_options()}


def iter_groups(options: Iterable[ConfigOption]) -> list[str]:
    """Return group names from `options` in first-seen order."""
    groups: list[str] = []
    for opt in options:
        if opt.group not in groups:
            groups.append(opt.group)
    return groups
