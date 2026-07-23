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
env beats `config.toml`, and the typed default is the final fallback. A
malformed numeric/list/PTC value, an unrecognized boolean token, or a
wrong-typed TOML value is logged and falls back to the next layer rather than
raising, so a bad config never blocks startup.

Structured, user-defined config is *not* a flat scalar option and is parsed by
dedicated typed loaders elsewhere. The manifest references `[threads].columns`
and `[warnings].suppress` as `STRUCTURED` options for discovery; other tables
such as `[models.providers.*]` and `[themes.*]` are handled entirely by their
own loaders and the manifest does not enumerate them at all.

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
from typing import TYPE_CHECKING, Any, Literal, assert_never, cast, get_args

from deepagents_code import _env_vars
from deepagents_code._env_vars import classify_env_bool

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


# --- Canonical typed defaults ----------------------------------------------
# These are single sources of truth for defaults shared across the manifest and
# their runtime consumers.

INTERPRETER_ENABLE_DEFAULT = True
INTERPRETER_TIMEOUT_SECONDS_DEFAULT = 5.0
INTERPRETER_MEMORY_LIMIT_MB_DEFAULT = 64
INTERPRETER_MAX_PTC_CALLS_DEFAULT = 256
INTERPRETER_MAX_RESULT_CHARS_DEFAULT = 4000
INTERPRETER_PTC_DEFAULT: str | bool | list[str] = "safe"
INTERPRETER_PTC_ACKNOWLEDGE_UNSAFE_DEFAULT = False

RECURSION_LIMIT_DEFAULT = 2000
"""Default LangGraph `recursion_limit` for the main agent.

Single source of truth shared by the `runtime.recursion_limit` option, the
`config.config` runnable-config default, and `resolve_recursion_limit`. Raised
above the LangGraph/SDK default (`25`) to accommodate deeply nested agent graphs
in long-running sessions without hitting `GRAPH_RECURSION_LIMIT`.
"""

RECURSION_LIMIT_FLOOR = 25
"""Smallest accepted `recursion_limit`; matches the LangGraph default ceiling.

A value below this would break otherwise-valid runs, so a resolved value under
the floor is rejected and falls through to the next layer / default.
"""

RECURSION_LIMIT_CEILING = 100_000
"""Largest accepted `recursion_limit`.

Bounds the graph step budget so a mistyped or hostile override cannot request
effectively unbounded traversal. A resolved value above the ceiling is rejected
and falls through to the next layer / default.
"""

LANGSMITH_PROJECT_DEFAULT = "deepagents-code"
"""Project agent traces fall back to when no project env var is set.

Single source of truth shared by the `tracing.langsmith_project` option and
`config.get_langsmith_project_name`."""

CursorStyle = Literal["block", "underline"]
"""Visual style for the chat input cursor (a block cell or an underline)."""

CURSOR_STYLE_DEFAULT: CursorStyle = "block"
VALID_CURSOR_STYLES: frozenset[str] = frozenset(get_args(CursorStyle))
"""Allowlist derived from `CursorStyle` so the two never drift."""


class OptionKind(Enum):
    """How an option's raw env/TOML value is coerced to a typed value.

    All kinds flow through `resolve_scalar`. The scalar kinds (`BOOL`,
    `BOOL_PRESENCE`, `INT`, `FLOAT`, `STR`) are coerced inline by
    `_coerce_env`/`_coerce_toml`. `LOG_LEVEL_DELEGATE`, `SHELL_LIST_DELEGATE`,
    `SKILLS_DIRS_DELEGATE`, `PTC_DELEGATE`, and `STARTUP_MODE_DELEGATE` defer to
    bespoke parsers (their semantics — dynamic debug fallback, colon-split Path
    resolution, comma + `recommended`/`all` sentinels, and the PTC/startup-mode
    allowlists — do not compress into a generic coercion). `THEME_DELEGATE` is
    resolved separately at the top of `resolve_scalar` and never reaches the
    inline coercers. `STRUCTURED` marks user-defined tables that the scalar
    resolver only passes through for display.
    """

    BOOL = "bool"
    """Recognized truthy (`1`/`true`/`yes`/`on`) or falsy (`0`/`false`/`no`/`off`)
    tokens; an unrecognized value is logged and skipped to the next layer."""

    BOOL_PRESENCE = "bool_presence"
    """Any non-empty env value enables the flag (e.g. debug injectors)."""

    INT = "int"

    FLOAT = "float"

    STR = "str"

    LOG_LEVEL_DELEGATE = "log_level"
    """Validates log levels and resolves the default from debug mode."""

    SHELL_LIST_DELEGATE = "shell_list"
    """Delegates to `config.parse_shell_allow_list`."""

    SKILLS_DIRS_DELEGATE = "skills_dirs"
    """Delegates to `config._parse_extra_skills_dirs`."""

    PTC_DELEGATE = "ptc"
    """Delegates to `config._parse_interpreter_ptc`."""

    CURSOR_STYLE_DELEGATE = "cursor_style"
    """Validates the `[ui].cursor_style` display allowlist."""

    STARTUP_MODE_DELEGATE = "startup_mode"
    """Delegates to the `[startup].mode` runtime allowlist."""

    THEME_DELEGATE = "theme"
    """Delegates to the app theme-preference loader semantics."""

    STRUCTURED = "structured"
    """User-defined table parsed by a dedicated loader; not scalar-coerced."""


_KIND_TYPE_LABEL: dict[OptionKind, str] = {
    OptionKind.BOOL: "bool",
    OptionKind.BOOL_PRESENCE: "bool",
    OptionKind.INT: "int",
    OptionKind.FLOAT: "float",
    OptionKind.STR: "str",
    OptionKind.LOG_LEVEL_DELEGATE: "str",
    OptionKind.SHELL_LIST_DELEGATE: "list[str]",
    OptionKind.SKILLS_DIRS_DELEGATE: "list[path]",
    OptionKind.PTC_DELEGATE: "str | list[str]",
    OptionKind.CURSOR_STYLE_DELEGATE: "str",
    OptionKind.STARTUP_MODE_DELEGATE: "str",
    OptionKind.THEME_DELEGATE: "theme",
    OptionKind.STRUCTURED: "table",
}

if _KIND_TYPE_LABEL.keys() != set(OptionKind):
    # Fail at import (and in the test suite) rather than KeyError-ing from
    # `ConfigOption.type` only when an unlabeled kind happens to be rendered.
    msg = "_KIND_TYPE_LABEL is missing an OptionKind entry"
    raise RuntimeError(msg)


# Python types accepted for a `ConfigOption.default` of each scalar kind,
# enforced by `ConfigOption.__post_init__`. Delegate kinds accept their parser's
# output shape and are validated by those parsers, so they are omitted here.
_KIND_DEFAULT_TYPES: dict[OptionKind, tuple[type, ...]] = {
    OptionKind.BOOL: (bool,),
    OptionKind.BOOL_PRESENCE: (bool,),
    OptionKind.INT: (int,),
    OptionKind.FLOAT: (int, float),
    OptionKind.STR: (str,),
    OptionKind.CURSOR_STYLE_DELEGATE: (str,),
    OptionKind.STARTUP_MODE_DELEGATE: (str,),
}


@dataclass(frozen=True)
class ConfigOption:
    """One user-tunable configuration option and where it can be set."""

    key: str
    """Canonical dotted identifier used by `config get`.

    Also used as the stable display key.
    """

    group: str
    """Human-readable grouping for `config`."""

    summary: str
    """One-line description of what the option controls."""

    kind: OptionKind
    """How env/TOML values are coerced to a typed value."""

    default: Any = None
    """Typed default value, or `None` when there is no static default."""

    env_var: str | None = None
    """Primary environment variable name the loader reads, or `None`.

    For provider credentials this is the canonical name; the
    `DEEPAGENTS_CODE_` prefix override is applied dynamically at resolution time.
    """

    fallback_env_vars: tuple[str, ...] = ()
    """Secondary env vars read (in order) when `env_var` is unset.

    Read literally — no `DEEPAGENTS_CODE_` prefix logic — so `config`/`config get`
    mirror runtime fallbacks such as `get_langsmith_project_name` reading bare
    `LANGSMITH_PROJECT`.
    """

    toml_keys: tuple[str, ...] | None = None
    """Section/key path within `config.toml`, or `None`."""

    invert_toml_bool: bool = False
    """Whether a TOML bool should be negated after validation."""

    cli_flag: str | None = None
    """Representative CLI flag that sets the option, or `None`."""

    redacted: bool = False
    """Whether `config` reports only set/not-set, never the raw value.

    Named `redacted` rather than `secret` so the value (and the JSON field it
    populates) carries no credential-suggesting identifier — the flag is
    boolean metadata, and a `secret`-named value tripped CodeQL's clear-text
    logging heuristic when written to stdout.
    """

    settings_field: str | None = None
    """Name of the `Settings` attribute this option backs, or `None`.

    `None` means the option is read elsewhere inline or is descriptive.
    """

    dependency_module: str | None = None
    """Import module required to use the option, or `None`.

    `None` means the option is always available or descriptive only.
    """

    install_extra: str | None = None
    """Optional `deepagents-code[...]` extra that provides `dependency_module`."""

    provider: str | None = None
    """Provider/service name a credential option authenticates, or `None`.

    Set only for `Credentials`-group options (e.g. `"anthropic"`, `"tavily"`),
    where it is the key `/auth` stores the credential under and the name passed
    to `model_config.is_service`. Carrying it as a structured field lets
    `config`/`config get` look up the stored credential without re-parsing it out
    of `key`. `None` for every other option.
    """

    empty_env_is_false: bool = False
    """Whether an explicitly present empty env value disables a bool option."""

    def __post_init__(self) -> None:
        """Reject a `default` that contradicts `kind` at construction time.

        The manifest is a hand-edited literal table with `default: Any`, so a
        mistyped default (an `INT` option defaulting to a `str`) or a mutable
        one would otherwise slip through to runtime — a wrong-typed default
        feeds `Settings` unchecked, and a mutable default is shared by reference
        through the `get_config_options` `lru_cache` and returned verbatim by
        `resolve_scalar`. Catching it here fails the import (and the test suite).

        Raises:
            TypeError: When `fallback_env_vars` is not a tuple of non-empty
                strings, `empty_env_is_false` is set on a non-bool option,
                `default` is mutable, a `STRUCTURED` option declares a default,
                or a scalar option's default has the wrong type.
        """
        # Guard `fallback_env_vars` independently of `default` (which has its own
        # early-return path below): like `default`, it is shared by reference
        # through the `get_config_options` `lru_cache`, so a mutable value (a
        # `list`) would reintroduce the aliasing hazard the default guard exists
        # to prevent. Empty names never match any env var, so reject those too.
        if not isinstance(self.fallback_env_vars, tuple) or any(
            not isinstance(name, str) or not name for name in self.fallback_env_vars
        ):
            msg = (
                f"{self.key}: fallback_env_vars must be a tuple of non-empty "
                f"strings, got {self.fallback_env_vars!r}"
            )
            raise TypeError(msg)
        if self.empty_env_is_false and self.kind is not OptionKind.BOOL:
            msg = f"{self.key}: empty_env_is_false requires a bool option kind"
            raise TypeError(msg)

        default = self.default
        if default is None:
            if self.invert_toml_bool:
                self._validate_invert_toml_bool()
            return
        if isinstance(default, (list, dict, set)):
            msg = (
                f"{self.key}: mutable default {default!r} is unsafe under the "
                "shared lru_cache; use an immutable value (e.g. a tuple)"
            )
            raise TypeError(msg)
        if self.kind is OptionKind.STRUCTURED:
            msg = f"{self.key}: STRUCTURED options must not declare a default"
            raise TypeError(msg)
        if self.invert_toml_bool:
            self._validate_invert_toml_bool()
        expected = _KIND_DEFAULT_TYPES.get(self.kind)
        if expected is None:
            # Delegate kinds validate their own (immutable) default shapes.
            return
        # `bool` is an `int` subclass; an INT/FLOAT default must not be a bool.
        if not isinstance(default, expected) or (
            self.kind in {OptionKind.INT, OptionKind.FLOAT}
            and isinstance(default, bool)
        ):
            msg = (
                f"{self.key}: default {default!r} is not valid for kind "
                f"{self.kind.value}"
            )
            raise TypeError(msg)

    def _validate_invert_toml_bool(self) -> None:
        """Validate the inverted TOML bool marker is only used where coherent.

        Raises:
            TypeError: When the marker is used without a boolean TOML source.
        """
        if self.kind not in {OptionKind.BOOL, OptionKind.BOOL_PRESENCE}:
            msg = f"{self.key}: invert_toml_bool requires a boolean option kind"
            raise TypeError(msg)
        if self.toml_keys is None:
            msg = f"{self.key}: invert_toml_bool requires toml_keys"
            raise TypeError(msg)

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
        # `exc_info=True` preserves the TOML line/column (or permission cause):
        # a corrupt file makes every option fall back to its default, so the
        # log must say *why*, not just that the read failed.
        logger.warning(
            "Could not read config from %s; using defaults for all options",
            DEFAULT_CONFIG_PATH,
            exc_info=True,
        )
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
        classified = classify_env_bool(raw)
        if classified is None:
            # Unrecognized boolean token: log and fall through like every other
            # malformed scalar, so `config` reports the real source
            # (config.toml/default) instead of crediting the env var with a
            # value it did not actually supply.
            logger.warning("Ignoring %s=%r (expected bool)", name, raw)
            return _INVALID
        return classified
    if kind is OptionKind.BOOL_PRESENCE:
        return bool(raw)
    if kind is OptionKind.STR:
        return raw
    if kind is OptionKind.LOG_LEVEL_DELEGATE:
        from deepagents_code._debug import LOG_LEVELS

        level = raw.strip().upper()
        if level in LOG_LEVELS:
            return level
        valid = ", ".join(LOG_LEVELS)
        logger.warning("Ignoring %s=%r (expected one of %s)", name, raw, valid)
        return _INVALID
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

        try:
            return _parse_extra_skills_dirs(raw, None)
        except (ValueError, RuntimeError):
            # `Path.expanduser()` raises on an unresolvable `~user`, `.resolve()`
            # on a NUL byte; fall back rather than crash resolution/startup.
            logger.warning("Ignoring %s (could not resolve a path)", name)
            return _INVALID
    if kind is OptionKind.THEME_DELEGATE:
        # Resolved upstream in `resolve_scalar` and never reaches here; the raw
        # passthrough is a defensive fallback only.
        return raw
    if kind is OptionKind.CURSOR_STYLE_DELEGATE:
        if raw in VALID_CURSOR_STYLES:
            return raw
        logger.warning(
            "Ignoring %s=%r (expected 'block' or 'underline')",
            name,
            raw,
        )
        return _INVALID
    if kind is OptionKind.PTC_DELEGATE or kind is OptionKind.STRUCTURED:
        # Neither kind declares an `env_var`, so the `if option.env_var` guard in
        # `resolve_scalar` means this is unreachable today. If a future option
        # ever adds an env var for one of these, return `_INVALID` rather than
        # the raw string: passing an uncoerced value into a typed `Settings`
        # field (e.g. `interpreter_ptc`) would bypass the delegate parser's
        # validation. Falling back to the validated default is the safe choice.
        logger.warning("%s is not env-backed; ignoring %s=%r", option.key, name, raw)
        return _INVALID
    if kind is OptionKind.STARTUP_MODE_DELEGATE:
        from deepagents_code.model_config import VALID_STARTUP_MODES

        if raw in VALID_STARTUP_MODES:
            return raw
        logger.warning(
            "Ignoring %s=%r (expected 'manual', 'auto', or 'yolo')",
            name,
            raw,
        )
        return _INVALID
    assert_never(kind)


def _coerce_toml(option: ConfigOption, raw: object) -> object:
    """Coerce a raw TOML value by the option's kind, logging on mismatch.

    Returns:
        The typed value, or `_INVALID` when the raw value has the wrong shape.
    """
    kind = option.kind
    label = option.toml_path or option.key

    if kind in {OptionKind.BOOL, OptionKind.BOOL_PRESENCE}:
        if isinstance(raw, bool):
            return not raw if option.invert_toml_bool else raw
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

            try:
                # `raw` is a TOML list of unknown element type; the callee
                # guards each entry with `isinstance(p, str)`.
                return _parse_extra_skills_dirs(None, cast("list[str]", raw))
            except (ValueError, RuntimeError):
                # Unresolvable `~user` / NUL byte in a path string: fall back
                # rather than crash resolution.
                logger.warning(
                    "Ignoring %s in config.toml (could not resolve a path)", label
                )
                return _INVALID
    elif kind is OptionKind.PTC_DELEGATE:
        from deepagents_code.config import _parse_interpreter_ptc

        try:
            return _parse_interpreter_ptc(raw)
        except ValueError as exc:
            logger.warning("Ignoring %s in config.toml: %s", label, exc)
            return _INVALID
    elif kind is OptionKind.CURSOR_STYLE_DELEGATE:
        if isinstance(raw, str) and raw in VALID_CURSOR_STYLES:
            return raw
        logger.warning(
            "Ignoring %s=%r in config.toml (expected 'block' or 'underline')",
            label,
            raw,
        )
        return _INVALID
    elif kind is OptionKind.STARTUP_MODE_DELEGATE:
        from deepagents_code.model_config import VALID_STARTUP_MODES

        if isinstance(raw, str) and raw in VALID_STARTUP_MODES:
            return raw
        logger.warning(
            "Ignoring %s=%r in config.toml (expected 'manual', 'auto', or 'yolo')",
            label,
            raw,
        )
        return _INVALID
    elif kind is OptionKind.STRUCTURED:
        # Passed through verbatim for display; parsed by a dedicated loader.
        return raw
    elif kind is OptionKind.SHELL_LIST_DELEGATE:
        # Env-only; never read from TOML, so passed through untouched.
        return raw
    # Any other (future) kind falls through to the warning below, so a missing
    # branch logs and falls back rather than passing a raw value through.

    logger.warning(
        "Ignoring %s=%r in config.toml (expected %s)", label, raw, option.type
    )
    return _INVALID


def _resolve_theme(toml_data: dict[str, Any]) -> tuple[str, str]:
    """Resolve the active theme using the same precedence as app startup.

    Returns:
        `(theme_name, source)` for the effective Textual theme.
    """
    from deepagents_code import theme
    from deepagents_code._env_vars import THEME
    from deepagents_code.app import _resolve_terminal_mapping, _resolve_theme_name

    env_name = os.environ.get(THEME)
    if env_name is not None:
        resolved = _resolve_theme_name(env_name)
        if resolved is not None:
            return resolved, f"env ({THEME})"
        logger.warning(
            "Unknown theme '%s' in %s; falling back to default",
            env_name,
            THEME,
        )
        return theme.DEFAULT_THEME, "default"

    ui = toml_data.get("ui", {})
    if not isinstance(ui, dict):
        if ui is not None:
            logger.warning(
                "[ui] should be a table; got %s while resolving theme",
                type(ui).__name__,
            )
        return theme.DEFAULT_THEME, "default"

    resolved = _resolve_terminal_mapping(ui)
    if resolved is not None:
        term_program = os.environ.get("TERM_PROGRAM", "").strip()
        return resolved, f"config.toml [ui.terminal_themes.{term_program}]"

    saved = ui.get("theme")
    resolved = _resolve_theme_name(saved)
    if resolved is not None:
        return resolved, "config.toml [ui.theme]"
    if isinstance(saved, str):
        logger.warning("Unknown theme '%s' in config; falling back to default", saved)

    return theme.DEFAULT_THEME, "default"


def resolve_scalar(
    option: ConfigOption, *, toml_data: dict[str, Any]
) -> tuple[Any, str]:
    """Resolve an option against the environment then `config.toml`.

    Args:
        option: The option to resolve.
        toml_data: Parsed `config.toml` mapping (see `load_config_toml`).

    Resolution order is: the prefixed primary `env_var`, then each
    `fallback_env_vars` name in declaration order, then `config.toml`, then the
    typed `default`.

    Returns:
        `(value, source)`, where `source` is `env (<name>)`, `config.toml`, or
        `default`. A malformed `int`/`float`/list/PTC value, an unrecognized
        boolean token, or any TOML value of the wrong type is logged and skipped
        so the next layer (or the typed default) applies. An empty env value is
        normally treated as unset (mirroring `resolve_env_var`), so it falls
        through to the next env var, then `config.toml`/`default`, rather than
        counting as set. Options declaring `empty_env_is_false` instead resolve
        an explicitly present empty value to `False`. Theme resolution
        (`THEME_DELEGATE`) reports its own richer `config.toml [ui.*]` sources.
    """
    if option.kind is OptionKind.THEME_DELEGATE:
        return _resolve_theme(toml_data)

    if option.env_var or option.fallback_env_vars:
        from deepagents_code.model_config import resolved_env_var_name

        names: list[str] = []
        if option.env_var:
            names.append(resolved_env_var_name(option.env_var))
        names.extend(option.fallback_env_vars)
        # An empty string normally counts as unset, matching `resolve_env_var`,
        # so it is skipped and the loop continues to the next name. Options
        # with an explicitly documented empty-value opt-out declare
        # `empty_env_is_false`. Names are tried in order, so the primary
        # `env_var` wins over any fallback.
        for name in names:
            raw = os.environ.get(name)
            if raw is None:
                continue
            if not raw:
                if option.empty_env_is_false:
                    return False, f"env ({name})"
                continue
            value = _coerce_env(option, raw, name)
            if value is not _INVALID:
                return value, f"env ({name})"

    if option.toml_keys:
        found, raw = _toml_lookup(toml_data, option.toml_keys)
        if found:
            value = _coerce_toml(option, raw)
            if value is not _INVALID:
                return value, "config.toml"

    if option.kind is OptionKind.LOG_LEVEL_DELEGATE:
        from deepagents_code._env_vars import DEBUG, is_env_truthy

        return ("DEBUG" if is_env_truthy(DEBUG) else "INFO"), "default"

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


def _is_valid_recursion_limit(value: object) -> bool:
    """Return whether `value` is an accepted main-agent `recursion_limit`."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and RECURSION_LIMIT_FLOOR <= value <= RECURSION_LIMIT_CEILING
    )


def resolve_recursion_limit(*, toml_data: dict[str, Any] | None = None) -> int:
    """Resolve the effective main-agent `recursion_limit`.

    Resolves `runtime.recursion_limit` through the standard env → `config.toml`
    → default precedence. An out-of-range value (below `RECURSION_LIMIT_FLOOR`
    or above `RECURSION_LIMIT_CEILING`) is discarded with a logged warning and
    the next lower-precedence layer is tried, so a bad higher-precedence
    override cannot mask a valid TOML setting (or the default).

    Args:
        toml_data: Parsed `config.toml`; loaded automatically when omitted.

    Returns:
        The resolved recursion limit, guaranteed within
            `[RECURSION_LIMIT_FLOOR, RECURSION_LIMIT_CEILING]`.
    """
    data = load_config_toml() if toml_data is None else toml_data
    option = get_option("runtime.recursion_limit")
    if option is None:
        return RECURSION_LIMIT_DEFAULT

    value, source = resolve_scalar(option, toml_data=data)
    if _is_valid_recursion_limit(value):
        return value

    # Invalid higher-precedence values must fall through instead of jumping
    # straight to the default. Hide the rejected env var (if any) and re-resolve
    # so remaining env fallbacks, then TOML, then the typed default still apply.
    if source.startswith("env (") and source.endswith(")"):
        env_name = source[len("env (") : -1]
        logger.warning(
            "Ignoring %s recursion_limit %r (expected int in [%d, %d]); "
            "falling through to the next config source",
            source,
            value,
            RECURSION_LIMIT_FLOOR,
            RECURSION_LIMIT_CEILING,
        )
        previous = os.environ.pop(env_name, None)
        try:
            return resolve_recursion_limit(toml_data=data)
        finally:
            if previous is not None:
                os.environ[env_name] = previous

    if source != "default":
        logger.warning(
            "Ignoring %s recursion_limit %r (expected int in [%d, %d]); using %d",
            source,
            value,
            RECURSION_LIMIT_FLOOR,
            RECURSION_LIMIT_CEILING,
            RECURSION_LIMIT_DEFAULT,
        )
    return RECURSION_LIMIT_DEFAULT


# --- Option definitions -----------------------------------------------------

# Search credentials that are not provider API keys live outside
# `PROVIDER_API_KEY_ENV`, so they are declared explicitly.
_EXTRA_CREDENTIAL_ENV: dict[str, str] = {
    "tavily": "TAVILY_API_KEY",
}

_SECRET_NAME_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "APIKEY")

_PROVIDER_DEPENDENCIES: dict[str, tuple[str, str]] = {
    "anthropic": ("langchain_anthropic", "anthropic"),
    "azure_openai": ("langchain_openai", "openai"),
    "baseten": ("langchain_baseten", "baseten"),
    "bedrock": ("langchain_aws", "bedrock"),
    "cohere": ("langchain_cohere", "cohere"),
    "deepseek": ("langchain_deepseek", "deepseek"),
    "fireworks": ("langchain_fireworks", "fireworks"),
    "google_genai": ("langchain_google_genai", "google-genai"),
    "google_vertexai": ("langchain_google_vertexai", "vertex"),
    "groq": ("langchain_groq", "groq"),
    "huggingface": ("langchain_huggingface", "huggingface"),
    "ibm": ("langchain_ibm", "ibm"),
    "litellm": ("langchain_litellm", "litellm"),
    "meta": ("langchain_meta", "meta"),
    "mistralai": ("langchain_mistralai", "mistralai"),
    "nvidia": ("langchain_nvidia_ai_endpoints", "nvidia"),
    "ollama": ("langchain_ollama", "ollama"),
    "openai": ("langchain_openai", "openai"),
    "openrouter": ("langchain_openrouter", "openrouter"),
    "perplexity": ("langchain_perplexity", "perplexity"),
    "together": ("langchain_together", "together"),
    "xai": ("langchain_xai", "xai"),
}
"""Provider integration import modules and the extras that install them."""


def provider_install_extra(provider: str) -> str | None:
    """Return the `deepagents-code` extra that installs `provider`, if known.

    Args:
        provider: Provider name (e.g. `"baseten"`, `"google_genai"`).

    Returns:
        The extra name (e.g. `"baseten"`, `"google-genai"`), or `None` when the
            provider has no curated extra (custom `class_path` providers,
            ambient-auth providers, etc.).
    """
    dependency = _PROVIDER_DEPENDENCIES.get(provider)
    return dependency[1] if dependency else None


def is_provider_package_installed(provider: str) -> bool:
    """Return whether `provider`'s integration package is importable.

    Providers without a curated extra (no `_PROVIDER_DEPENDENCIES` entry) are
    reported as installed — they manage their own dependencies, so the app
    should never prompt to install an extra for them.

    Args:
        provider: Provider name (e.g. `"baseten"`).

    Returns:
        `True` when the integration package is importable or the provider has
            no curated extra; `False` when the curated package is missing or
            cannot be resolved.
    """
    import importlib.util

    dependency = _PROVIDER_DEPENDENCIES.get(provider)
    if dependency is None:
        return True
    try:
        return importlib.util.find_spec(dependency[0]) is not None
    except (ImportError, ValueError):
        # `find_spec` re-raises errors from a broken parent package and raises
        # `ValueError` for a malformed spec. Treat "can't tell" as "missing"
        # so the model selector routes to the install prompt rather than
        # crashing a synchronous Textual handler.
        logger.warning(
            "Could not resolve provider package %r; treating as not installed",
            dependency[0],
            exc_info=True,
        )
        return False


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
    upper = name.upper()
    return any(marker in upper for marker in _SECRET_NAME_MARKERS)


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
        redacted = _is_secret_env(env_var)
        summary = (
            f"Credential for the {name} provider."
            if redacted
            else f"Project/identifier for the {name} provider."
        )
        dependency = _PROVIDER_DEPENDENCIES.get(name)
        options.append(
            ConfigOption(
                key=f"credentials.{name}",
                group="Credentials",
                summary=summary,
                kind=OptionKind.STR,
                env_var=env_var,
                redacted=redacted,
                provider=name,
                settings_field=_CREDENTIAL_SETTINGS_FIELD.get(env_var),
                dependency_module=dependency[0] if dependency else None,
                install_extra=dependency[1] if dependency else None,
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
        summary="Active CLI theme from env, terminal mapping, or saved preference.",
        kind=OptionKind.THEME_DELEGATE,
        env_var=_env_vars.THEME,
        toml_keys=("ui", "theme"),
    ),
    ConfigOption(
        key="display.cursor_style",
        group="Display",
        summary="Chat input cursor style ('block' or 'underline').",
        kind=OptionKind.CURSOR_STYLE_DELEGATE,
        default=CURSOR_STYLE_DEFAULT,
        env_var=_env_vars.CURSOR_STYLE,
        toml_keys=("ui", "cursor_style"),
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
        key="display.splash_show_model",
        group="Display",
        summary="Show the active model row in the startup welcome banner.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.SPLASH_SHOW_MODEL,
    ),
    ConfigOption(
        key="display.splash_show_cwd",
        group="Display",
        summary="Show the working-directory row in the startup welcome banner.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.SPLASH_SHOW_CWD,
    ),
    ConfigOption(
        key="display.kitty_keyboard",
        group="Display",
        summary="Override kitty-keyboard detection (1 forces on, 0 forces off).",
        kind=OptionKind.BOOL,
        env_var=_env_vars.KITTY_KEYBOARD,
    ),
    ConfigOption(
        key="display.show_scrollbar",
        group="Display",
        summary="Show the vertical scrollbar in the chat area (off by default).",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.SHOW_SCROLLBAR,
        toml_keys=("ui", "show_scrollbar"),
    ),
    ConfigOption(
        key="display.debug_console_click_to_copy",
        group="Display",
        summary="Copy on click in the Ctrl+\\ Debug Console (off by default).",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.DEBUG_CONSOLE_CLICK_TO_COPY,
        toml_keys=("ui", "debug_console_click_to_copy"),
    ),
    ConfigOption(
        key="display.collapse_pastes",
        group="Display",
        summary="Collapse large chat-input pastes into compact placeholders.",
        kind=OptionKind.BOOL,
        default=True,
        env_var=_env_vars.COLLAPSE_PASTES,
        toml_keys=("ui", "collapse_pastes"),
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
        key="display.show_langsmith_replica_tracing",
        group="Display",
        summary="Show LangSmith replica project info in the startup splash.",
        kind=OptionKind.BOOL,
        default=True,
        env_var=_env_vars.SHOW_LANGSMITH_REPLICA_TRACING,
    ),
    ConfigOption(
        key="display.hide_splash_tips",
        group="Display",
        summary="Hide the startup tip shown above the chat input.",
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
    ConfigOption(
        key="display.show_url_open_toast",
        group="Display",
        summary="Show a confirmation toast after clicking a URL.",
        kind=OptionKind.BOOL,
        default=True,
        env_var=_env_vars.SHOW_URL_OPEN_TOAST,
        toml_keys=("ui", "show_url_open_toast"),
    ),
    ConfigOption(
        key="display.onboarding_integrations_screen",
        group="Display",
        summary="Show the integrations summary screen during first-run onboarding.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.ONBOARDING_INTEGRATIONS_SCREEN,
    ),
    # --- Models --------------------------------------------------------
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
    # --- Tracing -------------------------------------------------------
    ConfigOption(
        key="tracing.langsmith_project",
        group="Tracing",
        summary="LangSmith project name for deepagents agent traces.",
        kind=OptionKind.STR,
        default=LANGSMITH_PROJECT_DEFAULT,
        env_var=_env_vars.LANGSMITH_PROJECT,
        fallback_env_vars=("LANGSMITH_PROJECT",),
        settings_field="deepagents_langchain_project",
    ),
    ConfigOption(
        key="tracing.langsmith_redact",
        group="Tracing",
        summary="Redact detected secrets from LangSmith agent traces before upload.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.LANGSMITH_REDACT,
        toml_keys=("tracing", "langsmith_redact"),
    ),
    ConfigOption(
        key="tracing.user_id",
        group="Tracing",
        summary="User identifier attached to LangSmith trace metadata.",
        kind=OptionKind.STR,
        env_var=_env_vars.USER_ID,
    ),
    ConfigOption(
        key="tracing.langsmith_replica_projects",
        group="Tracing",
        summary=(
            "Extra LangSmith project to also write agent traces to. "
            "Comma-separated for forward-compatibility, but only the first "
            "project is used; the server mirrors runs to one extra project."
        ),
        kind=OptionKind.STR,
        env_var=_env_vars.LANGSMITH_REPLICA_PROJECTS,
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
        key="models.openai_prompt_cache_key",
        group="Tools",
        summary=(
            "Attach a per-thread prompt_cache_key to OpenAI-provider model calls "
            "for reliable prompt-cache routing; disable for endpoints that reject "
            "unknown request fields."
        ),
        kind=OptionKind.BOOL,
        default=True,
        env_var=_env_vars.OPENAI_PROMPT_CACHE_KEY,
        empty_env_is_false=True,
        toml_keys=("models", "openai_prompt_cache_key"),
    ),
    ConfigOption(
        key="memory.auto_save",
        group="Tools",
        summary=(
            "Let the agent proactively save learnings to memory (AGENTS.md); "
            "disable to keep loading memory but stop auto-saving."
        ),
        kind=OptionKind.BOOL,
        default=True,
        env_var=_env_vars.MEMORY_AUTO_SAVE,
        empty_env_is_false=True,
        toml_keys=("memory", "auto_save"),
    ),
    ConfigOption(
        key="features.experimental",
        group="Tools",
        summary="Opt into experimental, unstable dcode behavior.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.EXPERIMENTAL,
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
        cli_flag="--interpreter",
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
        default=True,
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
    # --- Warnings ------------------------------------------------------
    ConfigOption(
        key="warnings.suppress",
        group="Warnings",
        summary="Warning keys to suppress (e.g. 'ripgrep').",
        kind=OptionKind.STRUCTURED,
        toml_keys=("warnings", "suppress"),
    ),
    ConfigOption(
        key="warnings.suppress_env_override",
        group="Warnings",
        summary="Silence the LangSmith env-var override warning at startup.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.SUPPRESS_ENV_OVERRIDE_WARNING,
    ),
    # --- MCP ------------------------------------------------------------
    # Project trust lists are parsed by `model_config.load_mcp_server_trust_lists`,
    # which reads them only from the user-level config.toml (never a project file),
    # so they are STRUCTURED-for-discovery here rather than env-backed scalars. The
    # related env settings are named in the summaries instead of `env_var` because
    # the scalar resolver rejects env-backed STRUCTURED options by design.
    ConfigOption(
        key="mcp.enabled_project_server_approvals",
        group="MCP",
        summary=(
            "Remote project MCP approvals with fixed URLs are shared across one "
            "local Git repository's worktrees; local commands and interpolated "
            "remote URLs use the exact worktree. All include the server name and "
            "fingerprint, so edited commands/URLs or transport changes require "
            "re-approval. Process-wide "
            "name allowlist (bypasses project/fingerprint binding): "
            "DEEPAGENTS_CODE_DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS."
        ),
        kind=OptionKind.STRUCTURED,
        toml_keys=("mcp", "enabled_project_server_approvals"),
    ),
    ConfigOption(
        key="mcp.enabled_project_servers",
        group="MCP",
        summary=(
            "Deprecated legacy flat project MCP server-name allowlist; ignored in "
            "config.toml. Use enabled_project_server_approvals instead."
        ),
        kind=OptionKind.STRUCTURED,
        toml_keys=("mcp", "enabled_project_servers"),
    ),
    ConfigOption(
        key="mcp.disabled_project_servers",
        group="MCP",
        summary=(
            "Project MCP server names to always reject; reject wins over approval "
            "and trust (env: DEEPAGENTS_CODE_DISABLED_PROJECT_MCP_SERVERS)."
        ),
        kind=OptionKind.STRUCTURED,
        toml_keys=("mcp", "disabled_project_servers"),
    ),
    # Unlike the two trust lists above, this one is managed by `mcp_disabled`
    # (the server viewer's disable toggle), not `load_mcp_server_trust_lists`;
    # it plays no part in the project-trust security boundary. It is STRUCTURED
    # here purely for discovery.
    ConfigOption(
        key="mcp.disabled_servers",
        group="MCP",
        summary="MCP server names disabled by the user from the server viewer.",
        kind=OptionKind.STRUCTURED,
        toml_keys=("mcp", "disabled_servers"),
    ),
    # --- Updates --------------------------------------------------------
    ConfigOption(
        key="update.auto_update",
        group="Updates",
        summary="Enable automatic app updates.",
        kind=OptionKind.BOOL,
        default=True,
        env_var=_env_vars.AUTO_UPDATE,
        toml_keys=("update", "auto_update"),
        cli_flag="--set-auto-update",
    ),
    ConfigOption(
        key="update.no_update_check",
        group="Updates",
        summary="Disable automatic update checking.",
        kind=OptionKind.BOOL_PRESENCE,
        default=False,
        env_var=_env_vars.NO_UPDATE_CHECK,
        toml_keys=("update", "check"),
        invert_toml_bool=True,
    ),
    # --- Runtime --------------------------------------------------------
    ConfigOption(
        key="runtime.recursion_limit",
        group="Runtime",
        summary="Main agent LangGraph recursion_limit (graph step budget).",
        kind=OptionKind.INT,
        default=RECURSION_LIMIT_DEFAULT,
        env_var=_env_vars.RECURSION_LIMIT,
        toml_keys=("runtime", "recursion_limit"),
        cli_flag="--recursion-limit",
    ),
    ConfigOption(
        key="runtime.offline",
        group="Runtime",
        summary="Disable managed binary downloads and use local fallbacks.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.OFFLINE,
    ),
    ConfigOption(
        key="runtime.ripgrep_installer",
        group="Runtime",
        summary="Select ripgrep provisioning mode ('managed' or 'system').",
        kind=OptionKind.STR,
        default="managed",
        env_var=_env_vars.RIPGREP_INSTALLER,
    ),
    # --- Startup --------------------------------------------------------
    ConfigOption(
        key="startup.mode",
        group="Startup",
        summary="Default approval mode at launch ('manual', 'auto', or 'yolo').",
        kind=OptionKind.STARTUP_MODE_DELEGATE,
        default="manual",
        toml_keys=("startup", "mode"),
        cli_flag="--auto-approve",
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
        key="debug.log_level",
        group="Debug",
        summary=(
            "Minimum runtime log level (DEBUG, INFO, WARNING, ERROR, or CRITICAL); "
            "defaults to DEBUG in debug mode and INFO otherwise."
        ),
        kind=OptionKind.LOG_LEVEL_DELEGATE,
        env_var=_env_vars.LOG_LEVEL,
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
)


# Env-var constants in `_env_vars` that are not standalone options: prefixes
# and aggregates the manifest does not enumerate, plus internal/transient
# signaling flags the app sets for itself rather than reading as user config.
NON_OPTION_ENV_VARS: frozenset[str] = frozenset(
    {
        _env_vars.SERVER_ENV_PREFIX,
        # Set then popped during the self-update restart handshake (main.py);
        # never user-configured.
        _env_vars.RESTARTED_AFTER_UPDATE,
        # Env equivalents of the STRUCTURED `[mcp]` lists. They are read by the
        # dedicated `model_config.load_mcp_server_trust_lists` loader (which the
        # `mcp.*` STRUCTURED options describe for discovery), not by the scalar
        # resolver, so they intentionally have no scalar `env_var` ConfigOption.
        _env_vars.DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS,
        _env_vars.DISABLED_PROJECT_MCP_SERVERS,
        # Detection-only migration sentinel; the removed env var is not an option.
        _env_vars.LEGACY_ENABLED_PROJECT_MCP_SERVERS,
        # Plugin cache root override; read directly by plugins.store
        _env_vars.PLUGIN_CACHE_DIR,
    }
)
"""`_env_vars` constants intentionally excluded from the option catalog."""


@lru_cache(maxsize=1)
def get_config_options() -> tuple[ConfigOption, ...]:
    """Return every option, credentials-first then by domain group.

    Cached: provider credentials are generated once from `PROVIDER_API_KEY_ENV`
    on first call (which lazily imports `model_config`). The cache assumes that
    registry is an immutable module constant; a test that monkeypatches it must
    call `get_config_options.cache_clear()` (and `_options_by_key.cache_clear()`).
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
