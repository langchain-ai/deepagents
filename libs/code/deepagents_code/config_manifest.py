"""Canonical manifest and resolver for every user-tunable scalar config option.

This module is the single source of truth for the configuration *surface*: the
set of options, their types, typed defaults, env-var names, and `config.toml`
locations. The typed defaults for config-file-only options (notably the
`[interpreter]` section) live here as module constants, and `Settings` derives
its dataclass defaults from them — so a default is defined in exactly one place.

`resolve_scalar` is the shared resolution engine used both by the runtime
(`Settings.from_environment`) and by the `config` CLI command, so introspection
can never drift from what the app actually reads. Resolution precedence mirrors
the loaders: managed TOML beats `DEEPAGENTS_CODE_`-prefixed and canonical env,
env beats user `config.toml`, and the typed default is the final fallback. A
malformed numeric/list/PTC value, an unrecognized boolean token, or a
wrong-typed TOML value is logged and falls back to the next layer rather than
raising, so one bad entry does not discard valid sibling policy.

Structured, user-defined config is *not* a flat scalar option and is parsed by
dedicated typed loaders elsewhere; the manifest references those tables as
`STRUCTURED` options for discovery only. Tables that can carry credentials —
`[async_subagents]` headers, `[models.providers]` auth/`params`, and
`[sandboxes.providers]` settings — are additionally flagged `redacted`, so
`dcode config` reports their source and presence but never prints the table.

Import discipline: the module top level stays stdlib + `_env_vars` only (both
light) so it is safe to import from `config.py` at class-definition time without
pulling the heavy `model_config`/agent runtime onto the startup fast path.
Anything needing `model_config` (provider credentials, the config path, env-var
prefix resolution) is imported lazily inside functions.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from enum import Enum, StrEnum
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, get_args

from deepagents_code import _env_vars

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from deepagents_code.configuration.resolver import ResolvedValue
    from deepagents_code.configuration.types import ProviderStatus

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

AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT = 20.0
"""Default wall-clock budget for one Auto classifier decision batch.

Single source of truth shared by the manifest option, the middleware default,
and the resolver, so the three cannot drift (pinned by test).
"""

AUTO_CLASSIFIER_TIMEOUT_FLOOR = 1.0
"""Smallest accepted Auto classifier timeout.

A sanity bound, not a workable budget: at least a second is required for any
provider round trip to have a chance, and below that every gated batch would be
denied as `classifier_unavailable`. A resolved value under the floor is rejected
and falls through to the next layer / default.
"""

AUTO_CLASSIFIER_TIMEOUT_CEILING = 300.0
"""Largest accepted Auto classifier timeout.

The deadline is what stops a stalled classifier from hanging every gated tool
call indefinitely, so it stays bounded: a mistyped or hostile override cannot
effectively remove it. A resolved value above the ceiling is rejected and falls
through to the next layer / default.
"""

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

COMPACT_ON_RESUME_THRESHOLD_DEFAULT = 400_000
"""Context size above which a resumed thread is offered compaction.

Zero or negative disables the suggestion.
"""

SESSION_COST_WARNING_THRESHOLD_USD_DEFAULT = 50.0
"""Default warning threshold in USD; zero or negative disables the warning."""

COLD_CACHE_WARNING_THRESHOLD_USD_DEFAULT = 0.50
"""Default incremental re-warm cost that triggers a cold-cache warning."""

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
    `BOOL_MODE_DEFAULT`, `BOOL_PRESENCE`, `INT`, `NON_NEGATIVE_INT`, `FLOAT`,
    `STR`, and `NON_EMPTY_STR`) are coerced by the source providers behind the
    compatibility `_coerce_env`/`_coerce_toml` wrappers.
    `LOG_LEVEL_DELEGATE`, `SHELL_LIST_DELEGATE`,
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

    BOOL_MODE_DEFAULT = "bool_mode_default"
    """Same token handling as `BOOL`, but with no static default: when no env or
    TOML value applies, `resolve_scalar` derives the default from debug or
    experimental mode. Declaring a `default` is rejected at construction."""

    BOOL_PRESENCE = "bool_presence"
    """Any non-empty env value enables the flag (e.g. debug injectors)."""

    INT = "int"

    NON_NEGATIVE_INT = "non_negative_int"
    """An integer count that must be zero or greater."""

    FLOAT = "float"

    STR = "str"

    NON_EMPTY_STR = "non_empty_str"
    """A string stripped of surrounding whitespace; blank values are unset."""

    LOG_LEVEL_DELEGATE = "log_level"
    """Validates log levels and resolves the default from debug mode."""

    SHELL_LIST_DELEGATE = "shell_list"
    """Delegates to `config.parse_shell_allow_list` / `parse_shell_allow_list_items`."""

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


class MergeStrategy(StrEnum):
    """How valid provider values for one manifest option compose."""

    REPLACE = "replace"
    UNION = "union"
    DEEP_MERGE = "deep_merge"


_KIND_TYPE_LABEL: dict[OptionKind, str] = {
    OptionKind.BOOL: "bool",
    OptionKind.BOOL_MODE_DEFAULT: "bool",
    OptionKind.BOOL_PRESENCE: "bool",
    OptionKind.INT: "int",
    OptionKind.NON_NEGATIVE_INT: "int (>= 0)",
    OptionKind.FLOAT: "float",
    OptionKind.STR: "str",
    OptionKind.NON_EMPTY_STR: "non-empty str",
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
# `BOOL_MODE_DEFAULT` is omitted for the opposite reason: it must not declare a
# default at all, so there is no value here to type-check.
_KIND_DEFAULT_TYPES: dict[OptionKind, tuple[type, ...]] = {
    OptionKind.BOOL: (bool,),
    OptionKind.BOOL_PRESENCE: (bool,),
    OptionKind.INT: (int,),
    OptionKind.NON_NEGATIVE_INT: (int,),
    OptionKind.FLOAT: (int, float),
    OptionKind.STR: (str,),
    OptionKind.NON_EMPTY_STR: (str,),
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

    merge_strategy: MergeStrategy = MergeStrategy.REPLACE
    """How provider values compose after each tier has coerced its own input."""

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
        if self.empty_env_is_false and self.kind not in {
            OptionKind.BOOL,
            OptionKind.BOOL_MODE_DEFAULT,
        }:
            msg = f"{self.key}: empty_env_is_false requires a bool option kind"
            raise TypeError(msg)
        if (
            self.merge_strategy is not MergeStrategy.REPLACE
            and self.kind is not OptionKind.STRUCTURED
        ):
            msg = f"{self.key}: accumulating merge strategies require STRUCTURED"
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
        if self.kind is OptionKind.BOOL_MODE_DEFAULT:
            # `resolve_scalar` computes this kind's default from debug/experimental
            # mode and returns before reading `default`, so a declared value would
            # be dead -- yet `dcode config` still renders it, advertising a default
            # that contradicts the real one.
            msg = (
                f"{self.key}: BOOL_MODE_DEFAULT options must not declare a "
                "default; the default follows debug/experimental mode"
            )
            raise TypeError(msg)
        if self.invert_toml_bool:
            self._validate_invert_toml_bool()
        expected = _KIND_DEFAULT_TYPES.get(self.kind)
        if expected is None:
            # Delegate kinds validate their own (immutable) default shapes.
            return
        # `bool` is an `int` subclass; integer/float defaults must not be bools.
        if not isinstance(default, expected) or (
            self.kind in {OptionKind.INT, OptionKind.NON_NEGATIVE_INT, OptionKind.FLOAT}
            and isinstance(default, bool)
        ):
            msg = (
                f"{self.key}: default {default!r} is not valid for kind "
                f"{self.kind.value}"
            )
            raise TypeError(msg)
        if self.kind is OptionKind.NON_NEGATIVE_INT and default < 0:
            msg = f"{self.key}: default {default!r} must be >= 0"
            raise TypeError(msg)
        if self.kind is OptionKind.NON_EMPTY_STR and not default.strip():
            msg = f"{self.key}: default must not be blank"
            raise TypeError(msg)

    def _validate_invert_toml_bool(self) -> None:
        """Validate the inverted TOML bool marker is only used where coherent.

        Raises:
            TypeError: When the marker is used without a boolean TOML source.
        """
        if self.kind not in {
            OptionKind.BOOL,
            OptionKind.BOOL_MODE_DEFAULT,
            OptionKind.BOOL_PRESENCE,
        }:
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
    except (OSError, tomllib.TOMLDecodeError, UnicodeDecodeError):
        # `UnicodeDecodeError` is neither of the other two (it subclasses
        # `ValueError`), but a file saved as UTF-16 or holding a stray byte is
        # unreadable TOML in exactly the same way -- without it, a config that
        # is merely mis-encoded raises out of every caller instead of falling
        # back to defaults.
        # `exc_info=True` preserves the TOML line/column (or permission cause):
        # a corrupt file makes every option fall back to its default, so the
        # log must say *why*, not just that the read failed.
        logger.warning(
            "Could not read config from %s; using defaults for all options",
            DEFAULT_CONFIG_PATH,
            exc_info=True,
        )
        return {}


def load_managed_config_toml(*, refresh: bool = False) -> Mapping[str, Any]:
    """Load the fixed operating-system managed TOML source.

    Returns:
        Parsed managed mapping, or an empty mapping when unavailable.
    """
    from deepagents_code.configuration.service import get_managed_snapshot

    return get_managed_snapshot(refresh=refresh).data


_warned_non_table_paths: set[tuple[str, ...]] = set()


def _coerce_env(option: ConfigOption, raw: str, name: str) -> object:
    """Delegate environment coercion to the provider boundary.

    Returns:
        The typed value, or `_INVALID` when the raw value cannot be coerced.
    """
    from deepagents_code.configuration.providers import coerce_environment_value
    from deepagents_code.configuration.types import Found, Invalid

    result = coerce_environment_value(option, raw, name)
    if isinstance(result, Found):
        return result.value
    if isinstance(result, Invalid):
        logger.warning("%s", result.reason)
    return _INVALID


def _coerce_toml(
    option: ConfigOption, raw: object, *, source: str = "config.toml"
) -> object:
    """Delegate TOML coercion to the provider boundary.

    Returns:
        The typed value, or `_INVALID` when the raw value has the wrong shape.
    """
    from deepagents_code.configuration.providers import coerce_toml_value
    from deepagents_code.configuration.types import Found, Invalid

    result = coerce_toml_value(option, raw, source=source)
    if isinstance(result, Found):
        return result.value
    if isinstance(result, Invalid):
        logger.warning("%s", result.reason)
    return _INVALID


def _resolve_theme(
    toml_data: Mapping[str, Any], *, source: str
) -> tuple[str, str] | None:
    """Resolve a theme from one TOML layer.

    Returns:
        Theme and source, or `None` when unset or invalid.
    """
    from deepagents_code.configuration.providers import ranked_theme_toml_value
    from deepagents_code.configuration.types import (
        Found,
        Invalid,
        ProviderHealth,
        ProviderStatus,
    )

    provider = ranked_theme_toml_value(
        toml_data,
        rank=0,
        durable=True,
        status=ProviderStatus(source, None, ProviderHealth.OK),
    )
    if isinstance(provider.result, Found) and isinstance(provider.result.value, str):
        return provider.result.value, provider.status.name
    if isinstance(provider.result, Invalid):
        logger.warning("%s", provider.result.reason)
    return None


def resolve_ranked_scalar(
    option: ConfigOption,
    *,
    toml_data: Mapping[str, Any],
    managed_toml_data: Mapping[str, Any] | None = None,
    managed_status: ProviderStatus | None = None,
    user_status: ProviderStatus | None = None,
) -> ResolvedValue[object]:
    """Resolve one option through the ranked durable-mask engine.

    Consumers that need health or per-leaf provenance use this typed form;
    `resolve_scalar` preserves the established `(value, source)` compatibility
    surface by rendering from the same result.

    Args:
        option: Manifest option to resolve.
        toml_data: Parsed user `config.toml` mapping.
        managed_toml_data: Parsed managed mapping, or the process snapshot when
            omitted.
        managed_status: Health/display metadata for the supplied managed table.
        user_status: Health/display metadata for the supplied user table.

    Returns:
        A rank-keyed `ResolvedValue`.

    Raises:
        RuntimeError: If the always-present default provider is unexpectedly unset.
    """
    from deepagents_code.configuration.providers import (
        ranked_default_value,
        ranked_environment_value,
        ranked_theme_environment_value,
        ranked_theme_toml_value,
        ranked_toml_value,
    )
    from deepagents_code.configuration.resolver import (
        DEFAULT_RANK,
        ENVIRONMENT_RANK,
        MANAGED_RANK,
        USER_RANK,
        RankedProviderValue,
        resolve_ranked,
    )
    from deepagents_code.configuration.types import (
        Found,
        ProviderHealth,
        ProviderStatus,
    )

    managed_status = managed_status or ProviderStatus(
        "managed config", None, ProviderHealth.OK
    )
    user_status = user_status or ProviderStatus("config.toml", None, ProviderHealth.OK)
    managed_data = (
        load_managed_config_toml() if managed_toml_data is None else managed_toml_data
    )
    if option.kind is OptionKind.THEME_DELEGATE:
        providers = (
            ranked_theme_toml_value(
                managed_data,
                rank=MANAGED_RANK,
                durable=True,
                status=managed_status,
            ),
            ranked_theme_environment_value(os.environ, rank=ENVIRONMENT_RANK),
            ranked_theme_toml_value(
                toml_data,
                rank=USER_RANK,
                durable=True,
                status=user_status,
            ),
            ranked_default_value(option, rank=DEFAULT_RANK),
        )
    else:
        providers = (
            ranked_toml_value(
                option,
                managed_data,
                rank=MANAGED_RANK,
                durable=True,
                status=managed_status,
            ),
            ranked_environment_value(option, os.environ, rank=ENVIRONMENT_RANK),
            ranked_toml_value(
                option,
                toml_data,
                rank=USER_RANK,
                durable=True,
                status=user_status,
            ),
            ranked_default_value(option, rank=DEFAULT_RANK),
        )
    resolved = resolve_ranked(providers, strategy=option.merge_strategy.value)
    if resolved is None:
        fallback = RankedProviderValue(
            DEFAULT_RANK,
            True,
            ProviderStatus("default", None, ProviderHealth.OK),
            Found(option.default),
        )
        resolved = resolve_ranked(
            (*providers[:-1], fallback),
            strategy=option.merge_strategy.value,
        )
    if resolved is None:
        msg = f"fallback provider was unset for {option.key}"
        raise RuntimeError(msg)
    return resolved


def _ranked_source(resolved: ResolvedValue[object]) -> str:
    """Render rank-keyed provenance through provider display metadata.

    Returns:
        The compatibility source label for `resolve_scalar`.
    """
    return " + ".join(resolved.provider_status[rank].name for rank in resolved.ranks)


def _emit_ranked_diagnostics(
    option: ConfigOption, resolved: ResolvedValue[object]
) -> None:
    """Emit provider rejections encountered before the effective value.

    Providers retain rejection reasons without logging so resolution can be
    inspected by diagnostics and startup policy without duplicating warnings.
    This compatibility boundary preserves `resolve_scalar`'s fall-through
    messages for callers that expect the historical logging behavior.

    A scalar that shadows a whole table defaults *every* option beneath it, so
    that rejection is emitted once per process. `config` resolves the full
    manifest in one pass, and logging per option would print the same line
    roughly a hundred times for a single typo.

    Args:
        option: Manifest option being resolved.
        resolved: Rank-keyed provider results and selected value.
    """
    from deepagents_code.configuration.providers import SHADOWED_TABLE_SUFFIX
    from deepagents_code.configuration.resolver import ENVIRONMENT_RANK
    from deepagents_code.configuration.types import Found, Invalid

    accumulating = option.merge_strategy is not MergeStrategy.REPLACE
    for rank in sorted(resolved.tier_health):
        result = resolved.tier_health[rank]
        if isinstance(result, Invalid):
            reasons = resolved.tier_diagnostics.get(rank) or (result.reason,)
            for reason in reasons:
                if SHADOWED_TABLE_SUFFIX in reason:
                    warning_key = ("ranked provider", reason)
                    if warning_key in _warned_non_table_paths:
                        continue
                    _warned_non_table_paths.add(warning_key)
                logger.warning("%s", reason)
            continue
        if isinstance(result, Found):
            for reason in resolved.tier_diagnostics.get(rank, ()):
                logger.warning("%s", reason)
        if not isinstance(result, Found):
            continue
        if rank == ENVIRONMENT_RANK and option.empty_env_is_false:
            source = resolved.provider_status[rank].name
            prefix = "env ("
            if source.startswith(prefix) and source.endswith(")"):
                name = source[len(prefix) : -1]
                raw = os.environ.get(name)
                if raw is not None and not raw.strip():
                    logger.debug(
                        "%s is blank (%r); resolving %s to False",
                        name,
                        raw,
                        option.key,
                    )
        if not accumulating:
            break


def resolve_scalar(
    option: ConfigOption,
    *,
    toml_data: Mapping[str, Any],
    managed_toml_data: Mapping[str, Any] | None = None,
) -> tuple[Any, str]:
    """Resolve an option through ranked providers with a stable return shape.

    The public-ish signature remains unchanged for embedders. Coercion happens
    at provider boundaries, precedence is numeric, and durable providers mask
    lower-priority ephemeral tiers structurally.

    Args:
        option: Manifest option to resolve.
        toml_data: Parsed user `config.toml` mapping.
        managed_toml_data: Parsed managed TOML mapping. The process snapshot is
            used when omitted; pass an empty mapping for an isolated user source.

    Returns:
        `(value, source)` from the ranked engine. Human-readable source labels
            are rendered from rank-keyed `ProviderStatus` metadata.
    """
    resolved = resolve_ranked_scalar(
        option,
        toml_data=toml_data,
        managed_toml_data=managed_toml_data,
    )
    _emit_ranked_diagnostics(option, resolved)
    return resolved.value, _ranked_source(resolved)


def load_bool_display_preference(
    key: str,
    *,
    fallback: bool,
    on_rejected: Callable[[str], None] | None = None,
) -> bool:
    """Resolve a boolean `Display` option through the config manifest.

    Precedence follows `resolve_scalar`: managed config wins, then the option's
    `DEEPAGENTS_CODE_*` env var if it declares one (not all do), then its
    `[ui]` key in `~/.deepagents/config.toml`, then the declared default.
    Resolution is intentionally forgiving — an unreadable config, a non-table
    `[ui]`, or a wrong-typed value is logged and falls through, so a typo in a
    cosmetic setting never breaks startup.

    Warnings from this path only reach the TUI Debug Console or a
    `DEEPAGENTS_CODE_DEBUG` log file, so by default a rejected value has no
    reader; `dcode config get <key>` shows one as source `default`. Pass
    `on_rejected` where that silence is itself the bug — see `on_rejected`
    below.

    Args:
        key: Manifest key of the option, e.g. `"display.cursor_blink"`.
        fallback: Value to use when `key` is not in the manifest at all, or is
            not a `BOOL` option — both mean a caller and the manifest disagree.
            Pinned against each option's declared default by
            `test_bool_display_preference_fallbacks_match_the_manifest`.
        on_rejected: Called once per rejection reason when a tier at or above
            the winning tier supplied a value that failed to parse — a
            rejection a weaker tier could not have influenced, and one at a
            stronger tier the stronger tier's own value already settled. Only
            worth passing for an option whose
            purpose is to *suppress* something: falling through to the default
            then produces the exact output the user asked to hide, which is
            indistinguishable from the option never having been set. Callers
            that fail toward a merely cosmetic default should omit it — this
            runs during TUI startup for most options, where writing to a
            stream would land on top of the interface.

    Returns:
        The resolved value.
    """
    from deepagents_code.configuration.types import Invalid

    option = get_option(key)
    if option is None:
        logger.warning("Unknown config option %r; falling back to %r", key, fallback)
        return fallback
    if option.kind is not OptionKind.BOOL:
        # Without this, a mistyped key naming a STR option would return
        # `bool("auto")` (`display.charset`) — silently `True` forever, with no
        # warning at all.
        logger.warning(
            "Config option %r is %s, not a bool; falling back to %r",
            key,
            option.kind.value,
            fallback,
        )
        return fallback
    # Inlines `resolve_scalar` rather than calling it, because the rejection
    # reasons `on_rejected` needs live on the ranked result that it discards.
    ranked = resolve_ranked_scalar(option, toml_data=load_config_toml())
    _emit_ranked_diagnostics(option, ranked)
    resolved = bool(ranked.value)
    # Keep the source: a display option turned off by managed policy is
    # otherwise indistinguishable from one the user turned off themselves.
    # Visible under `DEEPAGENTS_CODE_DEBUG`; `dcode config get` is the answer
    # without it.
    logger.debug("Resolved %s to %r from %s", key, resolved, _ranked_source(ranked))
    if on_rejected is not None:
        # The env tier is the only non-durable one, so a stronger durable tier
        # cannot mask it; every tier at or below the winner genuinely lost.
        winner_rank = ranked.ranks[0]
        for rank in sorted(ranked.tier_health):
            if rank > winner_rank:
                break
            result = ranked.tier_health[rank]
            if isinstance(result, Invalid):
                for reason in ranked.tier_diagnostics.get(rank) or (result.reason,):
                    on_rejected(reason)
    return resolved


def resolve_interpreter_kwargs(
    *,
    toml_data: Mapping[str, Any] | None = None,
    managed_toml_data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the `[interpreter]` options into `Settings` constructor kwargs.

    Only the interpreter group is resolved through the manifest. Credentials,
    the shell allow-list, and the LangSmith project keep their dedicated
    loaders in `config.py` (their empty-string-to-`None` and reload semantics
    do not fit the generic resolver), so this stays scoped to the section whose
    defaults this module owns.

    Args:
        toml_data: Parsed `config.toml`; loaded automatically when omitted.
        managed_toml_data: Parsed managed TOML; the process snapshot is used when
            omitted.

    Returns:
        Mapping of `Settings` field name to resolved value for the interpreter
        section, suitable for splatting into `Settings(...)`.
    """
    data = load_config_toml() if toml_data is None else toml_data
    resolved: dict[str, Any] = {}
    for option in get_config_options():
        if option.group != "Interpreter" or option.settings_field is None:
            continue
        value, _ = resolve_scalar(
            option,
            toml_data=data,
            managed_toml_data=managed_toml_data,
        )
        resolved[option.settings_field] = value
    return resolved


def _is_valid_auto_classifier_timeout(value: object) -> bool:
    """Return whether `value` is an accepted Auto classifier timeout.

    Expects the `float` that `OptionKind.FLOAT` resolution produces on every
    layer (`_coerce_env` and `_coerce_toml` both widen to `float`, and the typed
    default is a `float`); a bare `int` is rejected.
    """
    return (
        isinstance(value, float)
        and math.isfinite(value)
        and AUTO_CLASSIFIER_TIMEOUT_FLOOR <= value <= AUTO_CLASSIFIER_TIMEOUT_CEILING
    )


def resolve_auto_classifier_timeout_with_source(
    *,
    toml_data: Mapping[str, Any] | None = None,
    managed_toml_data: Mapping[str, Any] | None = None,
) -> tuple[float, str]:
    """Resolve the Auto classifier decision-batch budget and its source.

    Args:
        toml_data: Parsed `config.toml`; loaded automatically when omitted.
        managed_toml_data: Parsed managed TOML; the process snapshot is used when
            omitted.

    Returns:
        `(timeout_seconds, source)`, where `source` is the layer that supplied
            the effective value (`env (<name>)`, `config.toml`, or `default`).
            The timeout is guaranteed within
            `[AUTO_CLASSIFIER_TIMEOUT_FLOOR, AUTO_CLASSIFIER_TIMEOUT_CEILING]`;
            an out-of-range layer is discarded in favor of the next one, so the
            returned source never credits a rejected layer.
    """
    data = load_config_toml() if toml_data is None else toml_data
    option = get_option("models.auto_classifier_timeout")
    if option is None:
        return AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT, "default"

    managed_data = (
        load_managed_config_toml() if managed_toml_data is None else managed_toml_data
    )
    value, source = resolve_scalar(
        option,
        toml_data=data,
        managed_toml_data=managed_data,
    )
    if _is_valid_auto_classifier_timeout(value):
        return value, source

    from deepagents_code.configuration.service import managed_decided

    if managed_decided(source):
        logger.warning(
            "Ignoring managed auto_classifier_timeout %r (expected seconds in "
            "[%g, %g]); falling through to the next config source",
            value,
            AUTO_CLASSIFIER_TIMEOUT_FLOOR,
            AUTO_CLASSIFIER_TIMEOUT_CEILING,
        )
        return resolve_auto_classifier_timeout_with_source(
            toml_data=data,
            managed_toml_data={},
        )

    # Invalid higher-precedence values must fall through instead of jumping
    # straight to the default. Hide the rejected env var and re-resolve so
    # `config.toml` and then the typed default still apply, mirroring
    # `resolve_recursion_limit`.
    if source.startswith("env (") and source.endswith(")"):
        env_name = source[len("env (") : -1]
        logger.warning(
            "Ignoring %s auto_classifier_timeout %r (expected seconds in "
            "[%g, %g]); falling through to the next config source",
            source,
            value,
            AUTO_CLASSIFIER_TIMEOUT_FLOOR,
            AUTO_CLASSIFIER_TIMEOUT_CEILING,
        )
        previous = os.environ.pop(env_name, None)
        if previous is None:
            # The name reconstructed from `source` is not the key `resolve_scalar`
            # read, so re-resolving would see the same rejected value and recurse
            # forever. Unreachable today (the source label is built from the
            # environ key), but this runs on the startup path where a
            # `RecursionError` would surface only as an opaque launch failure.
            logger.warning(
                "Unexpected auto_classifier_timeout env source %r; using %g",
                source,
                AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT,
            )
            return AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT, "default"
        try:
            return resolve_auto_classifier_timeout_with_source(
                toml_data=data,
                managed_toml_data=managed_data,
            )
        finally:
            os.environ[env_name] = previous

    if source != "default":
        logger.warning(
            "Ignoring %s auto_classifier_timeout %r (expected seconds in "
            "[%g, %g]); using %g",
            source,
            value,
            AUTO_CLASSIFIER_TIMEOUT_FLOOR,
            AUTO_CLASSIFIER_TIMEOUT_CEILING,
            AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT,
        )
    return AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT, "default"


def resolve_auto_classifier_timeout(
    *,
    toml_data: Mapping[str, Any] | None = None,
    managed_toml_data: Mapping[str, Any] | None = None,
) -> float:
    """Resolve the wall-clock budget for one Auto classifier decision batch.

    Resolves `models.auto_classifier_timeout` through the standard managed →
    env → `config.toml` → default precedence. An out-of-range value (below
    `AUTO_CLASSIFIER_TIMEOUT_FLOOR` or above `AUTO_CLASSIFIER_TIMEOUT_CEILING`)
    is discarded with a logged warning and the next lower-precedence layer is
    tried, so a bad higher-precedence override cannot mask a valid TOML setting
    (or the default) and can never remove the deadline that keeps a stalled
    classifier from hanging every gated tool call.

    Args:
        toml_data: Parsed `config.toml`; loaded automatically when omitted.
        managed_toml_data: Parsed managed TOML; the process snapshot is used when
            omitted.

    Returns:
        The resolved timeout in seconds, guaranteed within
            `[AUTO_CLASSIFIER_TIMEOUT_FLOOR, AUTO_CLASSIFIER_TIMEOUT_CEILING]`.
    """
    value, _ = resolve_auto_classifier_timeout_with_source(
        toml_data=toml_data,
        managed_toml_data=managed_toml_data,
    )
    return value


def blank_auto_classifier_env_name() -> str | None:
    """Return the env var blanking the Auto classifier model, if any.

    A present-but-blank `DEEPAGENTS_CODE_AUTO_CLASSIFIER_MODEL` means "inherit
    the main agent model" and deliberately outranks `config.toml`, unlike every
    other option, where `resolve_scalar` treats a blank env value as unset. That
    veto is why this cannot go through `resolve_scalar`.

    Returns:
        The name of the blank env var, or `None` when no env var is set or the
            highest-precedence one carries a usable value.
    """
    option = get_option("models.auto_classifier")
    if option is None:
        return None
    from deepagents_code.model_config import resolved_env_var_name

    names: list[str] = []
    if option.env_var:
        names.append(resolved_env_var_name(option.env_var))
    names.extend(option.fallback_env_vars)
    for name in names:
        raw = os.environ.get(name)
        if raw is None:
            continue
        # The first *set* name decides; a blank fallback must not veto a usable
        # primary.
        return name if not raw.strip() else None
    return None


def resolve_auto_classifier_model_with_source(
    *,
    toml_data: Mapping[str, Any] | None = None,
    managed_toml_data: Mapping[str, Any] | None = None,
) -> tuple[str | None, str]:
    """Resolve the effective Auto classifier model spec and its source.

    Shares the blank-env veto with
    `config.resolve_auto_classifier_model_with_problem` so `dcode config` cannot
    report a classifier the runtime does not use. A managed value outranks that
    veto. `None` means the classifier inherits the main agent model; a blank
    managed value means inherit, credited to `managed config`.

    Args:
        toml_data: Parsed `config.toml`; loaded automatically when omitted.
        managed_toml_data: Parsed managed TOML; the process snapshot is used when
            omitted.

    Returns:
        `(spec, source)`. `source` credits the layer that decided the outcome,
            including the env var whose blank value forced the inherit, so the
            source never points at a value the runtime ignored.
    """
    data = load_config_toml() if toml_data is None else toml_data
    option = get_option("models.auto_classifier")
    if option is None:
        return None, "default"

    managed_data = (
        load_managed_config_toml() if managed_toml_data is None else managed_toml_data
    )
    value, source = resolve_scalar(
        option,
        toml_data=data,
        managed_toml_data=managed_data,
    )
    from deepagents_code.configuration.service import managed_decided

    if managed_decided(source):
        return (
            value.strip() if isinstance(value, str) and value.strip() else None
        ), source

    blank_env = blank_auto_classifier_env_name()
    if blank_env is not None:
        return None, f"env ({blank_env})"

    value, source = resolve_scalar(
        option,
        toml_data=data,
        managed_toml_data={},
    )
    if isinstance(value, str) and value.strip():
        return value.strip(), source
    # A blank or wrong-typed `config.toml` entry reverts to the main agent model;
    # keep the source so the surface still shows where the ignored value lives.
    return None, source


def option_accepts_toml(
    option: ConfigOption, value: object, *, source: str = "config.toml"
) -> bool:
    """Return whether `value` has the type `option` declares for TOML.

    The public form of the coercion check, so other modules can validate a
    value without reaching for `_coerce_toml` and the `_INVALID` sentinel.

    Args:
        option: The manifest option that owns the path.
        value: The raw TOML value.
        source: Source label used in the mismatch log.

    Returns:
        Whether the value would survive coercion.
    """
    return _coerce_toml(option, value, source=source) is not _INVALID


def is_valid_recursion_limit(value: object) -> bool:
    """Return whether `value` is an accepted main-agent `recursion_limit`."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and RECURSION_LIMIT_FLOOR <= value <= RECURSION_LIMIT_CEILING
    )


def resolve_recursion_limit(
    *,
    toml_data: Mapping[str, Any] | None = None,
    managed_toml_data: Mapping[str, Any] | None = None,
) -> int:
    """Resolve the effective main-agent `recursion_limit`.

    Resolves `runtime.recursion_limit` through the standard managed → env →
    `config.toml` → default precedence. An out-of-range value (below
    `RECURSION_LIMIT_FLOOR` or above `RECURSION_LIMIT_CEILING`) is discarded
    with a logged warning and the next lower-precedence layer is tried, so a bad
    higher-precedence override cannot mask a valid TOML setting (or the
    default).

    An out-of-range *managed* value falls through here, but stops an agent
    launch: `main._apply_managed_runtime_policy` treats it as unenforceable
    policy and exits 78, because the CLI flag it would otherwise leave in force
    outranks this bounded resolver.

    Args:
        toml_data: Parsed `config.toml`; loaded automatically when omitted.
        managed_toml_data: Parsed managed TOML; the process snapshot is used when
            omitted.

    Returns:
        The resolved recursion limit, guaranteed within
            `[RECURSION_LIMIT_FLOOR, RECURSION_LIMIT_CEILING]`.
    """
    data = load_config_toml() if toml_data is None else toml_data
    option = get_option("runtime.recursion_limit")
    if option is None:
        return RECURSION_LIMIT_DEFAULT

    managed_data = (
        load_managed_config_toml() if managed_toml_data is None else managed_toml_data
    )
    value, source = resolve_scalar(
        option,
        toml_data=data,
        managed_toml_data=managed_data,
    )
    if is_valid_recursion_limit(value):
        return value

    from deepagents_code.configuration.service import managed_decided

    if managed_decided(source):
        logger.warning(
            "Ignoring managed recursion_limit %r (expected int in [%d, %d]); "
            "falling through to the next config source",
            value,
            RECURSION_LIMIT_FLOOR,
            RECURSION_LIMIT_CEILING,
        )
        return resolve_recursion_limit(
            toml_data=data,
            managed_toml_data={},
        )

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
            return resolve_recursion_limit(
                toml_data=data,
                managed_toml_data=managed_data,
            )
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
"""Provider integration import modules and the extras that install them.

Every import module name here must equal its PyPI distribution name up to
underscore-for-hyphen substitution -- `provider_package_name` derives the
distribution from it to build a `pypi.org/project/...` link, and a provider
whose two names diverge would render a link to a nonexistent project.
"""


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


def provider_package_name(provider: str) -> str | None:
    """Return the PyPI distribution that provides `provider`, if known.

    Derived from the provider's integration import module by replacing
    underscores with hyphens (e.g. `langchain_google_genai` ->
    `langchain-google-genai`), which matches the distribution name for every
    curated entry -- see the `_PROVIDER_DEPENDENCIES` docstring. This is not
    PEP 503 normalization, and the result is not validated against PyPI.

    Args:
        provider: Provider name (e.g. `"baseten"`, `"google_genai"`).

    Returns:
        The distribution name, or `None` when the provider has no curated
            extra (custom `class_path` providers, ambient-auth providers, etc.).
    """
    dependency = _PROVIDER_DEPENDENCIES.get(provider)
    return dependency[0].replace("_", "-") if dependency else None


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
        key="display.cursor_blink",
        group="Display",
        summary=(
            "Blink the chat input cursor (tmux needs 'focus-events on' to hide "
            "it in unfocused panes)."
        ),
        kind=OptionKind.BOOL,
        default=True,
        env_var=_env_vars.CURSOR_BLINK,
        toml_keys=("ui", "cursor_blink"),
        empty_env_is_false=True,
    ),
    ConfigOption(
        key="display.terminal_progress",
        group="Display",
        summary="Report agent activity as terminal taskbar/dock/tab progress.",
        kind=OptionKind.BOOL,
        default=True,
        env_var=_env_vars.TERMINAL_PROGRESS,
        toml_keys=("ui", "terminal_progress"),
        empty_env_is_false=True,
    ),
    ConfigOption(
        key="display.show_message_timestamps",
        group="Display",
        summary="Show the timestamp footer under each chat message.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.SHOW_MESSAGE_TIMESTAMPS,
        toml_keys=("ui", "show_message_timestamps"),
    ),
    ConfigOption(
        key="display.show_usage_stats",
        group="Display",
        summary="Show session usage statistics when a session ends.",
        kind=OptionKind.BOOL,
        default=True,
        env_var=_env_vars.SHOW_USAGE_STATS,
        toml_keys=("ui", "show_usage_stats"),
        empty_env_is_false=True,
    ),
    ConfigOption(
        key="display.themes",
        group="Display",
        summary=(
            "User-defined themes and built-in theme overrides, keyed by theme name."
        ),
        kind=OptionKind.STRUCTURED,
        toml_keys=("themes",),
        merge_strategy=MergeStrategy.DEEP_MERGE,
    ),
    ConfigOption(
        key="display.terminal_themes",
        group="Display",
        summary="Per-`TERM_PROGRAM` default theme, written by the theme picker.",
        kind=OptionKind.STRUCTURED,
        toml_keys=("ui", "terminal_themes"),
        merge_strategy=MergeStrategy.DEEP_MERGE,
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
        key="display.show_diff_line_numbers",
        group="Display",
        summary="Show file line numbers in diff hunks.",
        kind=OptionKind.BOOL,
        default=True,
        toml_keys=("ui", "show_diff_line_numbers"),
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
    ConfigOption(
        key="models.auto_classifier",
        group="Models",
        summary=(
            "Model spec ('provider:model') used by the Auto approval classifier; "
            "unset reuses the main agent model. A weaker model weakens Auto's "
            "review of gated actions."
        ),
        kind=OptionKind.STR,
        env_var=_env_vars.AUTO_CLASSIFIER_MODEL,
        toml_keys=("models", "auto_classifier"),
        cli_flag="--auto-classifier-model",
    ),
    ConfigOption(
        key="models.auto_classifier_timeout",
        group="Models",
        summary=(
            "Seconds the Auto approval classifier may take to review one batch "
            "(1-300); a batch that misses the deadline is denied."
        ),
        kind=OptionKind.FLOAT,
        default=AUTO_CLASSIFIER_TIMEOUT_SECONDS_DEFAULT,
        env_var=_env_vars.AUTO_CLASSIFIER_TIMEOUT,
        toml_keys=("models", "auto_classifier_timeout"),
    ),
    ConfigOption(
        key="models.providers",
        group="Models",
        summary=(
            "Custom chat-model providers. A `class_path` entry is imported at "
            "model creation, so its module runs with your privileges."
        ),
        kind=OptionKind.STRUCTURED,
        toml_keys=("models", "providers"),
        merge_strategy=MergeStrategy.DEEP_MERGE,
        # A provider table's `params` are forwarded verbatim to the constructor
        # and can carry credentials, so the value is never printed — `config`
        # reports source and presence only.
        redacted=True,
    ),
    ConfigOption(
        key="retries.max_retries",
        group="Models",
        summary=(
            "Default provider retry count; override per provider with "
            "`[retries.<provider>]`."
        ),
        kind=OptionKind.NON_NEGATIVE_INT,
        toml_keys=("retries", "max_retries"),
    ),
    # --- Agents ---------------------------------------------------------
    ConfigOption(
        key="agents.default",
        group="Agents",
        summary=(
            "Sticky default agent; used when neither `--agent` nor `-r <thread>` "
            "is given, and ignored if it names an agent that no longer exists."
        ),
        kind=OptionKind.NON_EMPTY_STR,
        toml_keys=("agents", "default"),
    ),
    ConfigOption(
        key="agents.recent",
        group="Agents",
        summary=(
            "Last switched-to agent, written by the app; the fallback behind "
            "`agents.default`."
        ),
        kind=OptionKind.NON_EMPTY_STR,
        toml_keys=("agents", "recent"),
    ),
    ConfigOption(
        key="agents.async_subagents",
        group="Agents",
        summary="Remote LangGraph deployments exposed to the agent as subagents.",
        kind=OptionKind.STRUCTURED,
        toml_keys=("async_subagents",),
        merge_strategy=MergeStrategy.DEEP_MERGE,
        # A subagent `headers` table can carry `Authorization` tokens, so the
        # value is never printed — `config` reports source and presence only.
        redacted=True,
    ),
    # --- Sandboxes ------------------------------------------------------
    ConfigOption(
        key="sandboxes.default",
        group="Sandboxes",
        summary=(
            "Sandbox backend used by a bare `--sandbox` (passed with no value); "
            "omitting the flag entirely runs unsandboxed."
        ),
        kind=OptionKind.STR,
        toml_keys=("sandboxes", "default"),
    ),
    ConfigOption(
        key="sandboxes.providers",
        group="Sandboxes",
        summary=(
            "Custom sandbox backends. A `class_path` entry is imported at "
            "sandbox creation, so its module runs with your privileges."
        ),
        kind=OptionKind.STRUCTURED,
        toml_keys=("sandboxes", "providers"),
        merge_strategy=MergeStrategy.DEEP_MERGE,
        # A provider table's `params` are forwarded verbatim to the constructor
        # and can carry credentials, so the value is never printed — `config`
        # reports source and presence only.
        redacted=True,
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
            "Shell commands allowed without approval (comma-separated string "
            "or TOML array, or 'recommended'/'all')."
        ),
        kind=OptionKind.SHELL_LIST_DELEGATE,
        env_var=_env_vars.SHELL_ALLOW_LIST,
        toml_keys=("shell", "allow_list"),
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
        key="features.resume_term_program",
        group="Tools",
        summary=(
            "Include launch-time TERM_PROGRAM in resume hints; defaults on in "
            "experimental or debug mode."
        ),
        kind=OptionKind.BOOL_MODE_DEFAULT,
        env_var=_env_vars.RESUME_TERM_PROGRAM,
        empty_env_is_false=True,
        toml_keys=("features", "resume_term_program"),
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
    # --- Goals ----------------------------------------------------------
    ConfigOption(
        key="goals.auto_accept_criteria",
        group="Goals",
        summary="Apply generated goal criteria automatically in Auto mode.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.GOAL_AUTO_ACCEPT_CRITERIA,
        toml_keys=("goals", "auto_accept_criteria"),
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
        key="threads.compact_on_resume_threshold",
        group="Threads",
        summary=(
            "Offer to compact a resumed thread above this context size (0 disables)."
        ),
        kind=OptionKind.INT,
        default=COMPACT_ON_RESUME_THRESHOLD_DEFAULT,
        toml_keys=("threads", "compact_on_resume_threshold"),
    ),
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
        merge_strategy=MergeStrategy.DEEP_MERGE,
    ),
    # --- Warnings ------------------------------------------------------
    ConfigOption(
        key="warnings.cold_cache_min_delta_usd",
        group="Warnings",
        summary=(
            "Warn before a cold prompt-cache turn whose estimated extra cost "
            "reaches this amount (0 disables)."
        ),
        kind=OptionKind.FLOAT,
        default=COLD_CACHE_WARNING_THRESHOLD_USD_DEFAULT,
        toml_keys=("warnings", "cold_cache_min_delta_usd"),
    ),
    ConfigOption(
        key="warnings.trusted_cache_endpoints",
        group="Warnings",
        summary=(
            "Alternate endpoint hosts assumed to forward cache settings and "
            "honor provider cache retention "
            '(e.g. ["smith.langchain.com"]). Matched per exact host, so list '
            "each subdomain you actually connect to; trusting a host trusts it "
            "on every port. Bare hostnames or http(s) URLs; entries carrying a "
            "user:password prefix or a non-default port are rejected."
        ),
        kind=OptionKind.STRUCTURED,
        toml_keys=("warnings", "trusted_cache_endpoints"),
    ),
    ConfigOption(
        key="warnings.session_cost_threshold_usd",
        group="Warnings",
        summary=(
            "Warn once when estimated thread cost exceeds this USD amount (0 disables)."
        ),
        kind=OptionKind.FLOAT,
        default=SESSION_COST_WARNING_THRESHOLD_USD_DEFAULT,
        toml_keys=("warnings", "session_cost_threshold_usd"),
    ),
    ConfigOption(
        key="warnings.suppress",
        group="Warnings",
        summary=(
            "Warning keys to suppress (e.g. 'ripgrep', 'tavily', 'yolo'); "
            "also editable from /notifications."
        ),
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
        # The trust reader normalizes heterogeneous env-name and scoped TOML
        # grants into distinct mapping leaves. Deep merge keeps both when
        # managed policy is absent; an explicit managed value supplies both
        # leaves and therefore replaces both lower grant forms.
        merge_strategy=MergeStrategy.DEEP_MERGE,
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
        merge_strategy=MergeStrategy.UNION,
    ),
    # Read by `mcp_disabled` (the server viewer's disable toggle) rather than by
    # `load_mcp_server_trust_lists`, but it is security-load-bearing all the
    # same: it is in `UNION_PATHS`, so the managed and user lists accumulate,
    # and `mcp_disabled._managed_disabled_servers` fails closed on a managed
    # value it cannot read, which disables every MCP server.
    ConfigOption(
        key="mcp.disabled_servers",
        group="MCP",
        summary=(
            "MCP server names denied by the server viewer or by managed policy; "
            "the managed and user lists union."
        ),
        kind=OptionKind.STRUCTURED,
        toml_keys=("mcp", "disabled_servers"),
        merge_strategy=MergeStrategy.UNION,
    ),
    # --- Plugins --------------------------------------------------------
    ConfigOption(
        key="plugins.auto_update",
        group="Plugins",
        summary="Update opted-in plugins after the first prompt; disable globally.",
        kind=OptionKind.BOOL,
        default=True,
        env_var=_env_vars.PLUGIN_AUTO_UPDATE,
        toml_keys=("plugins", "auto_update"),
        empty_env_is_false=True,
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
    ConfigOption(
        key="update.prices_auto_update",
        group="Updates",
        summary=(
            "Refresh the model pricing catalog from upstream hourly in the background."
        ),
        kind=OptionKind.BOOL,
        default=True,
        env_var=_env_vars.PRICES_AUTO_UPDATE,
        toml_keys=("update", "prices_auto_update"),
        empty_env_is_false=True,
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
        key="startup.onboarding",
        group="Startup",
        summary=(
            "Force the first-run onboarding flow to open on every interactive "
            "startup, or disable it entirely; unset follows the completion marker."
        ),
        kind=OptionKind.BOOL,
        env_var=_env_vars.ONBOARDING,
        empty_env_is_false=True,
    ),
    ConfigOption(
        key="startup.mode",
        group="Startup",
        summary="Default approval mode at launch ('manual', 'auto', or 'yolo').",
        kind=OptionKind.STARTUP_MODE_DELEGATE,
        default="manual",
        toml_keys=("startup", "mode"),
        cli_flag="--auto-approve",
    ),
    ConfigOption(
        key="startup.yolo_switcher",
        group="Startup",
        summary=(
            "Include YOLO in the Shift+Tab approval-mode cycle "
            "(Manual → Auto → YOLO); disable to keep the cycle Manual/Auto only."
        ),
        kind=OptionKind.BOOL,
        default=True,
        env_var=_env_vars.YOLO_SWITCHER,
        empty_env_is_false=True,
        toml_keys=("startup", "yolo_switcher"),
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
        key="debug.dep_floor",
        group="Debug",
        summary="Synthesize the stale editable-dependency prompt/warning at launch.",
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.DEBUG_DEP_FLOOR,
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
        key="debug.cold_cache",
        group="Debug",
        summary=(
            "Force the cold prompt-cache warning modal on every interactive "
            "send, overriding suppression."
        ),
        kind=OptionKind.BOOL,
        default=False,
        env_var=_env_vars.DEBUG_COLD_CACHE,
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
        # Set by the self-update restart to carry the launched command name into
        # the re-exec'd process; never user-configured.
        _env_vars.INVOKED_AS,
        # Launch-time snapshot of `TERM_PROGRAM` recorded by `cli_main` so the
        # resume hint can distinguish an explicit launch value from a `.env`
        # file that sets `TERM_PROGRAM` after launch; never user-configured.
        _env_vars.LAUNCH_TERM_PROGRAM,
    }
)
"""`_env_vars` constants intentionally excluded from the option catalog."""


@lru_cache(maxsize=1)
def get_config_options() -> tuple[ConfigOption, ...]:
    """Return every option, credentials-first then by domain group.

    Cached: provider credentials are generated once from `PROVIDER_API_KEY_ENV`
    on first call (which lazily imports `model_config`). The cache assumes that
    registry is an immutable module constant; a test that monkeypatches it must
    call `get_config_options.cache_clear()` (and `_options_by_key.cache_clear()`,
    `_options_by_toml_path.cache_clear()`, and
    `configuration.service._managed_table_paths.cache_clear()`).
    """
    return _credential_options() + _STATIC_OPTIONS


def get_option(key: str) -> ConfigOption | None:
    """Return the manifest entry for `key`, or `None` when unknown."""
    return _options_by_key().get(key)


def option_keys() -> tuple[str, ...]:
    """Return every manifest key in definition order."""
    return tuple(opt.key for opt in get_config_options())


def options_with_key_prefix(prefix: str) -> tuple[ConfigOption, ...]:
    """Return every option whose key sits under the dotted `prefix` section.

    Matching is exact on segment boundaries: `credentials` matches
    `credentials.openai` but `credential` matches nothing, so `config get` can
    accept a section name without also accepting truncated guesses.

    Matching is case-insensitive, and key prefixes are the only section
    namespace: display group titles (`Credentials`, `Tools`) are not accepted,
    since several headings (`Models`, `Tools`) name a different set of options
    than the same word as a prefix — one namespace keeps a section unambiguous.

    Args:
        prefix: Dotted key prefix (e.g. `credentials`). A trailing dot is not
            stripped here — `credentials.` matches nothing, since it would look
            for keys under `credentials..`. Callers that accept user input
            should strip it first.

    Returns:
        Matching options in manifest order; empty when no key uses `prefix`.
    """
    if not prefix:
        return ()
    section = f"{prefix.casefold()}."
    return tuple(
        opt for opt in get_config_options() if opt.key.casefold().startswith(section)
    )


@lru_cache(maxsize=1)
def _options_by_key() -> dict[str, ConfigOption]:
    return {opt.key: opt for opt in get_config_options()}


@lru_cache(maxsize=1)
def _options_by_toml_path() -> dict[tuple[str, ...], ConfigOption]:
    return {opt.toml_keys: opt for opt in get_config_options() if opt.toml_keys}


def option_for_toml_path(path: tuple[str, ...]) -> ConfigOption | None:
    """Return the manifest entry that owns a TOML path, or `None` when unknown.

    Indexed rather than scanned: the merge validates every managed leaf, and a
    linear pass over the whole manifest per leaf runs on the startup path.

    Args:
        path: The dotted TOML path as a key tuple.

    Returns:
        The option that declares `path`, or `None`.
    """
    return _options_by_toml_path().get(path)


def iter_groups(options: Iterable[ConfigOption]) -> list[str]:
    """Return group names from `options` in first-seen order."""
    groups: list[str] = []
    for opt in options:
        if opt.group not in groups:
            groups.append(opt.group)
    return groups
