"""Synchronous providers and provider-domain option coercion."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, assert_never, cast

from deepagents_code._env_vars import classify_env_bool

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from deepagents_code.config_manifest import ConfigOption
    from deepagents_code.configuration.resolver import RankedProviderValue

from deepagents_code.configuration.types import (
    Found,
    Invalid,
    ProviderHealth,
    ProviderResult,
    ProviderStatus,
    TomlSnapshot,
    Unset,
)

SHADOWED_TABLE_SUFFIX = "— every option under it falls back to its next source"
"""Tail of the rejection raised when a scalar shadows a whole TOML table.

`config_manifest._emit_ranked_diagnostics` matches this text to deduplicate the
warning across a full-manifest pass. Both sides must share one constant: a
reworded message that no longer matches would silently restore roughly one
duplicated line per option for a single typo.
"""


def coerce_environment_value(
    option: ConfigOption, raw: str, name: str
) -> ProviderResult[object]:
    """Coerce one present environment value within the env provider domain.

    The returned reason preserves the established diagnostic text. Resolution
    decides when to emit it so health inspection does not log a rejection as a
    side effect of merely reading provider state.

    Args:
        option: Manifest declaration that defines the output type.
        raw: Present environment string.
        name: Environment variable spelling that supplied `raw`.

    Returns:
        `Found` with the typed value or `Invalid` with the rejection reason.
    """
    from deepagents_code.config_manifest import VALID_CURSOR_STYLES, OptionKind

    kind = option.kind
    if kind in {OptionKind.BOOL, OptionKind.BOOL_MODE_DEFAULT}:
        classified = classify_env_bool(raw)
        if classified is None:
            return Invalid(f"Ignoring {name}={raw!r} (expected bool)")
        return Found(classified)
    if kind is OptionKind.BOOL_PRESENCE:
        return Found(bool(raw))
    if kind is OptionKind.STR:
        return Found(raw)
    if kind is OptionKind.NON_EMPTY_STR:
        value = raw.strip()
        if value:
            return Found(value)
        return Invalid(f"Ignoring {name}={raw!r} (expected non-empty string)")
    if kind is OptionKind.LOG_LEVEL_DELEGATE:
        from deepagents_code._debug import LOG_LEVELS

        level = raw.strip().upper()
        if level in LOG_LEVELS:
            return Found(level)
        valid = ", ".join(LOG_LEVELS)
        return Invalid(f"Ignoring {name}={raw!r} (expected one of {valid})")
    if kind is OptionKind.INT:
        try:
            return Found(int(raw.strip()))
        except ValueError:
            return Invalid(f"Ignoring {name}={raw!r} (expected int)")
    if kind is OptionKind.NON_NEGATIVE_INT:
        try:
            value = int(raw.strip())
        except ValueError:
            return Invalid(f"Ignoring {name}={raw!r} (expected int >= 0)")
        if value >= 0:
            return Found(value)
        return Invalid(f"Ignoring {name}={raw!r} (expected int >= 0)")
    if kind is OptionKind.FLOAT:
        try:
            return Found(float(raw.strip()))
        except ValueError:
            return Invalid(f"Ignoring {name}={raw!r} (expected number)")
    if kind is OptionKind.SHELL_LIST_DELEGATE:
        from deepagents_code.config import parse_shell_allow_list

        try:
            return Found(parse_shell_allow_list(raw))
        except ValueError:
            return Invalid(f"Ignoring invalid {name}")
    if kind is OptionKind.SKILLS_DIRS_DELEGATE:
        from deepagents_code.config import _parse_extra_skills_dirs

        try:
            return Found(_parse_extra_skills_dirs(raw, None))
        except (ValueError, RuntimeError):
            return Invalid(f"Ignoring {name} (could not resolve a path)")
    if kind is OptionKind.THEME_DELEGATE:
        # Theme names are resolved by the theme-aware provider path. Keep this
        # defensive passthrough for the compatibility wrapper.
        return Found(raw)
    if kind is OptionKind.CURSOR_STYLE_DELEGATE:
        if raw in VALID_CURSOR_STYLES:
            return Found(raw)
        return Invalid(f"Ignoring {name}={raw!r} (expected 'block' or 'underline')")
    if kind is OptionKind.PTC_DELEGATE or kind is OptionKind.STRUCTURED:
        return Invalid(f"{option.key} is not env-backed; ignoring {name}={raw!r}")
    if kind is OptionKind.STARTUP_MODE_DELEGATE:
        from deepagents_code.model_config import VALID_STARTUP_MODES

        if raw in VALID_STARTUP_MODES:
            return Found(raw)
        return Invalid(
            f"Ignoring {name}={raw!r} (expected 'manual', 'auto', or 'yolo')"
        )
    assert_never(kind)


def coerce_toml_value(
    option: ConfigOption, raw: object, *, source: str
) -> ProviderResult[object]:
    """Coerce one present TOML value within the file-provider domain.

    Args:
        option: Manifest declaration that defines the output type.
        raw: Parsed TOML value.
        source: Human-readable provider name used in diagnostic text.

    Returns:
        `Found` with the typed value or `Invalid` with the rejection reason.
    """
    from deepagents_code.config_manifest import VALID_CURSOR_STYLES, OptionKind

    kind = option.kind
    label = option.toml_path or option.key

    if kind in {
        OptionKind.BOOL,
        OptionKind.BOOL_MODE_DEFAULT,
        OptionKind.BOOL_PRESENCE,
    }:
        if isinstance(raw, bool):
            value = not raw if option.invert_toml_bool else raw
            return Found(value)
    elif kind is OptionKind.INT:
        if isinstance(raw, int) and not isinstance(raw, bool):
            return Found(raw)
    elif kind is OptionKind.NON_NEGATIVE_INT:
        if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0:
            return Found(raw)
    elif kind is OptionKind.FLOAT:
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            return Found(float(raw))
    elif kind is OptionKind.STR:
        if isinstance(raw, str):
            return Found(raw)
    elif kind is OptionKind.NON_EMPTY_STR:
        if isinstance(raw, str) and (value := raw.strip()):
            return Found(value)
    elif kind is OptionKind.SKILLS_DIRS_DELEGATE:
        if isinstance(raw, list):
            from deepagents_code.config import _parse_extra_skills_dirs

            try:
                return Found(_parse_extra_skills_dirs(None, cast("list[str]", raw)))
            except (ValueError, RuntimeError):
                return Invalid(
                    f"Ignoring {label} in {source} (could not resolve a path)"
                )
    elif kind is OptionKind.PTC_DELEGATE:
        from deepagents_code.config import _parse_interpreter_ptc

        try:
            return Found(_parse_interpreter_ptc(raw))
        except ValueError as exc:
            return Invalid(f"Ignoring {label} in {source}: {exc}")
    elif kind is OptionKind.CURSOR_STYLE_DELEGATE:
        if isinstance(raw, str) and raw in VALID_CURSOR_STYLES:
            return Found(raw)
        return Invalid(
            f"Ignoring {label}={raw!r} in {source} (expected 'block' or 'underline')"
        )
    elif kind is OptionKind.STARTUP_MODE_DELEGATE:
        from deepagents_code.model_config import VALID_STARTUP_MODES

        if isinstance(raw, str) and raw in VALID_STARTUP_MODES:
            return Found(raw)
        return Invalid(
            f"Ignoring {label}={raw!r} in {source} "
            "(expected 'manual', 'auto', or 'yolo')"
        )
    elif kind is OptionKind.STRUCTURED:
        return Found(raw)
    elif kind is OptionKind.SHELL_LIST_DELEGATE:
        from deepagents_code.config import (
            parse_shell_allow_list,
            parse_shell_allow_list_items,
        )

        try:
            if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
                return Found(parse_shell_allow_list_items(cast("list[str]", raw)))
            if isinstance(raw, str):
                return Found(parse_shell_allow_list(raw))
        except ValueError as exc:
            return Invalid(f"Ignoring {label} in {source}: {exc}")

    return Invalid(f"Ignoring {label}={raw!r} in {source} (expected {option.type})")


def ranked_toml_value(
    option: ConfigOption,
    data: Mapping[str, Any],
    *,
    rank: int,
    durable: bool,
    status: ProviderStatus,
) -> RankedProviderValue[object]:
    """Read and coerce one option from a parsed TOML provider.

    Args:
        option: Manifest option to read.
        data: Parsed provider table.
        rank: Numeric precedence rank.
        durable: Whether this tier masks lower-priority ephemeral tiers.
        status: Provider health and display metadata.

    Returns:
        Ranked `Found`, `Unset`, or `Invalid` provider result.
    """
    from deepagents_code.configuration.resolver import RankedProviderValue

    if not status.usable or not option.toml_keys:
        result: ProviderResult[object] = Unset()
    else:
        node: object = data
        result = Unset()
        for index, key in enumerate(option.toml_keys):
            if not isinstance(node, dict):
                path = option.toml_keys[:index]
                result = Invalid(
                    f"Ignoring {status.name} [{'.'.join(path)}]; expected a "
                    f"table, got {type(node).__name__} {SHADOWED_TABLE_SUFFIX}"
                )
                break
            if key not in node:
                break
            node = node[key]
        else:
            result = coerce_toml_value(option, node, source=status.name)
    return RankedProviderValue(rank, durable, status, result)


def ranked_environment_value(
    option: ConfigOption,
    environ: Mapping[str, str],
    *,
    rank: int,
) -> RankedProviderValue[object]:
    """Read and coerce one option from the process-environment domain.

    Args:
        option: Manifest option to read.
        environ: Environment mapping, normally `os.environ`.
        rank: Numeric precedence rank.

    Returns:
        Ranked provider result. Fallback names remain one provider tier.
    """
    from deepagents_code.configuration.resolver import RankedProviderValue

    names: list[str] = []
    if option.env_var:
        canonical = option.env_var
        prefixed = (
            canonical
            if canonical.startswith("DEEPAGENTS_CODE_")
            else f"DEEPAGENTS_CODE_{canonical}"
        )
        names.append(prefixed if prefixed in environ else canonical)
    names.extend(option.fallback_env_vars)

    status = ProviderStatus("environment", None, ProviderHealth.OK)
    last_invalid: Invalid | None = None
    diagnostics: list[str] = []
    for name in names:
        raw = environ.get(name)
        if raw is None:
            continue
        status = replace(status, name=f"env ({name})")
        if not raw.strip():
            if option.empty_env_is_false:
                return RankedProviderValue(rank, False, status, Found(False))
            if raw:
                last_invalid = Invalid(
                    f"Ignoring {name}={raw!r} (whitespace-only; treated as unset)"
                )
                diagnostics.append(last_invalid.reason)
            continue
        result = coerce_environment_value(option, raw, name)
        if isinstance(result, Found):
            return RankedProviderValue(
                rank,
                False,
                status,
                result,
                tuple(diagnostics),
            )
        if isinstance(result, Invalid):
            last_invalid = result
            diagnostics.append(result.reason)
    return RankedProviderValue(
        rank,
        False,
        status,
        last_invalid or Unset(),
        tuple(diagnostics),
    )


def ranked_theme_toml_value(
    data: Mapping[str, Any],
    *,
    rank: int,
    durable: bool,
    status: ProviderStatus,
) -> RankedProviderValue[object]:
    """Resolve one file provider's terminal-aware theme preference.

    The terminal mapping and `[ui].theme` fallback are one provider domain:
    they share a durability boundary and source rank. Their internal ordering
    stays inside this provider while precedence between managed, environment,
    user, and default remains the ranked resolver's responsibility.

    Args:
        data: Parsed TOML provider table.
        rank: Numeric provider rank.
        durable: Whether this file tier masks lower ephemeral tiers.
        status: Provider health and display metadata.

    Returns:
        Ranked theme result with the selected TOML path in its display status.
    """
    from deepagents_code.app import _resolve_terminal_mapping, _resolve_theme_name
    from deepagents_code.configuration.resolver import RankedProviderValue

    if not status.usable:
        return RankedProviderValue(rank, durable, status, Unset())
    ui = data.get("ui")
    if ui is None:
        return RankedProviderValue(rank, durable, status, Unset())
    if not isinstance(ui, dict):
        result: ProviderResult[object] = Invalid(
            f"[ui] in {status.name} should be a table; got "
            f"{type(ui).__name__} while resolving theme"
        )
        return RankedProviderValue(rank, durable, status, result)

    resolved = _resolve_terminal_mapping(ui)
    if resolved is not None:
        import os

        term_program = os.environ.get("TERM_PROGRAM", "").strip()
        selected = replace(
            status,
            name=f"{status.name} [ui.terminal_themes.{term_program}]",
        )
        return RankedProviderValue(rank, durable, selected, Found(resolved))

    saved = ui.get("theme")
    resolved = _resolve_theme_name(saved)
    if resolved is not None:
        selected = replace(status, name=f"{status.name} [ui.theme]")
        return RankedProviderValue(rank, durable, selected, Found(resolved))
    if isinstance(saved, str):
        result = Invalid(f"Unknown theme '{saved}' in {status.name}; ignoring it")
        return RankedProviderValue(rank, durable, status, result)
    return RankedProviderValue(rank, durable, status, Unset())


def ranked_theme_environment_value(
    environ: Mapping[str, str], *, rank: int
) -> RankedProviderValue[object]:
    """Resolve the theme environment provider.

    Args:
        environ: Environment mapping, normally `os.environ`.
        rank: Numeric environment rank.

    Returns:
        Ranked theme result with the concrete variable name in its status.
    """
    from deepagents_code._env_vars import THEME
    from deepagents_code.app import _resolve_theme_name
    from deepagents_code.configuration.resolver import RankedProviderValue

    status = ProviderStatus(f"env ({THEME})", None, ProviderHealth.OK)
    raw = environ.get(THEME)
    if raw is None:
        return RankedProviderValue(rank, False, status, Unset())
    resolved = _resolve_theme_name(raw)
    if resolved is not None:
        return RankedProviderValue(rank, False, status, Found(resolved))
    return RankedProviderValue(
        rank,
        False,
        status,
        Invalid(f"Unknown theme '{raw}' in {THEME}; falling through"),
    )


def ranked_default_value(
    option: ConfigOption, *, rank: int
) -> RankedProviderValue[object]:
    """Produce an option's typed or mode-dependent default provider result.

    Args:
        option: Manifest option whose default should be produced.
        rank: Numeric precedence rank.

    Returns:
        Durable ranked default result.
    """
    from deepagents_code.config_manifest import OptionKind
    from deepagents_code.configuration.resolver import RankedProviderValue

    if option.kind is OptionKind.BOOL_MODE_DEFAULT:
        from deepagents_code._env_vars import DEBUG, EXPERIMENTAL, is_env_truthy

        value: object = is_env_truthy(DEBUG) or is_env_truthy(EXPERIMENTAL)
    elif option.kind is OptionKind.LOG_LEVEL_DELEGATE:
        from deepagents_code._env_vars import DEBUG, is_env_truthy

        value = "DEBUG" if is_env_truthy(DEBUG) else "INFO"
    elif option.kind is OptionKind.THEME_DELEGATE:
        from deepagents_code import theme

        value = theme.DEFAULT_THEME
    elif option.kind is OptionKind.STRUCTURED:
        status = ProviderStatus("default", None, ProviderHealth.OK)
        return RankedProviderValue(rank, True, status, Unset())
    else:
        value = option.default
    status = ProviderStatus("default", None, ProviderHealth.OK)
    return RankedProviderValue(rank, True, status, Found(value))


@dataclass(frozen=True, slots=True)
class TomlFileProvider:
    """Provider that parses one local TOML file per `load` call."""

    name: str
    path: Path

    def load(self) -> TomlSnapshot:
        """Parse the file and classify missing, unreadable, or corrupt states.

        Returns:
            Parsed data and provider health.
        """
        try:
            with self.path.open("rb") as handle:
                data = tomllib.load(handle)
        except FileNotFoundError:
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    self.path,
                    ProviderHealth.MISSING,
                ),
            )
        except OSError as exc:
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    self.path,
                    ProviderHealth.UNREADABLE,
                    type(exc).__name__,
                ),
            )
        except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
            detail = (
                "not UTF-8 encoded" if isinstance(exc, UnicodeDecodeError) else str(exc)
            )
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    self.path,
                    ProviderHealth.CORRUPT,
                    detail,
                ),
            )
        if not isinstance(data, dict):
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    self.path,
                    ProviderHealth.CORRUPT,
                    "top-level TOML value is not a table",
                ),
            )
        return TomlSnapshot(
            data,
            ProviderStatus(self.name, self.path, ProviderHealth.OK),
        )
