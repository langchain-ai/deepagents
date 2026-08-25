"""Synchronous providers and provider-domain option coercion."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, assert_never, cast

from deepagents_code._env_vars import classify_env_bool

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from deepagents_code.config_manifest import ConfigOption
    from deepagents_code.configuration.resolver import RankedProviderValue

from deepagents_code.configuration.resolver import (
    DEFAULT_RANK,
    ENVIRONMENT_RANK,
    USER_RANK,
)
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

UNUSABLE_SOURCE_SUFFIX = "— using defaults for every option it would have set"
"""Tail of the rejection raised when a whole TOML source could not be read.

Deduplicated the same way, and for the same reason: a rejected file affects
every option at once, so the warning belongs to the file, not to each key.
"""

RETAINED_SOURCE_SUFFIX = "— still applying the last readable version of it"
"""Tail of the rejection raised when a failed reload kept the previous values.

Distinct from `UNUSABLE_SOURCE_SUFFIX` because the consequence is different:
nothing fell back to a default, but the file on disk no longer describes what
the process is enforcing, and an edit the user just made is not in effect.
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
    if kind in {
        OptionKind.MODEL_LIST_DELEGATE,
        OptionKind.PTC_DELEGATE,
        OptionKind.STRUCTURED,
    }:
        # No option of these kinds declares `env_var` *or* `fallback_env_vars`,
        # so the env provider never reaches this branch; only the `_coerce_env`
        # compatibility wrapper, which has no env-var guard, can. If a future
        # option gains either name, reject rather than pass the raw string
        # through: an uncoerced value would bypass the delegate parser's
        # validation. Rejection drops to the next-weaker tier (TOML, then the
        # manifest default) -- note that for a policy option such as
        # `models.allowed` the default is "unrestricted", so what keeps that
        # fallback safe is managed config outranking the environment, not the
        # default itself being conservative.
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

    if option.key == "threads.sort_order":
        if isinstance(raw, str) and raw in {"created_at", "updated_at"}:
            return Found(raw)
        return Invalid(
            f"Ignoring {label}={raw!r} in {source} "
            "(expected 'created_at' or 'updated_at')"
        )
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
    elif kind is OptionKind.MODEL_LIST_DELEGATE:
        from deepagents_code.model_config import parse_model_allowlist

        try:
            return Found(parse_model_allowlist(raw))
        except (TypeError, ValueError) as exc:
            return Invalid(f"Ignoring {label} in {source}: {exc}")
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
    from deepagents_code.configuration.resolver import RankedProviderValue
    from deepagents_code.configuration.theme_resolution import (
        resolve_terminal_mapping,
        resolve_theme_name,
    )

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

    resolved = resolve_terminal_mapping(ui)
    if resolved is not None:
        import os

        term_program = os.environ.get("TERM_PROGRAM", "").strip()
        selected = replace(
            status,
            name=f"{status.name} [ui.terminal_themes.{term_program}]",
        )
        return RankedProviderValue(rank, durable, selected, Found(resolved))

    saved = ui.get("theme")
    resolved = resolve_theme_name(saved)
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
    from deepagents_code.configuration.resolver import RankedProviderValue
    from deepagents_code.configuration.theme_resolution import resolve_theme_name

    status = ProviderStatus(f"env ({THEME})", None, ProviderHealth.OK)
    raw = environ.get(THEME)
    if raw is None:
        return RankedProviderValue(rank, False, status, Unset())
    resolved = resolve_theme_name(raw)
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


@dataclass(slots=True)
class _TomlSnapshotState:
    """Mutable snapshot cell owned by a frozen provider."""

    value: TomlSnapshot | None = None
    """Last usable snapshot, or the empty failed snapshot from an initial read."""

    failure: ProviderStatus | None = None
    """Status of the most recent failed reload, kept for health reporting."""


@dataclass(frozen=True, slots=True)
class TomlFileProvider:
    """Ranked provider backed by one local TOML file snapshot."""

    name: str
    path: Path | None
    """File this provider reads, or `None` for a snapshot with no known origin.

    A `None` path is not a filename to guess at. Inventing one would make
    `load` read a relative path against the process working directory, and for
    the managed tier that is a trust boundary, not a cosmetic default.
    """

    rank: int = USER_RANK
    durable: bool = True
    snapshot: TomlSnapshot | None = field(default=None, repr=False, compare=False)
    loader: Callable[[], TomlSnapshot] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _state: _TomlSnapshotState = field(
        default_factory=_TomlSnapshotState,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        """Seed the mutable snapshot cell when a generation is supplied."""
        self._state.value = self.snapshot

    def load(self) -> TomlSnapshot:
        """Parse the file and classify missing, unreadable, or corrupt states.

        Returns:
            Parsed data and provider health. A provider with no path reports
            `INDETERMINATE`, which is not usable: an empty read proves nothing
            about a file whose location is unknown.
        """
        if self.path is None:
            return TomlSnapshot(
                {},
                ProviderStatus(
                    self.name,
                    None,
                    ProviderHealth.INDETERMINATE,
                    "no path is known for this source, so it cannot be re-read",
                ),
            )
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

    def get(self, option: ConfigOption) -> RankedProviderValue[object]:
        """Read one option from the current file snapshot.

        Args:
            option: Manifest option to read.

        Returns:
            Ranked and coerced provider result.
        """
        from deepagents_code.config_manifest import OptionKind

        snapshot = self.current_snapshot()
        if option.kind is OptionKind.THEME_DELEGATE:
            ranked = ranked_theme_toml_value(
                snapshot.data,
                rank=self.rank,
                durable=self.durable,
                status=snapshot.status,
            )
        else:
            ranked = ranked_toml_value(
                option,
                snapshot.data,
                rank=self.rank,
                durable=self.durable,
                status=snapshot.status,
            )
        # The status of the file on disk, which is not the status of the
        # snapshot resolution just read: a failed reload keeps enforcing the
        # last readable generation while `failure` records why the current
        # contents were refused.
        failure = self._state.failure
        return self._with_rejection_diagnostic(
            ranked,
            failure or snapshot.status,
            # Values were retained only when the generation in hand is itself
            # usable. A first read that fails records a failure too, but there
            # is no earlier generation behind it - those options fall back.
            retained=snapshot.status.usable,
        )

    @staticmethod
    def _with_rejection_diagnostic(
        ranked: RankedProviderValue[object],
        status: ProviderStatus,
        *,
        retained: bool,
    ) -> RankedProviderValue[object]:
        """Attach the reason when the file on disk was rejected as a whole.

        Neither rejection is visible in the result otherwise. A file refused on
        first read coerces to `Unset` for every option, which resolution reads
        as "this source declares nothing" — indistinguishable from a file that
        omits the key. A file refused on *reload* is quieter still: resolution
        keeps returning the previous generation's values, so nothing looks
        wrong at all while the user's latest edit silently fails to apply.

        Args:
            ranked: Result already coerced from the snapshot.
            status: Health of the file on disk.
            retained: Whether resolution is still serving an earlier generation
                rather than falling through to lower tiers.

        Returns:
            The result, with a rejection diagnostic when the source is unusable.
        """
        if status.usable:
            return ranked
        location = f" ({status.path})" if status.path is not None else ""
        detail = f": {status.detail}" if status.detail else ""
        suffix = RETAINED_SOURCE_SUFFIX if retained else UNUSABLE_SOURCE_SUFFIX
        reason = (
            f"Ignoring {status.name}{location} — it is "
            f"{status.health.value}{detail} {suffix}"
        )
        return replace(ranked, diagnostics=(reason, *ranked.diagnostics))

    def status(self) -> ProviderStatus:
        """Return health for the current file snapshot.

        A failed reload reports its own health even though resolution keeps
        using the retained snapshot: diagnostics must describe the file on
        disk, not the generation still being enforced.

        Raises:
            RuntimeError: If a reload produces no snapshot.
        """
        state = self._state
        if state.value is None:
            self.reload()
        if state.failure is not None:
            return state.failure
        snapshot = state.value
        if snapshot is None:
            msg = f"{self.name} reload produced no snapshot"
            raise RuntimeError(msg)
        return snapshot.status

    def reload(self) -> None:
        """Replace the current snapshot with a fresh file read.

        A reload the source cannot use never replaces the last usable
        snapshot. An unusable candidate carries an empty table, which
        resolution reads as "this source declares nothing"; installing it
        would drop the source's restrictions and let lower ranks win. The
        failed status is still recorded so health surfaces report the file on
        disk.
        """
        snapshot = self.loader() if self.loader is not None else self.load()
        if snapshot.status.usable:
            self._state.value = snapshot
            self._state.failure = None
        else:
            if self._state.value is None:
                self._state.value = snapshot
            self._state.failure = snapshot.status

    def current_snapshot(self) -> TomlSnapshot:
        """Return the cached snapshot, loading it on first access.

        Public so `ConfigResolver.toml_snapshot` can hand the generation this
        provider is serving to a caller building a masked variant resolver.

        Returns:
            Current parsed file snapshot.

        Raises:
            RuntimeError: If a reload produces no snapshot.
        """
        if self._state.value is None:
            self.reload()
        snapshot = self._state.value
        if snapshot is None:
            msg = f"{self.name} reload produced no snapshot"
            raise RuntimeError(msg)
        return snapshot


@dataclass(frozen=True, slots=True)
class EnvProvider:
    """Live process-environment configuration provider."""

    name: str = "environment"
    rank: int = ENVIRONMENT_RANK
    environ: Mapping[str, str] = field(
        default_factory=lambda: os.environ,
        repr=False,
        compare=False,
    )

    @property
    def durable(self) -> bool:
        """Never durable: the environment does not survive the process.

        A property rather than a field because the coercion helpers below stamp
        durability onto every value they emit. A settable field would
        type-check, enter `__eq__`, and change nothing about masking - a lie in
        the one attribute that decides whether a tier can hide another.
        """
        return False

    def get(self, option: ConfigOption) -> RankedProviderValue[object]:
        """Read one option from the live environment.

        Args:
            option: Manifest option to read.

        Returns:
            Ranked and coerced provider result.
        """
        from deepagents_code.config_manifest import OptionKind

        if option.kind is OptionKind.THEME_DELEGATE:
            return ranked_theme_environment_value(self.environ, rank=self.rank)
        return ranked_environment_value(option, self.environ, rank=self.rank)

    def status(self) -> ProviderStatus:
        """Return the always-healthy environment provider status."""
        return ProviderStatus(self.name, None, ProviderHealth.OK)

    def reload(self) -> None:
        """Keep reading the live environment without cached state."""


@dataclass(frozen=True, slots=True)
class DefaultProvider:
    """Typed manifest-default configuration provider."""

    name: str = "default"
    rank: int = DEFAULT_RANK

    @property
    def durable(self) -> bool:
        """Always durable: manifest defaults are compiled into the process.

        A property for the same reason as `EnvProvider.durable`: the value the
        helpers stamp on each result is the truth, so the attribute must not be
        able to disagree with it.
        """
        return True

    def get(self, option: ConfigOption) -> RankedProviderValue[object]:
        """Return one option's manifest default.

        Args:
            option: Manifest option whose default should be returned.

        Returns:
            Ranked default provider result.
        """
        ranked = ranked_default_value(option, rank=self.rank)
        if isinstance(ranked.result, Unset):
            return replace(ranked, result=Found(option.default))
        return ranked

    def status(self) -> ProviderStatus:
        """Return the always-healthy default provider status."""
        return ProviderStatus(self.name, None, ProviderHealth.OK)

    def reload(self) -> None:
        """Retain immutable manifest defaults without cached state."""
