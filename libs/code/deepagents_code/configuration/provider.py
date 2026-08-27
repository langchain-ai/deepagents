"""Structural contract and CLI implementation for configuration providers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import argparse

    from deepagents_code.config_manifest import CliSpec, ConfigOption

from deepagents_code.configuration.resolver import CLI_RANK, RankedProviderValue
from deepagents_code.configuration.types import (
    Invalid,
    ProviderHealth,
    ProviderResult,
    ProviderStatus,
    Unset,
)


def parse_interpreter_tools(raw: str) -> str | list[str] | Invalid:
    """Parse the shared `--interpreter-tools` grammar.

    One rule serves both callers. `CliProvider` frames a rejection as a
    warning and falls through to the next tier; `main` frames the same reason
    as a usage error and exits 2. Two hand-maintained copies had already
    drifted in wording.

    Args:
        raw: Raw flag text as the user typed it.

    Returns:
        The `"safe"`/`"all"` sentinel, a list of tool names, or `Invalid`
        carrying the bare reason for the caller to frame.
    """
    text = raw.strip()
    if not text:
        return Invalid(
            "requires a value: 'safe', 'all', or a comma-separated list of tool names"
        )
    normalized = text.lower()
    if normalized in {"safe", "all"}:
        return normalized
    names = [token.strip() for token in text.split(",") if token.strip()]
    if not names:
        return Invalid("list must contain at least one non-empty tool name")
    if any(name.lower() == "all" for name in names):
        return Invalid(
            "'all' cannot be combined with other tools; use 'all' on its own "
            "or list explicit tool names (optionally with the 'safe' preset)"
        )
    return names


@runtime_checkable
class ConfigProvider(Protocol):
    """A ranked source of typed configuration values."""

    @property
    def name(self) -> str:
        """Provider display label."""
        ...

    @property
    def rank(self) -> int:
        """Numeric precedence rank."""
        ...

    @property
    def durable(self) -> bool:
        """Whether the source survives the process."""
        ...

    def get[T](self, option: ConfigOption[T]) -> RankedProviderValue[T]:
        """Read and coerce one manifest option."""
        ...

    def status(self) -> ProviderStatus:
        """Return current provider health and display metadata."""
        ...

    def reload(self) -> None:
        """Refresh provider state when the source supports it."""
        ...


@dataclass(frozen=True, slots=True, init=False)
class CliProvider:
    """Config provider backed by parsed CLI arguments."""

    args: Mapping[str, object] = field(repr=False)
    name: str = "CLI argument"
    rank: int = CLI_RANK

    def __init__(self, args: argparse.Namespace | Mapping[str, object]) -> None:
        """Snapshot an already-parsed CLI namespace.

        Args:
            args: Values produced by `argparse`, or an equivalent mapping.
        """
        # Discriminate on the declared type, not on `__dict__`: every `Mapping`
        # implementation other than `dict`/`MappingProxyType` also has one, so
        # `hasattr` sent them down the `vars()` branch and snapshotted the
        # object's attributes instead of its items -- every option then read
        # back `Unset` with no error anywhere.
        values = args if isinstance(args, Mapping) else vars(args)
        object.__setattr__(self, "args", MappingProxyType(dict(values)))
        object.__setattr__(self, "name", "CLI argument")
        object.__setattr__(self, "rank", CLI_RANK)

    @property
    def durable(self) -> bool:
        """Never durable: CLI state dies with this process."""
        return False

    def get[T](self, option: ConfigOption[T]) -> RankedProviderValue[T]:
        """Read and coerce one option from the parsed namespace.

        Args:
            option: Manifest option to read.

        Returns:
            Ranked `Found`, `Unset`, or `Invalid` provider result.
        """
        status = self.status()
        spec = option.cli
        if spec is None:
            return RankedProviderValue(self.rank, self.durable, status, Unset())

        if option.key == "startup.mode":
            result = self._startup_mode(option, spec)
        else:
            raw = self.args.get(spec.dest_name)
            result = (
                Unset() if raw is None else self._coerce(option, raw, flag=spec.flag)
            )
        return RankedProviderValue(self.rank, self.durable, status, result)

    def _startup_mode[T](
        self, option: ConfigOption[T], spec: CliSpec
    ) -> ProviderResult[T]:
        """Map the mutually exclusive approval flags to one config value.

        The companion destinations are derived from the spec rather than
        spelled out here. A literal `"yolo"` made `CliSpec` the source of truth
        for the flag's *display* while this method independently re-derived the
        destination it *reads*, so an argparse `dest=` rename would silently
        degrade YOLO to `Unset` -- the same class of failure the `Mapping`
        discrimination above guards against.

        Args:
            option: Manifest option that owns the startup-mode value type.
            spec: CLI binding for `startup.mode`, carrying both flags.

        Returns:
            `Found` for an explicit mode, otherwise `Unset`.
        """
        from deepagents_code.config_manifest import CliSpec
        from deepagents_code.configuration.providers import _found_for

        # Companions are checked first: `--yolo` is the stronger mode and wins
        # if a caller somehow supplies both. Each companion flag is spelled
        # exactly like the mode it selects (`--yolo` -> `"yolo"`), which
        # `test_companion_flags_spell_their_startup_mode` pins.
        for flag in spec.companion_flags:
            if self.args.get(CliSpec(flag).dest_name) is True:
                return _found_for(option, flag.removeprefix("--"))
        if self.args.get(spec.dest_name) is True:
            return _found_for(option, "auto")
        return Unset()

    @staticmethod
    def _coerce[T](
        option: ConfigOption[T], raw: object, *, flag: str
    ) -> ProviderResult[T]:
        """Coerce one already-parsed CLI value to its manifest type.

        Returns:
            Typed `Found` or an `Invalid` rejection.
        """
        from deepagents_code.config_manifest import OptionKind
        from deepagents_code.configuration.providers import (
            _found_for,
            coerce_environment_value,
            coerce_toml_value,
        )

        if option.key == "threads.sort_order":
            if raw == "created":
                return _found_for(option, "created_at")
            if raw == "updated":
                return _found_for(option, "updated_at")
            return Invalid(f"Ignoring {flag}={raw!r} (expected 'created' or 'updated')")
        if option.kind is OptionKind.SHELL_LIST_DELEGATE and isinstance(raw, str):
            return coerce_environment_value(option, raw, flag)
        if option.kind is OptionKind.PTC_DELEGATE and isinstance(raw, str):
            parsed = parse_interpreter_tools(raw)
            if isinstance(parsed, Invalid):
                return Invalid(f"Ignoring {flag}={raw!r} ({parsed.reason})")
            return coerce_toml_value(option, parsed, source=flag)
        if option.kind is OptionKind.STRUCTURED and isinstance(raw, (dict, list)):
            return _found_for(option, raw)
        if isinstance(raw, str) and not raw.strip():
            # A CLI value is a shell string, not a TOML literal: `--flag ""` is
            # an absent value, not an empty one. `coerce_toml_value` accepts the
            # empty string, so without this the blank flag wins its rank and
            # masks every real value in env and `config.toml`.
            if option.key == "models.auto_classifier":
                # Except this one: `parse_args` strips the flag's value, so an
                # empty string here can only be `--auto-classifier-model ""` --
                # the explicit "inherit the main agent model" instruction that
                # `run_textual_cli_async` maps to `INHERIT_CLASSIFIER_MODEL`
                # ahead of env and `config.toml`. `Unset` would abstain the CLI
                # tier, and `dcode --auto-classifier-model "" config get
                # models.auto_classifier` would report the very env/TOML
                # classifier the launch ignores. Returning the sentinel keeps
                # introspection and the launch path on the same value.
                from deepagents_code._cli_context import INHERIT_CLASSIFIER_MODEL

                return _found_for(option, INHERIT_CLASSIFIER_MODEL)
            if raw:
                return Invalid(
                    f"Ignoring {flag}={raw!r} (whitespace-only; treated as unset)"
                )
            return Unset()
        return coerce_toml_value(option, raw, source=flag)

    def status(self) -> ProviderStatus:
        """Return the always-healthy in-memory CLI status."""
        return ProviderStatus(self.name, None, ProviderHealth.OK)

    def reload(self) -> None:
        """Retain the immutable parsed argument snapshot."""
