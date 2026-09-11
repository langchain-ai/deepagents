"""Parsing for the `[sandboxes]` section of `~/.deepagents/config.toml`.

Parallels the `[models]` provider configuration in `model_config.py`. Config
providers declare a `class_path` (same trust model as model `class_path`),
a `working_dir`, an optional install `package`, and `params` forwarded to
`provider.get_or_create()`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypedDict, cast

from deepagents_code.model_config import DEFAULT_CONFIG_PATH

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

logger = logging.getLogger(__name__)


def _normalize_provider_configs(
    providers: dict[str, Any],
) -> dict[str, SandboxProviderConfig]:
    """Drop malformed provider entries before constructing `SandboxConfig`.

    Args:
        providers: Raw provider mapping from TOML.

    Returns:
        Provider entries that are valid TOML tables.
    """
    normalized: dict[str, SandboxProviderConfig] = {}
    for name, provider in providers.items():
        if not isinstance(provider, dict):
            logger.warning(
                "Sandbox provider '%s' is not a table (%s); ignoring it",
                name,
                type(provider).__name__,
            )
            continue
        normalized[name] = cast("SandboxProviderConfig", provider)
    return normalized


class SandboxProviderConfig(TypedDict, total=False):
    """Configuration for a single config-declared sandbox provider.

    !!! warning

        Setting `class_path` executes arbitrary Python code imported from the
        user's config file. This has the same trust model as model
        `class_path` — the user controls their own machine.
    """

    class_path: str
    """Fully-qualified provider class in `module.path:ClassName` format."""

    working_dir: str
    """Default working directory inside the sandbox."""

    package: str
    """Package suggested when the provider's dependencies are missing."""

    supports_sandbox_id: bool
    """Whether the provider can reattach to an existing sandbox by id."""

    supports_snapshot_name: bool
    """Whether the provider honors `--sandbox-snapshot-name`."""

    params: dict[str, Any]
    """Extra keyword arguments forwarded to `provider.get_or_create()`."""


@dataclass(frozen=True)
class SandboxConfig:
    """Parsed `[sandboxes]` configuration from `config.toml`.

    Instances are immutable once constructed; `providers` is wrapped in a
    `MappingProxyType` to prevent accidental mutation.
    """

    default: str | None = None
    """The configured default provider (from `[sandboxes].default`).

    Only applied when the user explicitly opts into sandbox mode; a config
    value never silently enables sandbox mode.
    """

    providers: Mapping[str, SandboxProviderConfig] = field(default_factory=dict)
    """Read-only mapping of provider names to their configurations."""

    parse_error: str | None = None
    """Set when the config file existed but could not be read or parsed.

    `load()` degrades to an empty config on malformed TOML or an unreadable
    file so unrelated startup keeps working, but the user explicitly opted into
    a sandbox. Callers surface this so the failure isn't invisible (a bare
    `logger.warning` never reaches the TUI).
    """

    def __post_init__(self) -> None:
        """Freeze the providers dict into a read-only proxy."""
        if not isinstance(self.providers, MappingProxyType):
            object.__setattr__(self, "providers", MappingProxyType(self.providers))

    @classmethod
    def load(cls, config_path: Path | None = None) -> SandboxConfig:
        """Load the `[sandboxes]` section from a config file.

        Args:
            config_path: Passing a path also excludes managed policy from
                this read, so production callers must pass `None`. Defaults to
                `~/.deepagents/config.toml`.

        Returns:
            Parsed `SandboxConfig`. A missing user file yields managed values
                alone. An unreadable or invalid user file is reported through
                `parse_error`, and managed values still apply.

        Raises:
            RuntimeError: If required sandbox options are missing from the manifest.
        """
        is_default = config_path is None
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH

        from deepagents_code.config_manifest import get_option
        from deepagents_code.configuration.resolver import resolver_from_snapshots
        from deepagents_code.configuration.service import get_config_sources
        from deepagents_code.configuration.types import Invalid, ProviderHealth

        # `None` on the default path: that is what includes managed policy.
        sources = get_config_sources(user_path=None if is_default else config_path)
        # A bad user file is reported through `parse_error` but must not
        # discard administrator policy, which parsed cleanly on its own.
        parse_error: str | None = None
        if sources.user.status.health is ProviderHealth.CORRUPT:
            detail = sources.user.status.detail or "unknown parse error"
            logger.warning(
                "Config file %s has invalid TOML syntax: %s. "
                "Ignoring user sandbox config.",
                config_path,
                detail,
            )
            parse_error = f"invalid TOML syntax: {detail}"
        elif sources.user.status.health is ProviderHealth.UNREADABLE:
            detail = sources.user.status.detail or "unknown read error"
            logger.warning("Could not read config file %s: %s", config_path, detail)
            parse_error = f"could not read config file: {detail}"
        dropped = sources.dropped_managed_detail()
        if dropped is not None:
            logger.error(
                "Managed policy from %s is not being applied: %s",
                sources.managed.status.path,
                dropped,
            )
        section = (
            sources.managed.data.get("sandboxes")
            if "sandboxes" in sources.managed.data
            else sources.user.data.get("sandboxes", {})
        )
        if not isinstance(section, dict):
            logger.warning("[sandboxes] is not a table; ignoring sandbox config")
            return cls(parse_error=parse_error or "[sandboxes] is not a table")

        default_option = get_option("sandboxes.default")
        providers_option = get_option("sandboxes.providers")
        if default_option is None or providers_option is None:
            msg = "sandbox options are missing from the config manifest"
            raise RuntimeError(msg)
        # Resolve against the supplied snapshots: a non-default `config_path`
        # deliberately excludes managed policy, and the shared process cache
        # always reads the default path.
        resolver = resolver_from_snapshots(managed=sources.managed, user=sources.user)
        default_resolved = resolver.get(default_option)
        default = default_resolved.value
        if any(
            isinstance(result, Invalid)
            for result in default_resolved.tier_health.values()
        ):
            # Without this the value degrades to `None` and nothing is logged,
            # on the option that decides which sandbox executes agent code.
            # `dcode doctor` covers file health, not value health -- the file
            # parses fine, and the rejected entry is reported nowhere.
            logger.warning(
                "[sandboxes].default is not a string; ignoring the default sandbox"
            )
        providers_resolved = resolver.get(providers_option)
        providers = providers_resolved.value
        if providers is None:
            if any(
                isinstance(result, Invalid)
                for result in providers_resolved.tier_health.values()
            ):
                logger.warning(
                    "[sandboxes.providers] is not a table; ignoring sandbox providers"
                )
            providers = {}
        elif not isinstance(providers, dict):
            logger.warning(
                "[sandboxes.providers] is not a table; ignoring sandbox providers"
            )
            providers = {}
        provider_table = cast("dict[str, Any]", providers)

        config = cls(
            default=default if isinstance(default, str) else None,
            providers=_normalize_provider_configs(provider_table),
            parse_error=parse_error,
        )
        config._validate()
        return config

    def _validate(self) -> None:
        """Warn about malformed config without raising."""
        for name, provider in self.providers.items():
            class_path = provider.get("class_path")
            if not class_path:
                logger.warning(
                    "Sandbox provider '%s' is missing required 'class_path'", name
                )
            elif ":" not in class_path:
                logger.warning(
                    "Sandbox provider '%s' has invalid class_path '%s': "
                    "must be in module.path:ClassName format",
                    name,
                    class_path,
                )
            params = provider.get("params")
            if params is not None and not isinstance(params, dict):
                logger.warning(
                    "Sandbox provider '%s' has non-table 'params' (%s); ignoring it",
                    name,
                    type(params).__name__,
                )

    def get_params(self, provider_name: str) -> dict[str, Any]:
        """Return the `params` forwarded to a provider's `get_or_create()`.

        Args:
            provider_name: The provider to look up.

        Returns:
            A copy of the configured params (empty if none configured).
        """
        provider = self.providers.get(provider_name)
        if not provider:
            return {}
        params = provider.get("params", {})
        return dict(params) if isinstance(params, dict) else {}
