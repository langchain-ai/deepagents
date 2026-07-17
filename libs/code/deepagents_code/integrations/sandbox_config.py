"""Parsing for the `[sandboxes]` section of `~/.deepagents/config.toml`.

Parallels the `[models]` provider configuration in `model_config.py`. Config
providers declare a `class_path` (same trust model as model `class_path`),
a `working_dir`, an optional install `package`, and `params` forwarded to
`provider.get_or_create()`.
"""

from __future__ import annotations

import logging
import tomllib
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
            config_path: Path to config file. Defaults to
                `~/.deepagents/config.toml`.

        Returns:
            Parsed `SandboxConfig`. Returns an empty config if the file is
                missing, unreadable, or contains invalid TOML.
        """
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH

        if not config_path.exists():
            return cls()

        try:
            with config_path.open("rb") as f:
                data = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            logger.warning(
                "Config file %s has invalid TOML syntax: %s. Ignoring sandbox config.",
                config_path,
                e,
            )
            return cls(parse_error=f"invalid TOML syntax: {e}")
        except (PermissionError, OSError) as e:
            logger.warning("Could not read config file %s: %s", config_path, e)
            return cls(parse_error=f"could not read config file: {e}")

        section = data.get("sandboxes", {})
        if not isinstance(section, dict):
            logger.warning("[sandboxes] is not a table; ignoring sandbox config")
            return cls(parse_error="[sandboxes] is not a table")

        providers = section.get("providers", {})
        if not isinstance(providers, dict):
            logger.warning(
                "[sandboxes.providers] is not a table; ignoring sandbox providers"
            )
            providers = {}

        config = cls(
            default=section.get("default"),
            providers=_normalize_provider_configs(providers),
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
