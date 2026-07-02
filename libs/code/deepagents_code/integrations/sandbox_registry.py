"""Discovery and instantiation of sandbox providers.

Merges three provider sources into one registry:

1. Built-in providers curated in this repo (installed as `deepagents-code`
   extras).
2. Entry-point providers published by third-party packages under the
   `deepagents_code.sandbox_providers` group.
3. Config-declared providers from `[sandboxes.providers]` in
   `~/.deepagents/config.toml` (escape hatch for internal/local packages).

Precedence on name collision: config > entry point > built-in, so a user can
always override discovery via their config file.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import logging
from typing import TYPE_CHECKING

from deepagents_code.integrations.sandbox_config import SandboxConfig
from deepagents_code.integrations.sandbox_provider import (
    SandboxInstallHint,
    SandboxProvider,
    SandboxProviderMetadata,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "deepagents_code.sandbox_providers"
"""Entry-point group third-party packages publish providers under."""

BUILTIN_METADATA: dict[str, SandboxProviderMetadata] = {
    "agentcore": SandboxProviderMetadata(
        name="agentcore",
        working_dir="/tmp",  # noqa: S108  # AgentCore Code Interpreter working directory
        install=SandboxInstallHint(kind="extra", name="agentcore"),
        supports_sandbox_id=False,
        backend_module="langchain_agentcore_codeinterpreter",
    ),
    "daytona": SandboxProviderMetadata(
        name="daytona",
        working_dir="/home/daytona",
        install=SandboxInstallHint(kind="extra", name="daytona"),
        backend_module="langchain_daytona",
    ),
    "langsmith": SandboxProviderMetadata(
        name="langsmith",
        working_dir="/root",  # `$HOME` in the LangSmith sandbox
        # Bundled with `deepagents-code` via `langsmith[sandbox]`; no extra.
        supports_snapshot_name=True,
    ),
    "modal": SandboxProviderMetadata(
        name="modal",
        working_dir="/workspace",
        install=SandboxInstallHint(kind="extra", name="modal"),
        backend_module="langchain_modal",
    ),
    "runloop": SandboxProviderMetadata(
        name="runloop",
        working_dir="/home/user",
        install=SandboxInstallHint(kind="extra", name="runloop"),
        supports_snapshot_name=True,
        backend_module="langchain_runloop",
    ),
    "vercel": SandboxProviderMetadata(
        name="vercel",
        working_dir="/vercel/sandbox",
        install=SandboxInstallHint(kind="extra", name="vercel"),
        supports_sandbox_id=True,
        supports_snapshot_name=False,
        backend_module="langchain_vercel_sandbox",
    ),
}
"""Metadata for curated built-in providers, keyed by provider name."""


def _load_class(class_path: str) -> type:
    """Import a `module.path:ClassName` provider class.

    Args:
        class_path: Fully-qualified class path.

    Returns:
        The imported class object.

    Raises:
        ValueError: If `class_path` is malformed.
        ImportError: If the module cannot be imported or lacks the class.
    """
    if ":" not in class_path:
        msg = (
            f"Invalid class_path '{class_path}': must be in "
            "module.path:ClassName format"
        )
        raise ValueError(msg)
    module_path, class_name = class_path.rsplit(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name, None)
    if cls is None or not isinstance(cls, type):
        msg = f"Class '{class_name}' not found in module '{module_path}'"
        raise ImportError(msg)
    return cls


def _provider_metadata(provider: SandboxProvider, name: str) -> SandboxProviderMetadata:
    """Extract metadata from a provider instance or class.

    Providers may expose a `metadata` attribute/property; otherwise a minimal
    default is synthesized.

    Args:
        provider: Provider instance.
        name: Provider name to use when synthesizing defaults.

    Returns:
        The provider's metadata.
    """
    meta = getattr(provider, "metadata", None)
    if isinstance(meta, SandboxProviderMetadata):
        return meta
    return SandboxProviderMetadata(name=name, working_dir="/workspace")


class SandboxRegistry:
    """Merged view of built-in, entry-point, and config sandbox providers."""

    def __init__(
        self,
        *,
        config: SandboxConfig | None = None,
        include_entry_points: bool = True,
    ) -> None:
        """Build the registry.

        Args:
            config: Parsed sandbox config. Loaded from the default path when
                omitted.
            include_entry_points: Whether to discover entry-point providers.
                Disabled in tests that need a deterministic provider set.
        """
        self._config = config if config is not None else SandboxConfig.load()
        self._include_entry_points = include_entry_points
        self._entry_points: dict[str, importlib.metadata.EntryPoint] = (
            self._discover_entry_points() if include_entry_points else {}
        )

    @classmethod
    def load(cls, config_path: Path | None = None) -> SandboxRegistry:
        """Build a registry from the config file at `config_path`.

        Args:
            config_path: Path to config file. Defaults to the user config.

        Returns:
            A new `SandboxRegistry`.
        """
        return cls(config=SandboxConfig.load(config_path))

    @staticmethod
    def _discover_entry_points() -> dict[str, importlib.metadata.EntryPoint]:
        """Return entry-point providers keyed by name (best-effort)."""
        found: dict[str, importlib.metadata.EntryPoint] = {}
        try:
            entries = importlib.metadata.entry_points(group=ENTRY_POINT_GROUP)
        except Exception:
            logger.warning("Failed to discover sandbox entry points", exc_info=True)
            return found
        for entry in entries:
            found[entry.name] = entry
        return found

    @property
    def default(self) -> str | None:
        """The configured default provider, if any."""
        return self._config.default

    @property
    def config_error(self) -> str | None:
        """Why the config file failed to load, if it existed but couldn't parse.

        `None` when the config parsed cleanly or simply wasn't present.
        """
        return self._config.parse_error

    def available_providers(self) -> list[str]:
        """Return all known provider names, sorted."""
        names = (
            set(BUILTIN_METADATA)
            | set(self._entry_points)
            | set(self._config.providers)
        )
        return sorted(names)

    def is_available(self, name: str) -> bool:
        """Return whether `name` resolves to a known provider."""
        return (
            name in self._config.providers
            or name in self._entry_points
            or name in BUILTIN_METADATA
        )

    def get_metadata(self, name: str) -> SandboxProviderMetadata | None:
        """Return metadata for `name`.

        Config providers may override `working_dir`, `package`, and capability
        flags. Entry-point providers are probed for advertised metadata and
        take precedence over built-ins with the same name.

        Args:
            name: Provider name.

        Returns:
            The provider's metadata, or `None` if unknown.
        """
        config_entry = self._config.providers.get(name)
        if config_entry is not None:
            base = BUILTIN_METADATA.get(name)
            package = config_entry.get("package")
            return SandboxProviderMetadata(
                name=name,
                working_dir=config_entry.get(
                    "working_dir", base.working_dir if base else "/workspace"
                ),
                install=(
                    SandboxInstallHint(kind="package", name=package)
                    if package
                    else (base.install if base else None)
                ),
                supports_sandbox_id=config_entry.get(
                    "supports_sandbox_id",
                    base.supports_sandbox_id if base else True,
                ),
                supports_snapshot_name=config_entry.get(
                    "supports_snapshot_name",
                    base.supports_snapshot_name if base else False,
                ),
                # Carry the built-in's probe module so a config override of a
                # built-in keeps its dependency pre-flight check. A pure config
                # provider (no base) leaves this `None`; its package is resolved
                # when the provider class is imported.
                backend_module=base.backend_module if base else None,
            )
        if name in self._entry_points:
            return self.provider_metadata(name)
        if name in BUILTIN_METADATA:
            return BUILTIN_METADATA[name]
        return None

    def get_params(self, name: str) -> dict[str, object]:
        """Return config `params` forwarded to the provider's `get_or_create()`."""
        return self._config.get_params(name)

    def create_provider(self, name: str) -> SandboxProvider:
        """Instantiate the provider named `name`.

        Resolution order: config `class_path` > entry point > built-in.

        Args:
            name: Provider name.

        Returns:
            A `SandboxProvider` instance. Propagates `ImportError` from
                `_load_class` / `EntryPoint.load` if a config or entry-point
                class cannot be imported.

        Raises:
            ValueError: If `name` is unknown or a config provider omits
                `class_path`.
        """
        config_entry = self._config.providers.get(name)
        if config_entry is not None:
            class_path = config_entry.get("class_path")
            if not class_path:
                msg = f"Sandbox provider '{name}' config is missing 'class_path'"
                raise ValueError(msg)
            return _load_class(class_path)()

        entry = self._entry_points.get(name)
        if entry is not None:
            return entry.load()()

        if name in BUILTIN_METADATA:
            return _create_builtin_provider(name)

        msg = (
            f"Unknown sandbox provider: {name}. "
            f"Available providers: {', '.join(self.available_providers())}"
        )
        raise ValueError(msg)

    def provider_metadata(self, name: str) -> SandboxProviderMetadata:
        """Return authoritative metadata for `name`.

        Config providers are described statically. Entry-point providers are
        instantiated so capability flags they expose via a `metadata` attribute
        take effect; on failure this falls back to the static placeholder.
        Built-in metadata is used only when no entry point overrides that name.

        Args:
            name: Provider name.

        Returns:
            The provider's metadata.

        Raises:
            ValueError: If `name` is unknown.
        """
        if name in self._config.providers or (
            name in BUILTIN_METADATA and name not in self._entry_points
        ):
            meta = self.get_metadata(name)
            if meta is not None:
                return meta
        if name in self._entry_points:
            try:
                provider = self.create_provider(name)
            except Exception:  # noqa: BLE001  # Metadata probe must not crash discovery
                logger.debug("Could not instantiate provider %r for metadata", name)
                return SandboxProviderMetadata(name=name, working_dir="/workspace")
            return _provider_metadata(provider, name)
        meta = self.get_metadata(name)
        if meta is None:
            msg = f"Unknown sandbox provider: {name}"
            raise ValueError(msg)
        return meta


def _create_builtin_provider(name: str) -> SandboxProvider:
    """Instantiate a built-in provider class (lazy import avoids cycles).

    Returns:
        The built-in `SandboxProvider` instance for `name`.
    """
    from deepagents_code.integrations import sandbox_factory

    builders = {
        "agentcore": sandbox_factory._AgentCoreProvider,
        "daytona": sandbox_factory._DaytonaProvider,
        "langsmith": sandbox_factory._LangSmithProvider,
        "modal": sandbox_factory._ModalProvider,
        "runloop": sandbox_factory._RunloopProvider,
        "vercel": sandbox_factory._VercelProvider,
    }
    return builders[name]()
