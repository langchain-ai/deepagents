"""UI-agnostic helpers for resolving an MCP login target.

The MCP login flow historically inlined config discovery, trust gating,
shape validation, and `print()`-based error reporting. The TUI cannot
consume those print statements, so this module extracts the same logic
into pure functions that return structured results (`ConfigResolution`,
`ServerSelection`) plus a typed `ConfigResolutionError`. Callers decide
how to render those results.

No `print()` calls live in this module. No imports happen at module
top level beyond `dataclasses`/`typing`/`pathlib` so the CLI fast path
stays cheap; the actual config loaders are imported inside the
functions that need them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from deepagents_code.mcp_auth import McpServerSpec


class ConfigErrorKind(StrEnum):
    """Discriminator for `ConfigResolutionError` reasons.

    Only `NO_CONFIG_FOUND` maps to exit code 2 in `run_mcp_login`; all
    other kinds map to exit code 1. The TUI surface translates them into
    in-app status messages.
    """

    EXPLICIT_LOAD_FAILED = "explicit_load_failed"
    """The `--mcp-config` path could not be parsed."""

    NO_CONFIG_FOUND = "no_config_found"
    """Auto-discovery returned zero candidate paths."""

    NO_USABLE_CONFIG = "no_usable_config"
    """Discovered paths existed but none could be loaded successfully."""

    UNKNOWN_SERVER = "unknown_server"
    """The selected server is not present in the resolved config."""

    INVALID_SERVER_CONFIG = "invalid_server_config"
    """The selected server's entry failed shape validation."""


@dataclass(frozen=True)
class ConfigResolutionError:
    """Structured error returned when a login target cannot be resolved."""

    kind: ConfigErrorKind
    """Reason category — callers translate this into UI text or exit codes."""

    message: str
    """Plain-text description suitable for direct display to the user."""

    untrusted_project_paths: tuple[Path, ...] = ()
    """Project-level configs skipped because the trust store had no match.

    Populated only when at least one discovered project config was
    skipped during auto-discovery, regardless of `kind`. Callers can
    surface a "skipping untrusted project config" hint alongside the
    primary error.
    """


@dataclass(frozen=True)
class ConfigResolution:
    """Successful resolution of a merged MCP config for login."""

    config: dict[str, Any]
    """The merged `mcpServers`-shaped config dict."""

    used_paths: tuple[Path, ...]
    """Paths whose contents were merged into `config`, in precedence order."""

    untrusted_project_paths: tuple[Path, ...] = ()
    """Project-level configs that were skipped during this resolution."""

    def __post_init__(self) -> None:
        """Enforce the non-empty `used_paths` invariant.

        Raises:
            ValueError: If `used_paths` is empty.
        """
        if not self.used_paths:
            msg = "ConfigResolution must have at least one used path"
            raise ValueError(msg)

    @property
    def search_label(self) -> str:
        """Human-readable join of the paths backing this resolution."""
        return ", ".join(str(path) for path in self.used_paths)


@dataclass(frozen=True)
class ServerSelection:
    """Resolved server config plus enough context for error messages."""

    server_name: str
    """Selected MCP server name (matches an `mcpServers` key)."""

    server_config: McpServerSpec
    """Validated server config payload for `mcp_auth.login`."""

    search_label: str = ""
    """Where the config came from — used in not-found errors."""

    def __post_init__(self) -> None:
        """Enforce the non-empty `server_name` invariant.

        Raises:
            ValueError: If `server_name` is empty.
        """
        if not self.server_name:
            msg = "ServerSelection.server_name must not be empty"
            raise ValueError(msg)


def resolve_mcp_config(
    config_path: str | None,
) -> ConfigResolution | ConfigResolutionError:
    """Resolve an MCP config dict for login without printing anything.

    Args:
        config_path: Explicit `--mcp-config` path, or `None` for auto-discovery.

    Returns:
        A `ConfigResolution` on success, or a `ConfigResolutionError`
            describing why no usable config could be assembled.
    """
    from deepagents_code.mcp_tools import (
        classify_discovered_configs,
        discover_mcp_configs,
        load_mcp_config,
        load_mcp_config_lenient,
        merge_mcp_configs,
    )

    if config_path is not None:
        try:
            config = load_mcp_config(config_path)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            return ConfigResolutionError(
                kind=ConfigErrorKind.EXPLICIT_LOAD_FAILED,
                message=f"Failed to load MCP config {config_path}: {exc}",
            )
        return ConfigResolution(
            config=config,
            used_paths=(Path(config_path),),
        )

    found = discover_mcp_configs()
    if not found:
        return ConfigResolutionError(
            kind=ConfigErrorKind.NO_CONFIG_FOUND,
            message=(
                "No MCP config file found in any auto-discovered location. "
                "Pass --mcp-config <path>, or run `dcode mcp login --help` "
                "to see the search paths and config format."
            ),
        )

    user_paths, project_paths = classify_discovered_configs(found)
    configs: list[dict[str, Any]] = []
    used_paths: list[Path] = []
    untrusted: tuple[Path, ...] = ()

    for path in user_paths:
        loaded = load_mcp_config_lenient(path)
        if loaded is not None:
            configs.append(loaded)
            used_paths.append(path)

    if project_paths:
        from deepagents_code.mcp_trust import (
            compute_config_fingerprint,
            is_project_mcp_trusted,
        )
        from deepagents_code.project_utils import find_project_root

        project_root = str((find_project_root() or Path.cwd()).resolve())
        fingerprint = compute_config_fingerprint(project_paths)
        if is_project_mcp_trusted(project_root, fingerprint):
            for path in project_paths:
                loaded = load_mcp_config_lenient(path)
                if loaded is not None:
                    configs.append(loaded)
                    used_paths.append(path)
        else:
            untrusted = tuple(project_paths)

    if not configs:
        found_paths = ", ".join(str(path) for path in found)
        return ConfigResolutionError(
            kind=ConfigErrorKind.NO_USABLE_CONFIG,
            message=f"No usable MCP config found in: {found_paths}",
            untrusted_project_paths=untrusted,
        )

    return ConfigResolution(
        config=merge_mcp_configs(configs),
        used_paths=tuple(used_paths),
        untrusted_project_paths=untrusted,
    )


def select_server(
    resolution: ConfigResolution,
    server: str,
) -> ServerSelection | ConfigResolutionError:
    """Pull `server` out of a resolved config and validate its shape.

    Args:
        resolution: A successful `resolve_mcp_config` result.
        server: Target server name as supplied by the user.

    Returns:
        A `ServerSelection` on success, or a `ConfigResolutionError`
            describing why the server entry is unusable.
    """
    from deepagents_code.mcp_tools import _validate_server_config

    servers = resolution.config.get("mcpServers", {})
    if server not in servers:
        return ConfigResolutionError(
            kind=ConfigErrorKind.UNKNOWN_SERVER,
            message=(
                f"Server {server!r} not found in {resolution.search_label}. "
                f"Known servers: {sorted(servers)}"
            ),
        )

    try:
        _validate_server_config(server, servers[server])
    except (TypeError, ValueError) as exc:
        return ConfigResolutionError(
            kind=ConfigErrorKind.INVALID_SERVER_CONFIG,
            message=f"Invalid MCP server config for {server!r}: {exc}",
        )

    return ServerSelection(
        server_name=server,
        server_config=servers[server],
        search_label=resolution.search_label,
    )


def format_untrusted_project_notice(paths: tuple[Path, ...]) -> str:
    """Build the CLI-style hint string for skipped untrusted project configs.

    Args:
        paths: Project configs that were skipped during resolution.

    Returns:
        A single-line user-facing string. Empty when `paths` is empty.
    """
    if not paths:
        return ""
    skipped = ", ".join(str(path) for path in paths)
    return (
        "Skipping untrusted project MCP config "
        f"(not yet approved or config changed): {skipped}. "
        "Approve it by running `dcode` in this project, or "
        "pass --mcp-config <path> to use it explicitly."
    )


__all__ = [
    "ConfigErrorKind",
    "ConfigResolution",
    "ConfigResolutionError",
    "ServerSelection",
    "format_untrusted_project_notice",
    "resolve_mcp_config",
    "select_server",
]
