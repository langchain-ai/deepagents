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
    """Project-level configs with server entries skipped by the trust gate.

    Populated when at least one discovered project config had server entries
    skipped during auto-discovery (unapproved, disabled, or because the user's
    trust policy could not be read), regardless of `kind`. Callers can surface a
    "skipping untrusted project servers" hint alongside the primary error.
    """

    legacy_ignored: tuple[str, ...] = ()
    """Names found in a legacy `[mcp].enabled_project_servers` list, sorted.

    Mirrors `resolve_and_load_mcp_tools`: non-empty means the user relied on the
    removed flat allowlist, so those servers silently stopped loading. Callers
    should surface the migration hint so this non-interactive path explains the
    change instead of the servers just vanishing.
    """

    policy_error: str | None = None
    """Set when the user's trust policy (`config.toml`) could not be read.

    When non-`None`, project servers were dropped because the policy failed to
    load — not because they were unapproved — so callers should surface this
    reason instead of the misleading `untrusted_project_paths` notice. On this
    error type `message` already embeds it; callers use the field only to
    suppress the misleading untrusted notice.
    """

    legacy_env_ignored: bool = False
    """`True` when the removed `DEEPAGENTS_CODE_ENABLED_PROJECT_MCP_SERVERS` env
    var is set. Twin of `legacy_ignored` for the env surface; see
    `format_legacy_env_ignored_notice`."""


@dataclass(frozen=True)
class ConfigResolution:
    """Successful resolution of a merged MCP config for login."""

    config: dict[str, Any]
    """The merged `mcpServers`-shaped config dict."""

    used_paths: tuple[Path, ...]
    """Paths whose contents were merged into `config`, in precedence order."""

    untrusted_project_paths: tuple[Path, ...] = ()
    """Project-level configs with server entries skipped by the trust gate."""

    legacy_ignored: tuple[str, ...] = ()
    """Names from a legacy `[mcp].enabled_project_servers` list, now ignored.

    See `ConfigResolutionError.legacy_ignored`; surfaced even on success because
    the requested server may load while other legacy-listed servers do not.
    """

    policy_error: str | None = None
    """Set when the user's trust policy could not be read (fail-closed).

    Non-`None` even on a successful resolution — user-level configs and any
    `DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS` env names still load, but scoped
    project approvals were discarded. Surfaced so the read failure is never
    silently swallowed just because some other config remained usable. See
    `ConfigResolutionError.policy_error`.
    """

    legacy_env_ignored: bool = False
    """`True` when the removed `DEEPAGENTS_CODE_ENABLED_PROJECT_MCP_SERVERS` env
    var is set. See `ConfigResolutionError.legacy_env_ignored`."""

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
        _load_mcp_config_top_level_with_error,
        _merge_mcp_configs_with_sources,
        _resolve_project_config_base,
        _validate_mcp_config_servers,
        classify_discovered_configs,
        discover_mcp_configs,
        filter_trusted_project_servers,
        load_mcp_config,
        load_mcp_config_with_error,
        merge_mcp_configs,
        project_root_for_mcp_config_path,
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
    # Parse failures of discovered files, surfaced when nothing usable remains
    # so the user is told *why* (mirrors resolve_and_load_mcp_tools) instead of
    # a bare "no usable config" that hides a JSON syntax error.
    load_errors: list[tuple[Path, str]] = []
    policy_error: str | None = None
    legacy_ignored: tuple[str, ...] = ()
    legacy_env_ignored = False

    for path in user_paths:
        loaded, error = load_mcp_config_with_error(path)
        if loaded is not None:
            configs.append(loaded)
            used_paths.append(path)
        elif error is not None:
            load_errors.append((path, error))

    if project_paths:
        from deepagents_code.model_config import load_mcp_server_trust_lists

        trust_lists = load_mcp_server_trust_lists()
        legacy_ignored = tuple(sorted(trust_lists.legacy_ignored))
        legacy_env_ignored = trust_lists.legacy_env_ignored
        project_base = _resolve_project_config_base(None)
        untrusted_paths: list[Path] = []
        if trust_lists.load_failed:
            # Whole-config trust and scoped TOML approvals fail closed. The
            # trust-list loader has already discarded scoped approvals while
            # retaining names explicitly enabled through the readable env var.
            policy_error = trust_lists.read_error
        loaded_projects: list[tuple[Path, dict[str, Any]]] = []
        for path in project_paths:
            loaded, error = _load_mcp_config_top_level_with_error(path)
            if loaded is not None:
                loaded_projects.append((path, loaded))
            elif error is not None:
                load_errors.append((path, error))

        if loaded_projects:
            project_config, server_sources = _merge_mcp_configs_with_sources(
                loaded_projects
            )
            servers = project_config["mcpServers"]
            kept: dict[str, Any] = {}
            for name, server in servers.items():
                source = server_sources[name]
                project_root = project_root_for_mcp_config_path(
                    source, fallback=project_base
                )
                kept.update(
                    filter_trusted_project_servers(
                        {name: server}, trust_lists, project_root=project_root
                    )
                )

            if kept:
                filtered = {**project_config, "mcpServers": kept}
                try:
                    _validate_mcp_config_servers(filtered)
                except (ValueError, TypeError, RuntimeError) as exc:
                    load_errors.append((loaded_projects[-1][0], str(exc)))
                else:
                    configs.append(filtered)
                    kept_sources = {server_sources[name] for name in kept}
                    used_paths.extend(
                        path for path in project_paths if path in kept_sources
                    )

            dropped_sources = {
                server_sources[name] for name in servers if name not in kept
            }
            untrusted_paths.extend(
                path for path in project_paths if path in dropped_sources
            )
        untrusted = tuple(untrusted_paths)

    if not configs:
        if policy_error is not None:
            message = _policy_error_message(policy_error)
        elif load_errors:
            detail = "; ".join(f"{path}: {error}" for path, error in load_errors)
            message = f"No usable MCP config found (load errors: {detail})"
        else:
            found_paths = ", ".join(str(path) for path in found)
            message = f"No usable MCP config found in: {found_paths}"
        return ConfigResolutionError(
            kind=ConfigErrorKind.NO_USABLE_CONFIG,
            message=message,
            untrusted_project_paths=untrusted,
            legacy_ignored=legacy_ignored,
            policy_error=policy_error,
            legacy_env_ignored=legacy_env_ignored,
        )

    return ConfigResolution(
        config=merge_mcp_configs(configs),
        used_paths=tuple(used_paths),
        untrusted_project_paths=untrusted,
        legacy_ignored=legacy_ignored,
        policy_error=policy_error,
        legacy_env_ignored=legacy_env_ignored,
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


def _policy_error_message(policy_error: str) -> str:
    """Return the user-facing message for an unreadable trust policy."""
    return (
        f"Refusing to trust project MCP servers: {policy_error}. Fix "
        "~/.deepagents/config.toml, or pass --mcp-config <path> to load "
        "a file explicitly."
    )


def format_policy_error_notice(policy_error: str | None) -> str:
    """Build the CLI-style hint for an unreadable user trust policy.

    Surfaced by `dcode mcp login` so a `config.toml` read failure is never
    swallowed just because a user-level config or an env-enabled server still
    loaded — and so the reason is not misattributed to an "untrusted project"
    when the real fix is repairing `config.toml`.

    Args:
        policy_error: The read-failure reason, or `None` when the policy loaded.

    Returns:
        A single-line user-facing string. Empty when `policy_error` is `None`.
    """
    if policy_error is None:
        return ""
    return _policy_error_message(policy_error)


def format_untrusted_project_notice(paths: tuple[Path, ...]) -> str:
    """Build the CLI-style hint string for skipped project server entries.

    Args:
        paths: Project configs with entries skipped during resolution.

    Returns:
        A single-line user-facing string. Empty when `paths` is empty.
    """
    if not paths:
        return ""
    skipped = ", ".join(str(path) for path in paths)
    return (
        "Skipping untrusted project MCP server entries "
        f"(not yet approved or disabled): {skipped}. "
        "Approve them by running `dcode` in this project, or "
        "pass --mcp-config <path> to use the file explicitly."
    )


def format_legacy_ignored_notice(names: tuple[str, ...]) -> str:
    """Build the CLI-style hint for servers dropped by the legacy-key removal.

    Mirrors the `resolve_and_load_mcp_tools` migration message so
    non-interactive `dcode mcp login` explains why a previously allowlisted
    server stopped loading instead of leaving it to vanish silently.

    Args:
        names: Server names found in a legacy `[mcp].enabled_project_servers`
            list, now ignored.

    Returns:
        A single-line user-facing string. Empty when `names` is empty.
    """
    if not names:
        return ""
    ignored = ", ".join(names)
    return (
        "[mcp].enabled_project_servers is no longer used; re-approve by "
        f"running `dcode` in this project to keep loading: {ignored}"
    )


def format_legacy_env_ignored_notice(legacy_env_ignored: bool) -> str:
    """Build the CLI-style hint for the renamed, now-ignored env var.

    Mirrors `format_legacy_ignored_notice` for the env surface so a user who
    still exports `DEEPAGENTS_CODE_ENABLED_PROJECT_MCP_SERVERS` learns it was
    renamed instead of its servers silently ceasing to pre-approve.

    Args:
        legacy_env_ignored: Whether the removed env var is set.

    Returns:
        A single-line user-facing string. Empty when the flag is `False`.
    """
    if not legacy_env_ignored:
        return ""
    return (
        "DEEPAGENTS_CODE_ENABLED_PROJECT_MCP_SERVERS is no longer used; it was "
        "renamed to DEEPAGENTS_CODE_DANGEROUSLY_ENABLE_PROJECT_MCP_SERVERS"
    )


__all__ = [
    "ConfigErrorKind",
    "ConfigResolution",
    "ConfigResolutionError",
    "ServerSelection",
    "format_legacy_env_ignored_notice",
    "format_legacy_ignored_notice",
    "format_policy_error_notice",
    "format_untrusted_project_notice",
    "resolve_mcp_config",
    "select_server",
]
