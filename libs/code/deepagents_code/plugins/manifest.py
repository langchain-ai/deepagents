"""Plugin manifest parsing for plugins."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path, PureWindowsPath
from typing import Any

from deepagents_code.plugins.models import ComponentInventory, PluginManifest

logger = logging.getLogger(__name__)

_MANIFEST_RELATIVE_PATHS = (
    Path(".claude-plugin") / "plugin.json",
    Path(".codex-plugin") / "plugin.json",
)
_PATH_COMPONENT_FIELDS = {
    "skills",
    "commands",
    "agents",
    "hooks",
    "mcpServers",
    "lspServers",
    "outputStyles",
}
_UNSUPPORTED_FIELDS = {
    "lspServers",
    "outputStyles",
    "settings",
    "userConfig",
    "channels",
    "themes",
    "monitors",
    "bin",
    "apps",
    "interface",
}
_EXPERIMENTAL_UNSUPPORTED = {"themes", "monitors"}
_NAME_RE = re.compile(r"^[^\s]+$")
_DEP_RE = re.compile(
    r"^[a-z0-9][-a-z0-9._]*(?:@[a-z0-9][-a-z0-9._]*)?$",
    re.IGNORECASE,
)


class PluginManifestError(ValueError):
    """Raised when a plugin manifest is malformed enough to skip the plugin."""


def find_manifest_path(root: Path) -> Path | None:
    """Return the first supported manifest path under `root`, if present.

    Args:
        root: Plugin root directory.

    Returns:
        Manifest path or `None`.
    """
    for rel in _MANIFEST_RELATIVE_PATHS:
        path = root / rel
        try:
            if path.is_file():
                return path
        except OSError:
            logger.warning("Could not inspect plugin manifest path %s", path)
    return None


def _validate_name(name: object, *, fallback: str | None = None) -> str:
    if isinstance(name, str) and name and _NAME_RE.match(name):
        return name
    if fallback and _NAME_RE.match(fallback):
        return fallback
    msg = f"Invalid plugin name: {name!r}"
    raise PluginManifestError(msg)


def _is_windows_absolute(path: str) -> bool:
    return bool(PureWindowsPath(path).drive or PureWindowsPath(path).root)


def _resolve_component_path(
    raw: str, root: Path, field: str, warnings: list[str]
) -> Path | None:
    if not raw.startswith("./"):
        warnings.append(
            f"ignoring {field}: path must start with './' relative to plugin root"
        )
        return None
    relative = raw[2:]
    if not relative:
        warnings.append(f"ignoring {field}: path must not be './'")
        return None
    path = Path(relative)
    if any(part == ".." for part in path.parts):
        warnings.append(f"ignoring {field}: path must not contain '..'")
        return None
    if path.is_absolute() or _is_windows_absolute(relative):
        warnings.append(f"ignoring {field}: path must stay within the plugin root")
        return None
    try:
        root_resolved = root.resolve()
        resolved = (root / path).resolve()
    except OSError as exc:
        warnings.append(f"ignoring {field}: could not resolve {raw!r}: {exc}")
        return None
    if not resolved.is_relative_to(root_resolved):
        warnings.append(f"ignoring {field}: path escapes plugin root")
        return None
    return resolved


def _path_list(
    value: object, root: Path, field: str, warnings: list[str]
) -> tuple[Path, ...]:
    raw_paths: list[str]
    if isinstance(value, str):
        raw_paths = [value]
    elif isinstance(value, list):
        raw_paths = [item for item in value if isinstance(item, str)]
        warnings.extend(
            f"ignoring {field}: expected path string, got {type(item).__name__}"
            for item in value
            if not isinstance(item, str)
        )
    else:
        warnings.append(f"ignoring {field}: expected path string or list of strings")
        return ()
    paths: list[Path] = []
    for raw in raw_paths:
        resolved = _resolve_component_path(raw, root, field, warnings)
        if resolved is not None:
            paths.append(resolved)
    return tuple(paths)


def _string_key_map(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return {key: value for key, value in raw.items() if isinstance(key, str)}


def _normalize_dependencies(raw: object, warnings: list[str]) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        warnings.append("ignoring dependencies: expected list")
        return ()
    deps: list[str] = []
    for item in raw:
        dep: str | None = None
        if isinstance(item, str):
            dep = re.sub(r"@\^[^@]*$", "", item)
        elif isinstance(item, dict):
            name = item.get("name")
            marketplace = item.get("marketplace")
            if isinstance(name, str):
                dep = f"{name}@{marketplace}" if isinstance(marketplace, str) else name
        if dep and _DEP_RE.match(dep):
            deps.append(dep)
        else:
            warnings.append(f"ignoring dependency: invalid reference {item!r}")
    return tuple(deps)


def _inline_hooks(value: object) -> tuple[dict[str, Any], ...]:
    if isinstance(value, dict):
        return (_string_key_map(value),)
    if isinstance(value, list):
        return tuple(_string_key_map(item) for item in value if isinstance(item, dict))
    return ()


def _inline_mcp(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return _string_key_map(value)
    if isinstance(value, list):
        merged: dict[str, Any] = {}
        for item in value:
            if isinstance(item, dict):
                merged.update(_string_key_map(item))
        return merged
    return {}


def _inline_commands(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for name, metadata in value.items():
        if isinstance(name, str) and isinstance(metadata, dict):
            result[name] = _string_key_map(metadata)
    return result


def load_manifest(
    root: Path, *, fallback_name: str | None = None
) -> tuple[PluginManifest | None, Path | None, tuple[str, ...]]:
    """Load a Claude/Codex plugin manifest.

    Args:
        root: Plugin root directory.
        fallback_name: Name to use only when deriving a manifest-less plugin.

    Returns:
        `(manifest, manifest_path, warnings)`.

    Raises:
        PluginManifestError: If the manifest exists but is invalid.
    """
    manifest_path = find_manifest_path(root)
    if manifest_path is None:
        return None, None, ()
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON syntax in {manifest_path}: {exc}"
        raise PluginManifestError(msg) from exc
    except OSError as exc:
        msg = f"Could not read plugin manifest {manifest_path}: {exc}"
        raise PluginManifestError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"Plugin manifest {manifest_path} must be a JSON object"
        raise PluginManifestError(msg)

    warnings: list[str] = []
    name = _validate_name(raw.get("name"), fallback=fallback_name)
    component_paths: dict[str, tuple[Path, ...]] = {}
    for field in _PATH_COMPONENT_FIELDS:
        value = raw.get(field)
        if value is None:
            continue
        if field in {"hooks", "mcpServers", "commands"} and isinstance(value, dict):
            continue
        paths = _path_list(value, root, field, warnings)
        if paths:
            component_paths[field] = paths

    unsupported: list[str] = [field for field in _UNSUPPORTED_FIELDS if field in raw]
    experimental = raw.get("experimental")
    if isinstance(experimental, dict):
        for field in _EXPERIMENTAL_UNSUPPORTED:
            if field in experimental:
                unsupported.append(f"experimental.{field}")
                value = experimental[field]
                if isinstance(value, (str, list)):
                    paths = _path_list(value, root, f"experimental.{field}", warnings)
                    if paths:
                        component_paths[f"experimental.{field}"] = paths

    manifest = PluginManifest(
        name=name,
        version=raw.get("version") if isinstance(raw.get("version"), str) else None,
        description=raw.get("description")
        if isinstance(raw.get("description"), str)
        else None,
        author=raw.get("author")
        if isinstance(raw.get("author"), (str, dict))
        else None,
        display_name=raw.get("displayName")
        if isinstance(raw.get("displayName"), str)
        else None,
        default_enabled=raw.get("defaultEnabled")
        if isinstance(raw.get("defaultEnabled"), bool)
        else True,
        dependencies=_normalize_dependencies(raw.get("dependencies"), warnings),
        component_paths=component_paths,
        inline_hooks=_inline_hooks(raw.get("hooks")),
        inline_mcp=_inline_mcp(raw.get("mcpServers")),
        inline_commands=_inline_commands(raw.get("commands")),
        raw=raw,
    )
    warnings.extend(
        f"unsupported component detected: {field}" for field in sorted(set(unsupported))
    )
    return manifest, manifest_path, tuple(warnings)


def _existing(path: Path, root: Path) -> tuple[Path, ...]:
    try:
        if not path.exists():
            return ()
        resolved = path.resolve()
        if not resolved.is_relative_to(root.resolve()):
            logger.warning("Ignoring plugin component outside plugin root: %s", path)
            return ()
    except OSError:
        logger.warning("Could not inspect plugin component path %s", path)
        return ()
    else:
        return (resolved,)


def build_inventory(
    root: Path, manifest: PluginManifest | None, manifest_warnings: tuple[str, ...] = ()
) -> ComponentInventory:
    """Build component inventory for a plugin.

    Args:
        root: Plugin root directory.
        manifest: Parsed manifest or `None`.
        manifest_warnings: Warnings emitted during manifest parsing.

    Returns:
        Component inventory.
    """
    root = root.resolve()
    warnings = list(manifest_warnings)
    unsupported: list[tuple[str, str]] = []
    metadata_paths = manifest.component_paths if manifest else {}

    default_skills = _existing(root / "skills", root)
    root_skill = (
        ()
        if default_skills or (manifest and "skills" in manifest.component_paths)
        else _existing(root / "SKILL.md", root)
    )
    # Default skills remain additive when custom skill roots are configured.
    skills = (*default_skills, *metadata_paths.get("skills", ()), *root_skill)

    commands = (
        metadata_paths.get("commands", ())
        if manifest and "commands" in metadata_paths
        else _existing(root / "commands", root)
    )
    agents = (
        metadata_paths.get("agents", ())
        if manifest and "agents" in metadata_paths
        else _existing(root / "agents", root)
    )
    hooks_files = (
        *_existing(root / "hooks" / "hooks.json", root),
        *metadata_paths.get("hooks", ()),
    )
    mcp_files = (
        *_existing(root / ".mcp.json", root),
        *metadata_paths.get("mcpServers", ()),
    )

    for field in (
        "lspServers",
        "outputStyles",
        "experimental.themes",
        "experimental.monitors",
    ):
        unsupported.extend((field, str(path)) for path in metadata_paths.get(field, ()))
    for child in ("lspServers", "output-styles", "themes", "monitors", "bin"):
        candidate = root / child
        try:
            if candidate.exists():
                unsupported.append((child, str(candidate)))
        except OSError:
            logger.warning(
                "Could not inspect unsupported plugin component %s", candidate
            )
    metadata_dir = root / ".claude-plugin"
    for misplaced in ("skills", "commands", "agents", "hooks"):
        candidate = metadata_dir / misplaced
        try:
            if candidate.exists():
                warnings.append(
                    f"{candidate} is inside .claude-plugin; "
                    f"move {misplaced} to the plugin root"
                )
        except OSError:
            logger.warning("Could not inspect misplaced plugin component %s", candidate)

    if manifest:
        unsupported.extend(
            (field, field)
            for field in ("settings", "userConfig", "channels")
            if field in manifest.raw
        )
        experimental = manifest.raw.get("experimental")
        if isinstance(experimental, dict):
            unsupported.extend(
                (f"experimental.{field}", str(experimental[field]))
                for field in _EXPERIMENTAL_UNSUPPORTED
                if field in experimental
            )

    return ComponentInventory(
        skills=tuple(dict.fromkeys(skills)),
        commands=tuple(dict.fromkeys(commands)),
        agents=tuple(dict.fromkeys(agents)),
        hooks_files=tuple(dict.fromkeys(hooks_files)),
        mcp_files=tuple(dict.fromkeys(mcp_files)),
        unsupported=tuple(dict.fromkeys(unsupported)),
        warnings=tuple(warnings),
    )
