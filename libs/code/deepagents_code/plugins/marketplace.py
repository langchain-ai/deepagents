"""Marketplace parsing for plugins."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess  # noqa: S404  # Git is invoked with fixed argv and no shell.
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse

from deepagents_code.plugins.manifest import _resolve_component_path, _validate_name
from deepagents_code.plugins.models import (
    JsonObject,
    JsonValue,
    LocalMarketplaceSource,
    MarketplacePluginEntry,
    MarketplaceSource,
    PluginMarketplace,
    RepositoryMarketplaceSource,
    UrlMarketplaceSource,
)
from deepagents_code.plugins.store import marketplace_cache_dir, sanitize_plugin_id

logger = logging.getLogger(__name__)

_MARKETPLACE_RELATIVE_PATHS = (
    Path(".claude-plugin") / "marketplace.json",
    Path(".agents") / "plugins" / "marketplace.json",
    Path(".agents") / "plugins" / "api_marketplace.json",
)
_MARKETPLACE_ENTRY_METADATA_FIELDS = {"author"}
_SSH_GIT_RE = re.compile(r"^([A-Za-z0-9._-]+@[^:]+:.+?(?:\.git)?)(?:#(.+))?$")
_GITHUB_REPO_RE = re.compile(r"^[^/\s]+/[^/\s]+$")
_GIT_TIMEOUT_SECONDS = 120
_GITHUB_REPO_PART_COUNT = 2
_SENSITIVE_QUERY_TERMS = (
    "credential",
    "key",
    "password",
    "secret",
    "signature",
    "token",
)
_SENSITIVE_PATH_KEY_RE = re.compile(
    r"^(?:access[-_.]?token|api[-_.]?key|credential|key|password|secret|signature|token)s?$",
    re.IGNORECASE,
)


class MarketplaceError(ValueError):
    """Raised when a marketplace cannot be loaded."""


def _redact_url_credentials(value: str) -> str:
    try:
        parsed = urlparse(value)
    except ValueError:
        return value
    if parsed.scheme not in {"http", "https"}:
        return value
    netloc = parsed.netloc
    if "@" in netloc:
        try:
            host = parsed.hostname or ""
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
        except ValueError:
            return value
        netloc = f"***@{host}"
    query = urlencode(
        [
            (
                key,
                "***"
                if any(term in key.lower() for term in _SENSITIVE_QUERY_TERMS)
                else item,
            )
            for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        ]
    )
    path_parts = parsed.path.split("/")
    redact_next = False
    for index, part in enumerate(path_parts):
        if redact_next and part:
            path_parts[index] = "***"
            redact_next = False
        elif part:
            redact_next = _SENSITIVE_PATH_KEY_RE.fullmatch(unquote(part)) is not None
    path = "/".join(path_parts)
    return urlunparse(parsed._replace(netloc=netloc, path=path, query=query))


def redact_marketplace_source(value: str) -> str:
    """Return a marketplace source safe for persistence and display."""
    return _redact_url_credentials(value)


def parse_marketplace_source(raw: str) -> MarketplaceSource:
    """Parse a user-provided marketplace source.

    Args:
        raw: GitHub shorthand, Git URL, marketplace JSON URL, file, or directory.

    Returns:
        Parsed marketplace source.

    Raises:
        MarketplaceError: If the source string is empty or unsupported.
    """
    value = raw.strip()
    if not value:
        msg = "Please enter a marketplace source"
        raise MarketplaceError(msg)

    ssh_match = _SSH_GIT_RE.match(value)
    if ssh_match:
        return RepositoryMarketplaceSource(
            source_type="git", value=ssh_match.group(1), ref=ssh_match.group(2)
        )

    if value.startswith(("http://", "https://")):
        url, _, ref = value.partition("#")
        parsed = urlparse(url)
        if parsed.username is not None or parsed.password is not None:
            msg = "Marketplace URLs must not contain embedded credentials"
            raise MarketplaceError(msg)
        path = parsed.path
        if path.endswith(".git") or "/_git/" in path:
            return RepositoryMarketplaceSource(
                source_type="git", value=url, ref=ref or None
            )
        if parsed.hostname in {"github.com", "www.github.com"}:
            parts = [part for part in path.split("/") if part]
            if len(parts) >= _GITHUB_REPO_PART_COUNT:
                repo_path = "/".join(parts[:_GITHUB_REPO_PART_COUNT])
                git_url = f"https://github.com/{repo_path}.git"
                return RepositoryMarketplaceSource(
                    source_type="git", value=git_url, ref=ref or None
                )
        return UrlMarketplaceSource(source_type="url", value=url)

    if value.startswith(("./", "../", "/", "~")):
        return _marketplace_source_from_path(value)

    # Bare relative paths such as `marketplace` (no ./ prefix) are accepted when
    # they exist on disk, before GitHub-shorthand parsing.
    candidate = Path(value).expanduser()
    if candidate.exists():
        return _marketplace_source_from_path(value)

    repo, sep, ref = value.replace("#", "@", 1).partition("@")
    if (
        "/" in value
        and ":" not in value
        and not value.startswith("@")
        and _GITHUB_REPO_RE.match(repo)
    ):
        return RepositoryMarketplaceSource(
            source_type="github", value=repo, ref=ref if sep else None
        )

    msg = "Invalid marketplace source format. Try: owner/repo, https://..., or ./path"
    raise MarketplaceError(msg)


def _marketplace_source_from_path(value: str) -> MarketplaceSource:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        msg = f"Path does not exist: {path}"
        raise MarketplaceError(msg)
    if path.is_file():
        if path.suffix != ".json":
            msg = f"File path must point to a .json marketplace file: {path}"
            raise MarketplaceError(msg)
        return LocalMarketplaceSource(source_type="file", value=str(path))
    if path.is_dir():
        return LocalMarketplaceSource(source_type="directory", value=str(path))
    msg = f"Path is neither a file nor a directory: {path}"
    raise MarketplaceError(msg)


def _root_for_marketplace_file(path: Path) -> Path:
    if path.name == "marketplace.json" and path.parent.name == ".claude-plugin":
        return path.parent.parent
    return path.parent


def _load_marketplace_file(path: Path) -> PluginMarketplace:
    root = _root_for_marketplace_file(path.expanduser().resolve())
    return _load_marketplace_from_path(root, path.expanduser().resolve())


def _run_git(args: list[str]) -> None:
    git_path = shutil.which("git")
    if git_path is None:
        msg = "Git is required to add repository-backed plugin marketplaces"
        raise MarketplaceError(msg)
    env = {
        **os.environ,
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "",
    }
    try:
        result = subprocess.run(  # noqa: S603  # Fixed git executable, no shell.
            [git_path, *args],
            check=False,
            capture_output=True,
            env=env,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        msg = f"Failed to run git: {exc}"
        raise MarketplaceError(msg) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown git error"
        msg = f"Git command failed: {_redact_url_credentials(detail)}"
        raise MarketplaceError(msg)


def _clone_repository(
    source: RepositoryMarketplaceSource,
    git_url: str,
    *,
    cache_key: str,
    require_marketplace: bool,
) -> Path:
    cache_path = marketplace_cache_dir() / sanitize_plugin_id(cache_key)
    temp_path = Path(
        tempfile.mkdtemp(prefix=f".{cache_path.name}.", dir=cache_path.parent)
    )
    args = ["clone", "--depth", "1", "--recurse-submodules", "--shallow-submodules"]
    if source.ref:
        args.extend(["--branch", source.ref])
    args.extend([git_url, str(temp_path)])
    try:
        _run_git(args)
        if require_marketplace:
            load_marketplace(temp_path)
        backup_path = cache_path.with_name(f".{cache_path.name}.backup")
        if backup_path.exists():
            shutil.rmtree(backup_path, ignore_errors=True)
        if cache_path.exists():
            cache_path.replace(backup_path)
        try:
            temp_path.replace(cache_path)
        except OSError:
            if backup_path.exists() and not cache_path.exists():
                backup_path.replace(cache_path)
            raise
        if backup_path.exists():
            shutil.rmtree(backup_path, ignore_errors=True)
    except Exception:
        shutil.rmtree(temp_path, ignore_errors=True)
        raise
    return cache_path


def _clone_marketplace(source: RepositoryMarketplaceSource, git_url: str) -> Path:
    return _clone_repository(
        source,
        git_url,
        cache_key=f"marketplace-{source.source_type}-{source.value}",
        require_marketplace=True,
    )


def _download_marketplace(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        msg = f"Marketplace URL must use http or https: {_redact_url_credentials(url)}"
        raise MarketplaceError(msg)
    cache_path = marketplace_cache_dir() / f"{sanitize_plugin_id(url)}.json"
    request = urllib.request.Request(  # noqa: S310  # Scheme is restricted above.
        url, headers={"User-Agent": "dcode-plugin-manager"}
    )
    try:
        with urllib.request.urlopen(  # noqa: S310  # Scheme is restricted above.
            request, timeout=10
        ) as response:
            data = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        msg = (
            f"Failed to download marketplace from {_redact_url_credentials(url)}: {exc}"
        )
        raise MarketplaceError(msg) from exc
    if not isinstance(data, dict):
        msg = (
            f"Marketplace URL must return a JSON object: {_redact_url_credentials(url)}"
        )
        raise MarketplaceError(msg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return cache_path


def materialize_marketplace_source(
    source: MarketplaceSource,
) -> tuple[PluginMarketplace, Path]:
    """Load a marketplace source and return its local install location.

    Args:
        source: Parsed marketplace source.

    Returns:
        Parsed marketplace and its local install location.

    Raises:
        MarketplaceError: If loading, cloning, downloading, or parsing fails.
    """
    if source.source_type == "directory":
        root = Path(source.value).expanduser().resolve()
        return load_marketplace(root), root
    if source.source_type == "file":
        path = Path(source.value).expanduser().resolve()
        return _load_marketplace_file(path), path
    if source.source_type == "url":
        path = _download_marketplace(source.value)
        marketplace = _load_marketplace_file(path)
        _reject_url_marketplace_with_local_plugins(marketplace, source.value)
        return marketplace, path
    if source.source_type == "github":
        if not isinstance(source, RepositoryMarketplaceSource):
            msg = "GitHub marketplace source is missing repository metadata"
            raise MarketplaceError(msg)
        root = _clone_marketplace(source, f"https://github.com/{source.value}.git")
        return load_marketplace(root), root
    if source.source_type == "git":
        if not isinstance(source, RepositoryMarketplaceSource):
            msg = "Git marketplace source is missing repository metadata"
            raise MarketplaceError(msg)
        root = _clone_marketplace(source, source.value)
        return load_marketplace(root), root
    msg = f"Unsupported marketplace source type: {source.source_type}"
    raise MarketplaceError(msg)


def _reject_url_marketplace_with_local_plugins(
    marketplace: PluginMarketplace, url: str
) -> None:
    """Reject URL marketplaces whose plugins need a sibling filesystem tree.

    Direct marketplace JSON URLs only cache the catalog file. Relative plugin
    sources such as `./plugins/foo` cannot be resolved from that cache alone.

    Raises:
        MarketplaceError: If any plugin entry uses a local relative source.
    """
    local_plugins = [
        plugin.name
        for plugin in marketplace.plugins
        if _source_path(plugin.source) is not None
    ]
    if not local_plugins:
        unsupported = [
            plugin.name
            for plugin in marketplace.plugins
            if _plugin_repository_source(plugin) is None
        ]
        if not unsupported:
            return
        names = ", ".join(sorted(unsupported))
        msg = (
            f"Marketplace URL {_redact_url_credentials(url)} contains plugins "
            f"with unsupported remote sources: [{names}]"
        )
        raise MarketplaceError(msg)
    names = ", ".join(sorted(local_plugins))
    msg = (
        f"Marketplace URL {_redact_url_credentials(url)} only downloads the "
        f"catalog JSON, but plugins [{names}] use local relative sources. "
        "Use a git repository or local directory for this marketplace."
    )
    raise MarketplaceError(msg)


def load_marketplace_location(path: Path) -> PluginMarketplace:
    """Load a marketplace from either a cached directory or JSON file.

    Args:
        path: Directory or marketplace JSON path.

    Returns:
        Parsed marketplace.
    """
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return _load_marketplace_file(resolved)
    return load_marketplace(resolved)


def find_marketplace_manifest(root: Path) -> Path | None:
    """Return a marketplace manifest path under `root`, if present."""
    for rel in _MARKETPLACE_RELATIVE_PATHS:
        path = root / rel
        try:
            if path.is_file():
                return path
        except OSError:
            logger.warning("Could not inspect marketplace manifest path %s", path)
    return None


def _json_value(value: object) -> JsonValue | None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        result: list[JsonValue] = []
        for item in value:
            converted = _json_value(item)
            if converted is not None or item is None:
                result.append(converted)
        return result
    if isinstance(value, dict):
        result_object: JsonObject = {}
        for key, item in value.items():
            if not isinstance(key, str):
                continue
            converted = _json_value(item)
            if converted is not None or item is None:
                result_object[key] = converted
        return result_object
    return None


def _json_object(value: object) -> JsonObject:
    converted = _json_value(value)
    return converted if isinstance(converted, dict) else {}


def _source_path(source: str | JsonObject) -> str | None:
    if isinstance(source, str):
        return source
    kind = source.get("source") if isinstance(source, dict) else None
    path = source.get("path")
    if kind == "local" and isinstance(path, str):
        return path
    return None


def _plugin_repository_source(
    plugin: MarketplacePluginEntry,
) -> tuple[RepositoryMarketplaceSource, str, str | None] | None:
    if not isinstance(plugin.source, dict):
        return None
    kind = plugin.source.get("source")
    ref = plugin.source.get("ref")
    ref_value = ref if isinstance(ref, str) else None
    subpath = plugin.source.get("path")
    subpath_value = subpath if isinstance(subpath, str) else None
    if kind == "github":
        repo = plugin.source.get("repo")
        if not isinstance(repo, str):
            return None
        parsed = parse_marketplace_source(f"{repo}#{ref_value}" if ref_value else repo)
        if not isinstance(parsed, RepositoryMarketplaceSource):
            return None
        return parsed, f"https://github.com/{parsed.value}.git", subpath_value
    if kind not in {"git-subdir", "url"}:
        return None
    raw_url = plugin.source.get("url")
    if not isinstance(raw_url, str):
        return None
    parsed = parse_marketplace_source(
        f"{raw_url}#{ref_value}" if ref_value else raw_url
    )
    if parsed.source_type == "github":
        git_url = f"https://github.com/{parsed.value}.git"
    elif parsed.source_type == "git":
        git_url = parsed.value
    else:
        return None
    if not isinstance(parsed, RepositoryMarketplaceSource):
        return None
    return parsed, git_url, subpath_value


def resolve_plugin_source(
    marketplace: PluginMarketplace, plugin: MarketplacePluginEntry
) -> Path | None:
    """Resolve or materialize a marketplace plugin entry to a plugin root.

    Args:
        marketplace: Marketplace containing the plugin.
        plugin: Plugin entry.

    Returns:
        Resolved plugin root, or `None` for unsupported sources.
    """
    raw = _source_path(plugin.source)
    if raw is not None:
        metadata_root = marketplace.metadata.get("pluginRoot")
        warnings: list[str] = []
        base = marketplace.root
        if isinstance(metadata_root, str) and raw.startswith("./"):
            base_path = _resolve_component_path(
                metadata_root, marketplace.root, "metadata.pluginRoot", warnings
            )
            if base_path is not None:
                base = base_path
        resolved = _resolve_component_path(
            raw, base, f"plugins.{plugin.name}.source", warnings
        )
        for warning in warnings:
            logger.warning("Marketplace %s: %s", marketplace.name, warning)
        return resolved

    repository = _plugin_repository_source(plugin)
    if repository is None:
        return None
    source, git_url, subpath = repository
    root = _clone_repository(
        source,
        git_url,
        cache_key=(
            f"plugin-source-{marketplace.name}-{plugin.name}-"
            f"{json.dumps(plugin.source, sort_keys=True)}"
        ),
        require_marketplace=False,
    )
    if subpath is None:
        return root
    warnings = []
    resolved = _resolve_component_path(
        subpath, root, f"plugins.{plugin.name}.source.path", warnings
    )
    for warning in warnings:
        logger.warning("Marketplace %s: %s", marketplace.name, warning)
    return resolved


def _parse_entry(entry: object) -> MarketplacePluginEntry | None:
    if not isinstance(entry, dict):
        logger.warning(
            "Skipping marketplace plugin entry: expected object, got %s",
            type(entry).__name__,
        )
        return None
    source_value = entry.get("source")
    if isinstance(source_value, str):
        source: str | JsonObject = source_value
    elif isinstance(source_value, dict):
        source = _json_object(source_value)
    else:
        logger.warning(
            "Skipping marketplace plugin %r: missing source", entry.get("name")
        )
        return None
    try:
        name = _validate_name(entry.get("name"))
    except ValueError as exc:
        logger.warning("Skipping marketplace plugin with invalid name: %s", exc)
        return None
    description_value = entry.get("description")
    manifest_fields = {
        key: value
        for key, value in _json_object(entry).items()
        if key in _MARKETPLACE_ENTRY_METADATA_FIELDS
    }
    return MarketplacePluginEntry(
        name=name,
        source=source,
        description=description_value if isinstance(description_value, str) else None,
        manifest_fields=manifest_fields,
    )


def _load_marketplace_from_path(root: Path, manifest_path: Path) -> PluginMarketplace:
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"Invalid JSON syntax in {manifest_path}: {exc}"
        raise MarketplaceError(msg) from exc
    except OSError as exc:
        msg = f"Could not read marketplace manifest {manifest_path}: {exc}"
        raise MarketplaceError(msg) from exc
    if not isinstance(raw, dict):
        msg = f"Marketplace manifest {manifest_path} must be a JSON object"
        raise MarketplaceError(msg)
    try:
        name = _validate_name(raw.get("name"))
    except ValueError as exc:
        raise MarketplaceError(str(exc)) from exc
    plugins_raw = raw.get("plugins")
    if not isinstance(plugins_raw, list):
        msg = f"Marketplace {name} must contain a plugins array"
        raise MarketplaceError(msg)
    plugins = tuple(
        plugin for entry in plugins_raw if (plugin := _parse_entry(entry)) is not None
    )
    owner = raw.get("owner") if isinstance(raw.get("owner"), dict) else None
    metadata = _json_object(raw.get("metadata"))
    return PluginMarketplace(
        name=name,
        root=root,
        manifest_path=manifest_path,
        owner=_json_object(owner) if owner is not None else None,
        metadata=metadata,
        plugins=plugins,
    )


def load_marketplace(root: Path) -> PluginMarketplace:
    """Load a marketplace manifest from a root directory.

    Args:
        root: Marketplace root directory.

    Returns:
        Parsed marketplace.

    Raises:
        MarketplaceError: If no marketplace manifest exists or it is invalid.
    """
    root = root.expanduser().resolve()
    manifest_path = find_marketplace_manifest(root)
    if manifest_path is None:
        msg = f"No marketplace manifest found under {root}"
        raise MarketplaceError(msg)
    return _load_marketplace_from_path(root, manifest_path)
