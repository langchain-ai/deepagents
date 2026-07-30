"""Adapter from plugin hook declarations to Hooks v2 configuration sources."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from deepagents_code.hooks.loading import HooksSource
from deepagents_code.plugins._json import json_object
from deepagents_code.plugins.substitution import plugin_environment, substitute_json

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.plugins.models import JsonObject, PluginInstance

logger = logging.getLogger(__name__)

PluginHooksDocument = tuple[HooksSource, "JsonObject"]
"""One plugin `hooks.json` document with the provenance Hooks v2 loading needs."""


def _load_hooks_document(path: Path) -> JsonObject:
    """Load a plugin hooks document.

    Returns:
        The decoded document, or an empty object when it cannot be read.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Skipping plugin hooks config %s: %s", path, exc)
        return {}
    return json_object(raw)


def _plugin_documents(plugin: PluginInstance) -> list[tuple[Path, JsonObject]]:
    """Collect a plugin's hook documents with the path each was declared at.

    Inline manifest hooks are loaded after file documents, matching how MCP
    servers let the manifest win over `.mcp.json`.

    Returns:
        Non-empty documents keyed by the path used to locate diagnostics.
    """
    documents = [
        (path, document)
        for path in plugin.inventory.hook_files
        if (document := _load_hooks_document(path))
    ]
    manifest = plugin.manifest
    if manifest and manifest.inline_hooks:
        manifest_path = plugin.root / ".claude-plugin" / "plugin.json"
        documents.append((manifest_path, manifest.inline_hooks))
    return documents


def plugin_hook_sources(
    plugins: tuple[PluginInstance, ...], *, project_dir: Path | None = None
) -> tuple[PluginHooksDocument, ...]:
    """Build Hooks v2 configuration sources for enabled plugins.

    Plugin path variables are substituted here rather than at execution time, so
    `${CLAUDE_PLUGIN_ROOT}` resolves identically for shell and `argv` handlers and
    the result participates in the configuration snapshot hash. The same variables
    are also exported to each handler's environment.

    Args:
        plugins: Enabled plugin instances.
        project_dir: Project directory for `${CLAUDE_PROJECT_DIR}` substitution.

    Returns:
        Sourced hook documents in plugin declaration order.
    """
    sources: list[PluginHooksDocument] = []
    for plugin in plugins:
        documents = _plugin_documents(plugin)
        if not documents:
            continue
        # Create the writable data dir only for plugins that actually contribute
        # hooks, so discovery stays side-effect free for everyone else.
        try:
            plugin.data_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            logger.warning(
                "Could not create plugin data dir for %s: %s",
                plugin.plugin_id,
                plugin.data_dir,
                exc_info=True,
            )
        env = plugin_environment(
            plugin_root=plugin.root,
            plugin_data=plugin.data_dir,
            project_dir=project_dir,
        )
        for path, document in documents:
            substituted = substitute_json(
                document,
                plugin_root=plugin.root,
                plugin_data=plugin.data_dir,
                project_dir=project_dir,
            )
            if not isinstance(substituted, dict):
                continue
            sources.append(
                (
                    HooksSource(
                        location=str(path),
                        origin=plugin.plugin_id,
                        env=env,
                    ),
                    substituted,
                )
            )
    return tuple(sources)


def discover_plugin_hook_sources(
    *, project_dir: Path | None = None
) -> tuple[PluginHooksDocument, ...]:
    """Discover enabled plugins and build their hook configuration sources.

    Args:
        project_dir: Project directory for variable substitution.

    Returns:
        Sourced hook documents, or an empty tuple when discovery fails.
    """
    try:
        from deepagents_code.plugins import discover_plugins

        result = discover_plugins()
    except (OSError, RuntimeError):
        logger.warning("Could not discover plugin hooks", exc_info=True)
        return ()
    if result.warnings:
        logger.warning(
            "Plugin discovery warnings while loading hooks: %s", result.warnings
        )
    return plugin_hook_sources(result.plugins, project_dir=project_dir)


def plugin_hook_event_names(plugin: PluginInstance) -> tuple[str, ...]:
    """List the hook events a plugin declares, for display before it loads.

    Args:
        plugin: Plugin whose declarations should be inspected.

    Returns:
        Declared event names in declaration order, deduplicated.
    """
    events: list[str] = []
    for _path, document in _plugin_documents(plugin):
        hooks = document.get("hooks")
        if not isinstance(hooks, dict):
            continue
        events.extend(name for name in hooks if isinstance(name, str))
    return tuple(dict.fromkeys(events))
