"""Adapter from plugin hook declarations to Hooks v2 configuration sources."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from deepagents_code.hooks.loading import PluginHooksSource, read_hooks_json
from deepagents_code.hooks.models.domain import HookDiagnostic, HookEvent
from deepagents_code.plugins.manifest import find_manifest_path
from deepagents_code.plugins.substitution import plugin_environment

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.plugins.models import JsonValue, PluginInstance

logger = logging.getLogger(__name__)

_KNOWN_EVENTS = frozenset(event.value for event in HookEvent)

PluginHooksDocument = tuple[PluginHooksSource, "JsonValue"]
PluginHookSources = tuple[tuple[PluginHooksDocument, ...], tuple[HookDiagnostic, ...]]


def _diagnostic(
    message: str, *, field: str | None = None, exc_info: bool = False
) -> HookDiagnostic:
    logger.warning(message, exc_info=exc_info)
    return HookDiagnostic(
        code="plugin_hooks_failed",
        severity="warning",
        message=message,
        field=field,
    )


def _plugin_documents(
    plugin: PluginInstance,
) -> tuple[list[tuple[Path, JsonValue]], list[HookDiagnostic]]:
    """Collect decoded file and inline hook documents in declaration order.

    Returns:
        Documents with diagnostic locations, plus read diagnostics.
    """
    documents: list[tuple[Path, JsonValue]] = []
    diagnostics: list[HookDiagnostic] = []
    for path in plugin.inventory.hook_files:
        decoded, document, read_diagnostics = read_hooks_json(path)
        diagnostics.extend(read_diagnostics)
        if decoded:
            documents.append((path, document))
    manifest = plugin.manifest
    if manifest and manifest.inline_hooks:
        manifest_path = find_manifest_path(plugin.root) or plugin.root
        documents.append((manifest_path, manifest.inline_hooks))
    return documents, diagnostics


def discover_plugin_hook_sources(
    *,
    project_dir: Path | None = None,
    plugins: tuple[PluginInstance, ...] | None = None,
) -> PluginHookSources:
    """Build hook sources from enabled or already-discovered plugins.

    Args:
        project_dir: Project directory exposed as `${CLAUDE_PROJECT_DIR}`.
        plugins: Already-discovered plugins, or `None` to discover them here.

    Returns:
        Sourced hook documents and collection diagnostics.
    """
    diagnostics: list[HookDiagnostic] = []
    if plugins is None:
        try:
            from deepagents_code.plugins import discover_plugins

            result = discover_plugins()
        # Discovery failure must not take user and project hooks down with it.
        except Exception as exc:  # noqa: BLE001
            return (), (
                _diagnostic(f"Could not discover plugin hooks: {exc}", exc_info=True),
            )
        plugins = result.plugins
        diagnostics.extend(
            _diagnostic(f"Plugin discovery warning: {warning}")
            for warning in result.warnings
        )
    documents: list[PluginHooksDocument] = []
    for plugin in plugins:
        try:
            plugin_documents, plugin_diagnostics = _plugin_documents(plugin)
            if not plugin_documents:
                diagnostics.extend(plugin_diagnostics)
                continue
            try:
                plugin.data_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                plugin_diagnostics.append(
                    _diagnostic(
                        f"Could not create the data directory for plugin "
                        f"{plugin.plugin_id}: {exc}",
                        field=str(plugin.data_dir),
                    )
                )
            env = plugin_environment(
                plugin_root=plugin.root,
                plugin_data=plugin.data_dir,
                project_dir=project_dir,
            )
            sources = tuple(
                (
                    PluginHooksSource(
                        location=str(path), plugin_id=plugin.plugin_id, env=env
                    ),
                    document,
                )
                for path, document in plugin_documents
            )
        # A broken plugin must not withhold every other plugin's hooks.
        except Exception as exc:  # noqa: BLE001
            diagnostics.append(
                _diagnostic(
                    f"Could not load hooks for plugin {plugin.plugin_id}: {exc}",
                    field=str(plugin.root),
                    exc_info=True,
                )
            )
            continue
        documents.extend(sources)
        diagnostics.extend(plugin_diagnostics)
    return tuple(documents), tuple(diagnostics)


def plugin_hook_event_names(plugin: PluginInstance) -> tuple[str, ...]:
    """List the hook events a plugin declares, for display before it loads.

    Only events Hooks v2 recognizes are returned, so the plugin manager never
    advertises a hook that the loader will later reject.

    Args:
        plugin: Plugin whose declarations should be inspected.

    Returns:
        Declared event names in declaration order, deduplicated.
    """
    events: list[str] = []
    documents, _diagnostics = _plugin_documents(plugin)
    for _path, document in documents:
        if not isinstance(document, dict):
            continue
        hooks = document.get("hooks")
        if not isinstance(hooks, dict):
            continue
        events.extend(name for name in hooks if name in _KNOWN_EVENTS)
    return tuple(dict.fromkeys(events))
