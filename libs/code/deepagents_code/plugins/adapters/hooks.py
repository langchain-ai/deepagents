"""Adapter from plugin hook declarations to Hooks v2 configuration sources."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from deepagents_code.hooks.loading import HooksSource, read_hooks_json
from deepagents_code.hooks.models.domain import HookDiagnostic, HookEvent
from deepagents_code.plugins.manifest import find_manifest_path
from deepagents_code.plugins.substitution import plugin_environment

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.plugins.models import JsonValue, PluginInstance

logger = logging.getLogger(__name__)

_KNOWN_EVENTS: Final = frozenset(event.value for event in HookEvent)

PluginHooksDocument = tuple[HooksSource, "JsonValue"]
"""One plugin hooks document with the provenance Hooks v2 loading needs.

The document is handed over undecided so that Hooks v2 validates it like any
other hooks document, and a malformed one produces a diagnostic instead of
disappearing.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class PluginHookSources:
    """Plugin hook documents plus the diagnostics collecting them produced."""

    documents: tuple[PluginHooksDocument, ...] = ()
    diagnostics: tuple[HookDiagnostic, ...] = ()


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
    """Collect a plugin's hook documents with the path each was declared at.

    Inline manifest hooks follow file documents in declaration order.

    Args:
        plugin: Plugin whose declarations should be read.

    Returns:
        Decoded documents keyed by the path used to locate diagnostics, and the
        read diagnostics for the documents that could not be decoded.
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


def _sources_for_plugin(
    plugin: PluginInstance, *, project_dir: Path | None
) -> PluginHookSources:
    """Build one plugin's hook sources.

    Args:
        plugin: Plugin to read.
        project_dir: Project directory for `${CLAUDE_PROJECT_DIR}` substitution.

    Returns:
        The plugin's sourced documents and its read diagnostics.
    """
    documents, diagnostics = _plugin_documents(plugin)
    if not documents:
        return PluginHookSources(diagnostics=tuple(diagnostics))
    # Create the writable data dir only for plugins that actually contribute
    # hooks, so discovery stays side-effect free for everyone else.
    try:
        plugin.data_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        diagnostics.append(
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
        (HooksSource(location=str(path), plugin_id=plugin.plugin_id, env=env), document)
        for path, document in documents
    )
    return PluginHookSources(documents=sources, diagnostics=tuple(diagnostics))


def discover_plugin_hook_sources(
    *,
    project_dir: Path | None = None,
    plugins: tuple[PluginInstance, ...] | None = None,
) -> PluginHookSources:
    """Build hook sources from enabled or already-discovered plugins.

    Shell-form commands resolve path variables from their environment; direct-exec
    `argv` values are resolved when the immutable hook snapshot is compiled.

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
        # `_diagnostic` logs the traceback. Discovery failure must not take user
        # and project hooks down with plugin hooks.
        except Exception as exc:  # noqa: BLE001
            return PluginHookSources(
                diagnostics=(
                    _diagnostic(
                        f"Could not discover plugin hooks: {exc}", exc_info=True
                    ),
                )
            )
        plugins = result.plugins
        diagnostics.extend(
            _diagnostic(f"Plugin discovery warning: {warning}")
            for warning in result.warnings
        )
    documents: list[PluginHooksDocument] = []
    for plugin in plugins:
        try:
            sources = _sources_for_plugin(plugin, project_dir=project_dir)
        # `_diagnostic` logs the traceback. The catch is deliberately broader
        # than the sibling MCP and skills adapters because the isolation
        # guarantee above only holds if an unforeseen failure in one plugin
        # cannot withhold every other plugin's hooks.
        except Exception as exc:  # noqa: BLE001
            diagnostics.append(
                _diagnostic(
                    f"Could not load hooks for plugin {plugin.plugin_id}: {exc}",
                    field=str(plugin.root),
                    exc_info=True,
                )
            )
            continue
        documents.extend(sources.documents)
        diagnostics.extend(sources.diagnostics)
    return PluginHookSources(documents=tuple(documents), diagnostics=tuple(diagnostics))


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
