"""Adapter from plugin hook declarations to Hooks v2 configuration sources."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deepagents_code.hooks.loading import HooksSource, read_hooks_json
from deepagents_code.hooks.models.domain import HookDiagnostic, HookEvent
from deepagents_code.plugins.manifest import find_manifest_path
from deepagents_code.plugins.substitution import plugin_environment, substitute_json

if TYPE_CHECKING:
    from pathlib import Path

    from deepagents_code.plugins.models import JsonValue, PluginInstance

logger = logging.getLogger(__name__)

PluginHooksDocument = tuple[HooksSource, "JsonValue"]
"""One plugin hooks document with the provenance Hooks v2 loading needs.

The document is handed over undecided: Hooks v2 validates it like any other
hooks document, so a plugin shipping a non-object or otherwise malformed
document produces a diagnostic instead of disappearing.
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

    Inline manifest hooks are loaded after file documents, matching how MCP
    servers let the manifest win over `.mcp.json`.

    Args:
        plugin: Plugin whose declarations should be read.

    Returns:
        Decoded documents keyed by the path used to locate diagnostics, and the
        read diagnostics for the documents that could not be decoded.
    """
    documents: list[tuple[Path, JsonValue]] = []
    diagnostics: list[HookDiagnostic] = []
    for path in plugin.inventory.hook_files:
        document, read_diagnostics = read_hooks_json(path)
        diagnostics.extend(read_diagnostics)
        if document is not None:
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
        (
            HooksSource(
                location=str(path),
                plugin_id=plugin.plugin_id,
                env=env,
            ),
            substitute_json(
                document,
                plugin_root=plugin.root,
                plugin_data=plugin.data_dir,
                project_dir=project_dir,
            ),
        )
        for path, document in documents
    )
    return PluginHookSources(documents=sources, diagnostics=tuple(diagnostics))


def plugin_hook_sources(
    plugins: tuple[PluginInstance, ...], *, project_dir: Path | None = None
) -> PluginHookSources:
    """Build Hooks v2 configuration sources for enabled plugins.

    Plugin path variables are substituted here rather than at execution time, so
    `${CLAUDE_PLUGIN_ROOT}` resolves identically for shell and `argv` handlers and
    the result participates in the configuration snapshot hash. The same variables
    are also exported to each handler's environment. A shell-form `command`
    receives the path literally, so a plugin whose install path may contain
    spaces has to quote the variable.

    One failing plugin never withholds another's hooks: each is read in
    isolation and reports its own diagnostics.

    Args:
        plugins: Enabled plugin instances.
        project_dir: Project directory for `${CLAUDE_PROJECT_DIR}` substitution.

    Returns:
        Sourced hook documents in plugin declaration order, with the
        diagnostics collecting them produced.
    """
    documents: list[PluginHooksDocument] = []
    diagnostics: list[HookDiagnostic] = []
    for plugin in plugins:
        try:
            result = _sources_for_plugin(plugin, project_dir=project_dir)
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
        documents.extend(result.documents)
        diagnostics.extend(result.diagnostics)
    return PluginHookSources(documents=tuple(documents), diagnostics=tuple(diagnostics))


def discover_plugin_hook_sources(
    *, project_dir: Path | None = None
) -> PluginHookSources:
    """Discover enabled plugins and build their hook configuration sources.

    Args:
        project_dir: Project directory for variable substitution.

    Returns:
        Sourced hook documents, or only a diagnostic when discovery fails.
    """
    try:
        from deepagents_code.plugins import discover_plugins

        result = discover_plugins()
    # `_diagnostic` logs the traceback. Discovery failing must degrade to "no
    # plugin hooks" rather than propagate and take the user and project hooks
    # down with it.
    except Exception as exc:  # noqa: BLE001
        return PluginHookSources(
            diagnostics=(
                _diagnostic(f"Could not discover plugin hooks: {exc}", exc_info=True),
            )
        )
    if result.warnings:
        logger.warning(
            "Plugin discovery warnings while loading hooks: %s", result.warnings
        )
    return plugin_hook_sources(result.plugins, project_dir=project_dir)


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
        events.extend(
            name for name in hooks if isinstance(name, str) and _is_known_event(name)
        )
    return tuple(dict.fromkeys(events))


def _is_known_event(name: str) -> bool:
    try:
        HookEvent(name)
    except ValueError:
        return False
    return True
