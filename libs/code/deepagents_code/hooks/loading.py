"""Validated Hooks v2 configuration loading, merging, and hashing.

Precedence (highest first, earlier in reduction order):

1. Project: `{project_root}/.deepagents/hooks.json`
2. User: `~/.deepagents/hooks.json` (or `config_dir/hooks.json` in tests)
3. Plugin: `hooks.json` documents contributed by enabled plugins

Sources are concatenated per event. Project groups precede user groups so a
project `continue: false` wins before lower-precedence handlers run, and
third-party plugin groups run last so a user's own configuration can preempt
them.

Legacy list-shaped documents are migrated only for events whose lifecycle
semantics genuinely match Hooks v2.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from pydantic import ValidationError

from deepagents_code.hooks.migration import (
    is_legacy_hooks_document,
    migrate_legacy_hooks,
)
from deepagents_code.hooks.models.config import (
    CommandHandlerSpec,
    HooksConfig,
    MatcherGroup,
)
from deepagents_code.hooks.models.domain import HookDiagnostic, HookEvent
from deepagents_code.model_config import DEFAULT_CONFIG_DIR

logger = logging.getLogger(__name__)
_LEGACY_HOOKS_REMOVAL_DATE = "September 1, 2026"
_EMPTY_ENV: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class HooksSource:
    """Provenance for the matcher groups contributed by one configuration source.

    Attributes:
        location: Path of the document the groups were read from, used to locate
            diagnostics.
        origin: Stable identity of a source whose authority is not its path,
            currently a plugin id. `None` for the project and user files. Unlike
            `location` it is portable across machines, so it is what the snapshot
            hash records.
        env: Environment overlay applied to every handler from this source, on
            top of the sanitized process environment.
    """

    location: str
    origin: str | None = None
    env: Mapping[str, str] = _EMPTY_ENV


@dataclass(frozen=True, slots=True)
class SourcedGroup:
    """A matcher group paired with the source that contributed it."""

    source: HooksSource
    group: MatcherGroup


_EMPTY_GROUPS: Mapping[HookEvent, tuple[SourcedGroup, ...]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class LoadedHooksConfig:
    """Validated configuration plus load diagnostics and source paths."""

    config: HooksConfig
    diagnostics: tuple[HookDiagnostic, ...]
    sources: tuple[Path, ...]
    snapshot_id: str
    groups: Mapping[HookEvent, tuple[SourcedGroup, ...]] = _EMPTY_GROUPS
    """Merged matcher groups with provenance, in the same order as `config`."""

    project_source_loaded: bool = False
    """Whether the project-scoped source was selected and successfully loaded.

    Set only when workspace trust allowed the project source and that file
    contributed configuration. Never inferred from path membership after
    canonical deduplication (symlinks / shared config dirs can alias paths).
    """


def project_hooks_path(project_root: Path) -> Path:
    """Return the project-scoped hooks configuration path.

    Args:
        project_root: Project root directory.

    Returns:
        `{project_root}/.deepagents/hooks.json`.
    """
    return project_root / ".deepagents" / "hooks.json"


def user_hooks_path(config_dir: Path | None = None) -> Path:
    """Return the user-scoped hooks configuration path.

    Args:
        config_dir: Alternate user config directory (tests).

    Returns:
        `{config_dir}/hooks.json`, defaulting to `~/.deepagents/hooks.json`.
    """
    return (config_dir or DEFAULT_CONFIG_DIR) / "hooks.json"


def load_hooks_config(
    *,
    project_root: Path,
    workspace_trusted: bool,
    config_dir: Path | None = None,
    paths: Sequence[Path] | None = None,
    documents: Sequence[tuple[HooksSource, Mapping[str, object]]] = (),
) -> LoadedHooksConfig:
    """Load, validate, merge, and hash Hooks v2 configuration.

    Args:
        project_root: Project root used for project precedence.
        workspace_trusted: Whether project-scoped hooks may be loaded.
        config_dir: Alternate user config directory.
        paths: Explicit trusted source paths in precedence order (highest first).
            When omitted, project hooks are included only for trusted workspaces,
            followed by user hooks.
        documents: Already-decoded documents with their provenance, merged after
            every file source so they hold the least authority. Used for plugin
            hooks, which are not read from a configuration path.

    Returns:
        Frozen load result with canonical `snapshot_id` and explicit project
        source provenance.
    """
    diagnostics: list[HookDiagnostic] = []
    merged: dict[HookEvent, list[SourcedGroup]] = {}
    loaded_paths: list[Path] = []
    project_source_loaded = False

    def _merge(document: HooksConfig, source: HooksSource) -> None:
        for event, groups in document.hooks.items():
            merged.setdefault(event, []).extend(
                SourcedGroup(source=source, group=group) for group in groups
            )

    def _ingest(path: Path, *, as_project: bool) -> None:
        nonlocal project_source_loaded
        resolved = path.expanduser().resolve(strict=False)
        document, file_diagnostics = _read_hooks_document(resolved)
        diagnostics.extend(file_diagnostics)
        if document is None:
            return
        if as_project:
            project_source_loaded = True
        loaded_paths.append(resolved)
        _merge(document, HooksSource(location=str(resolved)))

    if paths is not None:
        for path in dict.fromkeys(
            path.expanduser().resolve(strict=False) for path in paths
        ):
            _ingest(path, as_project=False)
    elif workspace_trusted:
        project_path = (
            project_hooks_path(project_root).expanduser().resolve(strict=False)
        )
        user_path = user_hooks_path(config_dir).expanduser().resolve(strict=False)
        _ingest(project_path, as_project=True)
        if user_path != project_path:
            _ingest(user_path, as_project=False)
    else:
        _ingest(user_hooks_path(config_dir), as_project=False)

    for source, raw_document in documents:
        document, document_diagnostics = _validate_hooks_document(
            raw_document, Path(source.location)
        )
        diagnostics.extend(document_diagnostics)
        if document is not None:
            _merge(document, source)

    groups = MappingProxyType(
        {event: tuple(sourced) for event, sourced in merged.items()}
    )
    config = HooksConfig(
        hooks={
            event: [sourced.group for sourced in sourced_groups]
            for event, sourced_groups in groups.items()
        }
    )
    return LoadedHooksConfig(
        config=config,
        diagnostics=tuple(diagnostics),
        sources=tuple(loaded_paths),
        snapshot_id=compute_snapshot_id(config, groups=groups),
        groups=groups,
        project_source_loaded=project_source_loaded,
    )


def compute_snapshot_id(
    config: HooksConfig,
    *,
    groups: Mapping[HookEvent, Sequence[SourcedGroup]] | None = None,
) -> str:
    """Return the canonical SHA-256 snapshot id for `config`.

    Args:
        config: Validated Hooks v2 configuration.
        groups: Matching sourced groups, so provenance participates in the hash.

    Returns:
        Lowercase hex digest of the canonical JSON serialization.
    """
    return hashlib.sha256(canonical_hooks_bytes(config, groups=groups)).hexdigest()


def canonical_hooks_bytes(
    config: HooksConfig,
    *,
    groups: Mapping[HookEvent, Sequence[SourcedGroup]] | None = None,
) -> bytes:
    """Serialize configuration into a stable byte representation.

    Args:
        config: Validated Hooks v2 configuration.
        groups: Matching sourced groups. When supplied, each group additionally
            records its non-file origin and environment overlay, so enabling a
            plugin that contributes hooks changes the snapshot id. Groups from
            the project and user files serialize identically either way.

    Returns:
        UTF-8 JSON with sorted keys, event order fixed to `HookEvent`, and
        `None` fields omitted. Unsupported fields such as `async` are
        excluded so equivalent configs hash identically.
    """
    payload = {
        "hooks": {
            event.value: _canonical_groups(event, config, groups)
            for event in HookEvent
            if event in config.hooks
        }
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _canonical_groups(
    event: HookEvent,
    config: HooksConfig,
    groups: Mapping[HookEvent, Sequence[SourcedGroup]] | None,
) -> list[dict[str, object]]:
    """Serialize one event's groups, annotating provenance when it is known.

    Returns:
        Canonical group payloads in configuration order.
    """
    sourced = None if groups is None else groups.get(event)
    if sourced is None:
        return [_canonical_group(group) for group in config.hooks.get(event, [])]
    return [_canonical_group(item.group, source=item.source) for item in sourced]


def _canonical_group(
    group: MatcherGroup, *, source: HooksSource | None = None
) -> dict[str, object]:
    raw = group.model_dump(
        mode="json", by_alias=True, exclude_none=True, exclude_defaults=True
    )
    handlers: list[dict[str, object]] = []
    hooks_raw = raw.get("hooks")
    if isinstance(hooks_raw, list):
        for item in hooks_raw:
            if not isinstance(item, dict):
                continue
            handler = {str(key): value for key, value in item.items() if key != "async"}
            handlers.append(handler)
    result: dict[str, object] = {"hooks": handlers}
    matcher = raw.get("matcher")
    if matcher is not None:
        result["matcher"] = matcher
    if source is not None and source.origin is not None:
        result["origin"] = source.origin
        if source.env:
            result["env"] = dict(sorted(source.env.items()))
    return result


def _read_hooks_document(
    path: Path,
) -> tuple[HooksConfig | None, tuple[HookDiagnostic, ...]]:
    if not path.is_file():
        return None, ()
    try:
        raw = path.read_text(encoding="utf-8")
        data: object = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        message = f"Failed to read hooks config at {path}: {exc}"
        logger.warning(message)
        return None, (
            HookDiagnostic(
                code="config_read_failed",
                severity="warning",
                message=message,
                field=str(path),
            ),
        )

    if is_legacy_hooks_document(data):
        hooks = data.get("hooks", []) if isinstance(data, dict) else []
        if not isinstance(hooks, list):
            return None, (
                HookDiagnostic(
                    code="invalid_config",
                    severity="warning",
                    message=f"Legacy hooks list missing at {path}",
                    field=str(path),
                ),
            )
        legacy_entries: list[dict[str, object]] = [
            {str(key): value for key, value in item.items()}
            for item in hooks
            if isinstance(item, Mapping)
        ]
        migrated = migrate_legacy_hooks(legacy_entries)
        migration_message = (
            f"Migrated semantically equivalent legacy hooks from {path}; "
            "unsupported legacy events remain unmapped"
            if migrated.hooks
            else (
                f"Legacy hooks at {path} contained no events that are safe to "
                "migrate to Hooks v2"
            )
        )
        return migrated, (
            HookDiagnostic(
                code="legacy_deprecated",
                severity="warning",
                message=(
                    f"Legacy hooks configuration at {path} is deprecated and will "
                    f"stop being supported on {_LEGACY_HOOKS_REMOVAL_DATE}"
                ),
                field=str(path),
            ),
            HookDiagnostic(
                code="legacy_migrated" if migrated.hooks else "legacy_unmapped",
                severity="warning",
                message=migration_message,
                field=str(path),
            ),
        )

    return _validate_hooks_document(data, path)


def _validate_hooks_document(
    data: object,
    path: Path,
) -> tuple[HooksConfig | None, tuple[HookDiagnostic, ...]]:
    if not isinstance(data, Mapping):
        return None, (_invalid_config(path, "", "expected an object"),)
    raw_hooks = data.get("hooks")
    if not isinstance(raw_hooks, Mapping):
        return None, (_invalid_config(path, "hooks", "expected an object"),)

    hooks: dict[HookEvent, list[MatcherGroup]] = {}
    diagnostics: list[HookDiagnostic] = []
    for raw_event, raw_groups in raw_hooks.items():
        event_field = f"hooks.{raw_event}"
        if not isinstance(raw_event, str):
            diagnostics.append(_invalid_config(path, event_field, "unknown hook event"))
            continue
        try:
            event = HookEvent(raw_event)
        except ValueError:
            diagnostics.append(_invalid_config(path, event_field, "unknown hook event"))
            continue
        if not isinstance(raw_groups, list):
            diagnostics.append(
                _invalid_config(path, event_field, "expected a list of matcher groups")
            )
            continue

        groups: list[MatcherGroup] = []
        for group_index, raw_group in enumerate(raw_groups):
            group_field = f"{event_field}[{group_index}]"
            group, group_diagnostics = _validate_matcher_group(
                raw_group,
                path,
                group_field,
            )
            diagnostics.extend(group_diagnostics)
            if group is not None:
                groups.append(group)
        if groups or not raw_groups:
            hooks[event] = groups

    if raw_hooks and not hooks:
        return None, tuple(diagnostics)
    return HooksConfig(hooks=hooks), tuple(diagnostics)


def _validate_matcher_group(
    data: object,
    path: Path,
    field: str,
) -> tuple[MatcherGroup | None, tuple[HookDiagnostic, ...]]:
    if not isinstance(data, Mapping):
        return None, (_invalid_config(path, field, "expected an object"),)
    raw_handlers = data.get("hooks")
    if not isinstance(raw_handlers, list):
        return None, (
            _invalid_config(path, f"{field}.hooks", "expected a list of handlers"),
        )

    handlers: list[CommandHandlerSpec] = []
    diagnostics: list[HookDiagnostic] = []
    for handler_index, raw_handler in enumerate(raw_handlers):
        handler_field = f"{field}.hooks[{handler_index}]"
        try:
            handlers.append(CommandHandlerSpec.model_validate(raw_handler))
        except ValidationError as exc:
            diagnostics.append(_validation_error(path, handler_field, exc))

    if raw_handlers and not handlers:
        return None, tuple(diagnostics)

    group_data = dict(data)
    group_data["hooks"] = handlers
    try:
        return MatcherGroup.model_validate(group_data), tuple(diagnostics)
    except ValidationError as exc:
        diagnostics.append(_validation_error(path, field, exc))
        return None, tuple(diagnostics)


def _validation_error(
    path: Path,
    field: str,
    error: ValidationError,
) -> HookDiagnostic:
    details = "; ".join(
        str(item["msg"])
        for item in error.errors(include_url=False, include_input=False)
    )
    return _invalid_config(path, field, details)


def _invalid_config(path: Path, field: str, detail: str) -> HookDiagnostic:
    location = f"{path}:{field}" if field else str(path)
    message = f"Invalid hooks config at {location}: {detail}"
    logger.warning(message)
    return HookDiagnostic(
        code="invalid_config",
        severity="warning",
        message=message,
        field=location,
    )
