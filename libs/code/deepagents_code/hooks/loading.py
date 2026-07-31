"""Validated Hooks v2 configuration loading, merging, and hashing.

Precedence (highest first, earlier in reduction order):

1. Project: `{project_root}/.deepagents/hooks.json`
2. User: `~/.deepagents/hooks.json` (or `config_dir/hooks.json` in tests)
3. Plugin: `hooks.json` documents contributed by enabled plugins

Sources are concatenated per event. Precedence is reduction order, not execution
order: every matching handler runs, and the first one that stops processing
decides the event.

Legacy list-shaped documents are migrated only for events whose lifecycle
semantics genuinely match Hooks v2.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from pydantic import ValidationError
from typing_extensions import override

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

if TYPE_CHECKING:
    from deepagents_code.json_types import JsonValue

logger = logging.getLogger(__name__)
_LEGACY_HOOKS_REMOVAL_DATE = "September 1, 2026"


@dataclass(frozen=True, slots=True)
class HooksSource(ABC):
    """Origin of the matcher groups contributed by one hooks document."""

    location: str

    @abstractmethod
    def resolve_variables(self, value: str, *, shell_syntax: bool = False) -> str:
        """Resolve the variable references this source defines.

        Args:
            value: One `argv` element, or a shell-form `command`.
            shell_syntax: Whether a shell interprets `value`, in which case a
                reference is rewritten for the shell instead of substituted.

        Returns:
            The resolved argument or command.
        """


@dataclass(frozen=True, slots=True)
class FileHooksSource(HooksSource):
    """A project or user hooks file, which defines no variables."""

    @override
    def resolve_variables(self, value: str, *, shell_syntax: bool = False) -> str:
        """Return `value` as authored.

        Returns:
            The unchanged argument or command.
        """
        return value


@dataclass(frozen=True, slots=True)
class PluginHooksSource(HooksSource):
    """Origin and environment for groups one enabled plugin contributed."""

    plugin_id: str
    env: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze the environment overlay so the snapshot cannot be mutated."""
        object.__setattr__(self, "env", MappingProxyType(dict(self.env)))

    @override
    def resolve_variables(self, value: str, *, shell_syntax: bool = False) -> str:
        """Substitute direct arguments, or adapt shell references for Windows.

        Returns:
            The resolved argument or command.
        """
        if shell_syntax and os.name != "nt":
            return value
        for key, replacement in self.env.items():
            value = value.replace(
                f"${{{key}}}", f"%{key}%" if shell_syntax else replacement
            )
        return value


SourcedGroup = tuple[HooksSource, MatcherGroup]


UNSOURCED: Final = FileHooksSource(location="")
"""Provenance for groups handled without it, adding no origin or env overlay."""


@dataclass(frozen=True, slots=True)
class LoadedHooksConfig:
    """Validated configuration plus load diagnostics and source paths."""

    config: HooksConfig
    diagnostics: tuple[HookDiagnostic, ...]
    sources: tuple[Path, ...]
    snapshot_id: str
    groups: Mapping[HookEvent, tuple[SourcedGroup, ...]]
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
    documents: Sequence[tuple[HooksSource, JsonValue]] = (),
    document_diagnostics: Sequence[HookDiagnostic] = (),
) -> LoadedHooksConfig:
    """Load, validate, merge, and hash Hooks v2 configuration.

    Args:
        project_root: Project root used for project precedence.
        workspace_trusted: Whether project-scoped hooks may be loaded.
        config_dir: Alternate user config directory.
        paths: Explicit trusted source paths in precedence order (highest first).
            When omitted, project hooks are included only for trusted workspaces,
            followed by user hooks.
        documents: Already-decoded plugin documents with their provenance, merged
            after every file source so they hold the least authority. Validated
            here, so a malformed one is reported rather than dropped.
        document_diagnostics: Diagnostics the caller collected while producing
            `documents`, carried into the load result.

    Returns:
        Frozen load result with canonical `snapshot_id` and explicit project
        source provenance.
    """
    diagnostics: list[HookDiagnostic] = list(document_diagnostics)
    merged: dict[HookEvent, list[SourcedGroup]] = {}
    loaded_paths: list[Path] = []
    project_source_loaded = False

    def _merge(document: HooksConfig, source: HooksSource) -> None:
        for event, groups in document.hooks.items():
            merged.setdefault(event, []).extend((source, group) for group in groups)

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
        _merge(document, FileHooksSource(location=str(resolved)))

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
        document, validation_diagnostics = _validate_hooks_document(
            raw_document, Path(source.location)
        )
        diagnostics.extend(validation_diagnostics)
        if document is not None:
            _merge(document, source)

    groups = MappingProxyType(
        {event: tuple(sourced) for event, sourced in merged.items()}
    )
    config = HooksConfig(
        hooks={
            event: [group for _source, group in sourced_groups]
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
    known = groups or {}
    payload = {
        "hooks": {
            event.value: [
                _canonical_group(group, source=source)
                for source, group in known.get(event)
                or [(UNSOURCED, group) for group in config.hooks[event]]
            ]
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


def _canonical_group(group: MatcherGroup, *, source: HooksSource) -> dict[str, object]:
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
    if isinstance(source, PluginHooksSource):
        result["origin"] = source.plugin_id
        if source.env:
            result["env"] = dict(sorted(source.env.items()))
    return result


def read_hooks_json(
    path: Path,
) -> tuple[bool, JsonValue, tuple[HookDiagnostic, ...]]:
    """Decode one hooks JSON document, reporting read failures as diagnostics.

    Shared by the project/user file loader and by callers that must transform a
    document before it is validated, so every hooks document on disk reports
    unreadable bytes, invalid UTF-8, and malformed JSON the same way instead of
    raising into an unrelated caller.

    Args:
        path: Document path.

    Returns:
        Whether decoding succeeded, the decoded JSON value, and any diagnostics.
        An absent file is not a diagnostic.
    """
    if not path.is_file():
        return False, None, ()
    try:
        decoded: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        message = f"Failed to read hooks config at {path}: {exc}"
        logger.warning(message)
        return (
            False,
            None,
            (
                HookDiagnostic(
                    code="config_read_failed",
                    severity="warning",
                    message=message,
                    field=str(path),
                ),
            ),
        )
    return True, decoded, ()


def _read_hooks_document(
    path: Path,
) -> tuple[HooksConfig | None, tuple[HookDiagnostic, ...]]:
    decoded, data, read_diagnostics = read_hooks_json(path)
    if not decoded:
        return None, read_diagnostics

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
