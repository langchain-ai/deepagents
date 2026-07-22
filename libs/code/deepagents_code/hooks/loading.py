"""Validated Hooks v2 configuration loading, merging, and hashing.

Precedence (highest first, earlier in reduction order):

1. Project: `{project_root}/.deepagents/hooks.json`
2. User: `~/.deepagents/hooks.json` (or `config_dir/hooks.json` in tests)

Sources are concatenated per event. Project groups precede user groups so a
project `continue: false` wins before lower-precedence handlers run.

Legacy list-shaped documents are migrated only for events whose lifecycle
semantics genuinely match Hooks v2.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import (  # noqa: TC003 - used in runtime dataclass fields and path ops
    Path,
)

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


@dataclass(frozen=True, slots=True)
class LoadedHooksConfig:
    """Validated configuration plus load diagnostics and source paths."""

    config: HooksConfig
    diagnostics: tuple[HookDiagnostic, ...]
    sources: tuple[Path, ...]
    snapshot_id: str


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
) -> LoadedHooksConfig:
    """Load, validate, merge, and hash Hooks v2 configuration.

    Args:
        project_root: Project root used for project precedence.
        workspace_trusted: Whether project-scoped hooks may be loaded.
        config_dir: Alternate user config directory.
        paths: Explicit trusted source paths in precedence order (highest first).
            When omitted, project hooks are included only for trusted workspaces,
            followed by user hooks.

    Returns:
        Frozen load result with canonical `snapshot_id`.
    """
    configured_sources = (
        tuple(paths)
        if paths is not None
        else (
            (project_hooks_path(project_root), user_hooks_path(config_dir))
            if workspace_trusted
            else (user_hooks_path(config_dir),)
        )
    )
    sources = tuple(
        dict.fromkeys(
            path.expanduser().resolve(strict=False) for path in configured_sources
        )
    )
    diagnostics: list[HookDiagnostic] = []
    merged: dict[HookEvent, list[MatcherGroup]] = {}
    loaded_paths: list[Path] = []

    for path in sources:
        document, file_diagnostics = _read_hooks_document(path)
        diagnostics.extend(file_diagnostics)
        if document is None:
            continue
        loaded_paths.append(path)
        for event, groups in document.hooks.items():
            merged.setdefault(event, []).extend(groups)

    config = HooksConfig(hooks=merged)
    return LoadedHooksConfig(
        config=config,
        diagnostics=tuple(diagnostics),
        sources=tuple(loaded_paths),
        snapshot_id=compute_snapshot_id(config),
    )


def compute_snapshot_id(config: HooksConfig) -> str:
    """Return the canonical SHA-256 snapshot id for `config`.

    Args:
        config: Validated Hooks v2 configuration.

    Returns:
        Lowercase hex digest of the canonical JSON serialization.
    """
    return hashlib.sha256(canonical_hooks_bytes(config)).hexdigest()


def canonical_hooks_bytes(config: HooksConfig) -> bytes:
    """Serialize configuration into a stable byte representation.

    Args:
        config: Validated Hooks v2 configuration.

    Returns:
        UTF-8 JSON with sorted keys, event order fixed to `HookEvent`, and
        `None` fields omitted. Unsupported MVP fields such as `async` are
        excluded so equivalent configs hash identically.
    """
    payload = {
        "hooks": {
            event.value: [
                _canonical_group(group) for group in config.hooks.get(event, [])
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


def _canonical_group(group: MatcherGroup) -> dict[str, object]:
    raw = group.model_dump(mode="json", by_alias=True, exclude_none=True)
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
