"""Validated Hooks v2 configuration loading, merging, and hashing.

Precedence (highest first, earlier in reduction order):

1. Project: `{cwd}/.deepagents/hooks.json`
2. User: `~/.deepagents/hooks.json` (or `config_dir/hooks.json` in tests)

Sources are concatenated per event. Project groups precede user groups so a
project `continue: false` wins before lower-precedence handlers run.

Legacy list-shaped documents are migrated only for events whose lifecycle
semantics genuinely match Hooks v2. `tool.use` is never treated as `PreToolUse`.
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
from deepagents_code.hooks.models.adapters import HOOKS_CONFIG_ADAPTER
from deepagents_code.hooks.models.config import HooksConfig, MatcherGroup
from deepagents_code.hooks.models.domain import HookDiagnostic, HookEvent
from deepagents_code.model_config import DEFAULT_CONFIG_DIR

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoadedHooksConfig:
    """Validated configuration plus load diagnostics and source paths."""

    config: HooksConfig
    diagnostics: tuple[HookDiagnostic, ...]
    sources: tuple[Path, ...]
    snapshot_id: str


def project_hooks_path(cwd: Path) -> Path:
    """Return the project-scoped hooks configuration path.

    Args:
        cwd: Session working directory.

    Returns:
        `{cwd}/.deepagents/hooks.json`.
    """
    return cwd / ".deepagents" / "hooks.json"


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
    cwd: Path,
    workspace_trusted: bool,
    config_dir: Path | None = None,
    paths: Sequence[Path] | None = None,
) -> LoadedHooksConfig:
    """Load, validate, merge, and hash Hooks v2 configuration.

    Args:
        cwd: Session working directory used for project precedence.
        workspace_trusted: Whether project-scoped hooks may be loaded.
        config_dir: Alternate user config directory.
        paths: Explicit trusted source paths in precedence order (highest first).
            When omitted, project hooks are included only for trusted workspaces,
            followed by user hooks.

    Returns:
        Frozen load result with canonical `snapshot_id`.
    """
    sources = (
        tuple(paths)
        if paths is not None
        else (
            (project_hooks_path(cwd), user_hooks_path(config_dir))
            if workspace_trusted
            else (user_hooks_path(config_dir),)
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
        data = json.loads(raw)
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
        legacy_entries = [item for item in hooks if isinstance(item, Mapping)]
        migrated = migrate_legacy_hooks(legacy_entries)
        message = (
            f"Migrated semantically equivalent session.end hooks from {path}; "
            "all other legacy events remain unmapped"
            if migrated.hooks
            else (
                f"Legacy hooks at {path} contained no events that are safe to "
                "migrate to Hooks v2"
            )
        )
        diagnostic = HookDiagnostic(
            code="legacy_migrated" if migrated.hooks else "legacy_unmapped",
            severity="debug",
            message=message,
            field=str(path),
        )
        return migrated, (diagnostic,)

    try:
        config = HOOKS_CONFIG_ADAPTER.validate_python(data)
    except ValidationError as exc:
        message = f"Invalid hooks config at {path}: {exc.title}"
        logger.warning(message)
        return None, (
            HookDiagnostic(
                code="invalid_config",
                severity="warning",
                message=message,
                field=str(path),
            ),
        )
    return config, ()
