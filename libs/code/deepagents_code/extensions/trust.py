"""Persist project-extension trust decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from deepagents_code.hooks.trust import (
    is_project_hooks_trusted,
    trust_project_hooks,
)

if TYPE_CHECKING:
    from pathlib import Path


def _store_path() -> Path:
    from deepagents_code.model_config import DEFAULT_STATE_DIR

    return DEFAULT_STATE_DIR / "extension_trust.json"


def is_project_extensions_trusted(
    project_root: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Return whether extension execution is trusted for a project.

    Args:
        project_root: Project root to inspect.
        store_path: Alternate extension trust store.

    Returns:
        Whether the canonical project root has a persisted grant.
    """
    return is_project_hooks_trusted(
        project_root,
        store_path=store_path or _store_path(),
    )


def trust_project_extensions(
    project_root: Path | str,
    *,
    store_path: Path | None = None,
) -> bool:
    """Persist extension execution trust for a project.

    Args:
        project_root: Project root to trust.
        store_path: Alternate extension trust store.

    Returns:
        Whether the grant was saved.
    """
    return trust_project_hooks(
        project_root,
        store_path=store_path or _store_path(),
    )
