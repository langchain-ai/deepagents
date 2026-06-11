"""Surface patching helpers."""

from __future__ import annotations

import contextlib
import importlib
import os
from collections.abc import Iterator
from pathlib import Path

from better_harness.core import Experiment, Variant

VARIANT_ENV = "BETTER_HARNESS_VARIANT_FILE"


def build_baseline_variant(experiment: Experiment) -> Variant:
    """Build the baseline variant from the configured surface bases."""
    values = {name: surface.base_value for name, surface in experiment.surfaces.items()}
    return Variant(
        label="baseline",
        model=experiment.model,
        changed_surfaces=(),
        surfaces=experiment.surfaces,
        values=values,
    )


def build_variant(
    *,
    experiment: Experiment,
    label: str,
    values: dict[str, str],
) -> Variant:
    """Build one variant from raw surface values."""
    changed_surfaces = tuple(
        sorted(
            name
            for name, surface in experiment.surfaces.items()
            if values[name] != surface.base_value
        )
    )
    return Variant(
        label=label,
        model=experiment.model,
        changed_surfaces=changed_surfaces,
        surfaces=experiment.surfaces,
        values=values,
    )


def patch_from_env() -> None:
    """Patch module attrs from the saved variant file."""
    raw_path = os.environ.get(VARIANT_ENV)
    if not raw_path:
        return
    variant = Variant.load(Path(raw_path))
    patch_module_attrs(variant.attr_overrides())


def patch_module_attrs(overrides: dict[str, str]) -> None:
    """Apply `module:attribute -> value` overrides."""
    for target, value in overrides.items():
        module_name, separator, attribute = target.partition(":")
        if not separator:
            msg = f"invalid module_attr target {target!r}; expected module:attribute"
            raise ValueError(msg)
        module = importlib.import_module(module_name)
        setattr(module, attribute, value)


@contextlib.contextmanager
def workspace_override_context(
    workspace_root: Path,
    overrides: dict[str, str],
) -> Iterator[None]:
    """Temporarily replace files in the target workspace."""
    backups: dict[Path, str | None] = {}
    try:
        for relative_path, value in overrides.items():
            target = workspace_root / relative_path
            backups[target] = target.read_text() if target.exists() else None
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(value)
        yield
    finally:
        for target, original in backups.items():
            if original is None:
                if target.exists():
                    target.unlink()
            else:
                target.write_text(original)


def prepend_pythonpath(paths: list[Path], existing: str | None) -> str:
    """Put one or more paths first on PYTHONPATH."""
    parts = [str(path) for path in paths]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def ensure_sitecustomize(runtime_dir: Path) -> Path:
    """Write a sitecustomize.py that applies module_attr patches from the env."""
    runtime_dir.mkdir(parents=True, exist_ok=True)
    sitecustomize_path = runtime_dir / "sitecustomize.py"
    sitecustomize_path.write_text(
        "from better_harness.patching import patch_from_env\n"
        "patch_from_env()\n"
    )
    return runtime_dir
