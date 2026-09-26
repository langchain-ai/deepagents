"""Load optional providers with install guidance that preserves the real cause."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def load_driver(module: str, extra: str, subject: str) -> ModuleType:
    """Import an optional provider, naming the extra only when that extra is missing.

    Args:
        module: Importable module supplied by the extra.
        extra: Project extra that installs it.
        subject: Sentence lead-in ending in its verb, e.g. `History backend requires`.

    Raises:
        ImportError: The extra is absent, or the module itself failed to import. An
            installed extra that fails for its own reason (a broken native
            dependency, a missing system library) re-raises unchanged, because
            telling the user to install what they already have hides the cause.
    """
    try:
        return importlib.import_module(module)
    except ImportError as error:
        if (
            error.name is not None
            and module != error.name
            and not module.startswith(error.name + ".")
        ):
            raise
        msg = f"{subject} deepagents-talon[{extra}]: uv sync --extra {extra}"
        raise ImportError(msg) from error
