"""Hatchling build hook that stamps the release commit into the package.

When `DEEPAGENTS_CODE_BUILD_COMMIT` is set (CI release builds), this writes
`deepagents_code/_build_info.py` so `dcode doctor` can report the exact commit
a wheel was built from. Editable and local builds leave the env var unset, so
no file is generated and `dcode doctor` falls back to a live `git` probe of the
working tree.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import (  # ty: ignore[unresolved-import]  # build-time-only dependency
    BuildHookInterface,
)

_COMMIT_ENV = "DEEPAGENTS_CODE_BUILD_COMMIT"
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_TARGET = Path("deepagents_code") / "_build_info.py"


class CustomBuildHook(BuildHookInterface):
    """Stamps `_build_info.py` with the release commit when the env var is set."""

    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:  # noqa: ARG002  # `version` is part of the hatchling hook signature
        """Write the generated build-info module before the build collects files.

        Raises:
            ValueError: If the commit env var is set but not a hex git SHA.
        """
        commit = os.environ.get(_COMMIT_ENV, "").strip()
        if not commit:
            return
        if not _COMMIT_RE.match(commit):
            msg = f"{_COMMIT_ENV} must be a hex git SHA, got: {commit!r}"
            raise ValueError(msg)
        short = commit[:7].lower()
        target = Path(self.root) / _TARGET
        target.write_text(
            '"""Generated at build time. Do not edit or commit."""\n\n'
            f'BUILD_COMMIT = "{short}"\n',
            encoding="utf-8",
        )
        # `_TARGET` is gitignored, so hatchling excludes it from the build by
        # default; registering it as an artifact force-includes it in the dist.
        build_data.setdefault("artifacts", []).append(str(_TARGET))

    def finalize(
        self,
        version: str,  # noqa: ARG002  # part of the hatchling hook signature
        build_data: dict[str, Any],  # noqa: ARG002  # part of the hook signature
        artifact_path: str,  # noqa: ARG002  # part of the hook signature
    ) -> None:
        """Remove the generated module so the working tree stays clean.

        Runs only when the stamp env var was set, mirroring `initialize`, so
        editable and local builds (which never wrote the file) skip the unlink.
        """
        if not os.environ.get(_COMMIT_ENV, "").strip():
            return
        (Path(self.root) / _TARGET).unlink(missing_ok=True)
