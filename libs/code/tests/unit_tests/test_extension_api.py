"""Tests for the public extension factory API."""

import json
from pathlib import Path
from typing import Any, cast

import pytest
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.types import AgentMiddleware

from deepagents_code._env_vars import EXPERIMENTAL
from deepagents_code.extensions import ExtensionAPI, ExtensionMode
from deepagents_code.extensions.discovery import discover_extensions
from deepagents_code.extensions.registry import (
    ExtensionError,
    ExtensionRegistry,
    SourceInfo,
)
from deepagents_code.plugins.manifest import load_manifest
from deepagents_code.plugins.models import ComponentInventory, PluginInstance


class _Middleware(AgentMiddleware):
    """Minimal extension middleware."""


def _write_manifest(
    root: Path, *, path: str = "./extension.py", version: str | None = None
) -> None:
    manifest: dict[str, Any] = {
        "name": "example",
        "extensions": {"com.langchain.deepagents.code": {"pythonExtensions": path}},
    }
    if version is not None:
        manifest["version"] = version
    (root / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def api() -> ExtensionAPI:
    """Return a factory-scoped registrar."""
    return ExtensionAPI(
        ExtensionRegistry(),
        SourceInfo(Path("/extensions/example.py")),
        cwd=Path("/workspace"),
        mode=ExtensionMode.INTERACTIVE,
    )
