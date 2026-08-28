"""Tests for the public extension factory API."""

from pathlib import Path

import pytest
from langchain.agents.middleware.types import AgentMiddleware

from deepagents_code.extensions import ExtensionAPI, ExtensionMode
from deepagents_code.extensions.registry import ExtensionRegistry, SourceInfo


class _Middleware(AgentMiddleware):
    """Minimal extension middleware."""


@pytest.fixture
def api() -> ExtensionAPI:
    """Return a factory-scoped registrar."""
    return ExtensionAPI(
        ExtensionRegistry(),
        SourceInfo(Path("/extensions/example.py")),
        cwd=Path("/workspace"),
        mode=ExtensionMode.INTERACTIVE,
    )
