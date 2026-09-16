"""Tests for agent-server extension hosting."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from deepagents.backends import FilesystemBackend
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from deepagents_code.agent import create_cli_agent
from deepagents_code.extensions.registry import (
    ExtensionError,
    ExtensionRegistry,
    SourceInfo,
)
from deepagents_code.offload import _ArtifactsStorage


@pytest.fixture(autouse=True)
def _experimental(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPAGENTS_CODE_EXPERIMENTAL", "1")


@tool("existing")
def _existing() -> str:
    """Represent an existing dcode tool."""
    return "existing"


@tool("existing")
def _replacement() -> str:
    """Represent an extension override."""
    return "replacement"


class _Middleware(AgentMiddleware):
    """Represent extension middleware."""

    name = "extension-middleware"


def test_internal_backend_route_overlap_fails_agent_startup(tmp_path: Path) -> None:
    """An internal storage collision fails instead of being silently ignored."""
    registry = ExtensionRegistry()
    registry.add_backend_route(
        "/artifacts/",
        FilesystemBackend(root_dir=tmp_path / "blocked", virtual_mode=True),
        SourceInfo(tmp_path / "extension.py"),
    )
    model = GenericFakeChatModel(messages=iter([AIMessage(content="ok")]))
    model.profile = {"max_input_tokens": 200000}

    with (
        patch("deepagents_code.agent.create_deep_agent", return_value=Mock()),
        patch(
            "deepagents_code.agent._artifacts_root",
            return_value=_ArtifactsStorage(root="/artifacts"),
        ),
        pytest.raises(ExtensionError, match="overlaps an internal route"),
    ):
        create_cli_agent(
            model,
            "test",
            enable_memory=False,
            enable_skills=False,
            enable_shell=False,
            system_prompt="test",
            cwd=tmp_path,
            extension_registry=registry,
        )
