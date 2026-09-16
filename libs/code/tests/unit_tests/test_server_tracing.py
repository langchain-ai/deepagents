"""A server cannot reroute cached or concurrent runtimes to another identity."""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, Mock, patch

import pytest

from deepagents_code import config as config_module
from deepagents_code._server_config import ServerConfig
from deepagents_code.workspace import WorkspaceConflictError, resolve_workspace

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from types import ModuleType

    from deepagents_code.workspace import WorkspaceBinding


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """Use real tracing publication with an inert client and no model or tools."""
    from langsmith import Client, configure, run_trees, utils

    sys.modules.pop("deepagents_code.server_graph", None)
    module = importlib.import_module("deepagents_code.server_graph")
    monkeypatch.setattr(run_trees, "_CLIENT", None)
    monkeypatch.setattr(
        config_module,
        "configure_langsmith_secret_redaction",
        lambda: configure(client=Mock(spec=Client)),
    )
    monkeypatch.setattr(
        module,
        "_make_graphs_in_environment",
        AsyncMock(
            side_effect=lambda **_: module.ServerRuntime(object(), Mock(), Mock())
        ),
    )
    with patch.dict(os.environ, {}, clear=True):
        utils.get_env_var.cache_clear()
        utils.get_tracer_project.cache_clear()
        yield module
    utils.get_env_var.cache_clear()
    utils.get_tracer_project.cache_clear()


def _workspace(path: Path, environment: dict[str, str]) -> WorkspaceBinding:
    """Give a synthetic workspace its own dotenv and default server policy."""
    path.mkdir()
    (path / ".env").write_text(
        "".join(f"{key}={value}\n" for key, value in environment.items())
    )
    config = ServerConfig()
    return resolve_workspace(
        str(path),
        config.to_workspace_payload(),
        config_fingerprint=config.workspace_fingerprint(),
    )


@pytest.mark.parametrize(
    ("selector", "value"),
    [
        ("LANGSMITH_TRACING", "false"),
        ("DEEPAGENTS_CODE_LANGSMITH_TRACING", "false"),
        ("LANGSMITH_API_KEY", "other-test-key"),
        ("DEEPAGENTS_CODE_LANGSMITH_API_KEY", "other-test-key"),
        ("LANGSMITH_ENDPOINT", "https://other.example.com"),
        ("LANGSMITH_PROJECT", "other-project"),
        ("DEEPAGENTS_CODE_LANGSMITH_PROJECT", "other-project"),
        ("LANGSMITH_PROFILE", "other-profile"),
        ("LANGSMITH_CONFIG_FILE", "/unused/profile.json"),
        ("LANGSMITH_WORKSPACE_ID", "other-workspace"),
        ("LANGSMITH_RUNS_ENDPOINTS", '{"https://other.example.com":"test-key"}'),
        ("DEEPAGENTS_CODE_LANGSMITH_REDACT", "false"),
    ],
)
async def test_conflicting_workspace_preserves_cached_tracing(
    server: ModuleType, tmp_path: Path, selector: str, value: str
) -> None:
    from langsmith import run_trees, utils

    environment = {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "first-test-key",
        "LANGSMITH_PROJECT": "first-project",
    }
    first = _workspace(tmp_path / "first", environment)
    other = _workspace(tmp_path / "other", {**environment, selector: value})
    runtime = await server._workspace_runtime(first)
    client = run_trees.get_cached_client()

    with pytest.raises(WorkspaceConflictError, match="separate server"):
        await server._workspace_runtime(other)

    assert await server._workspace_runtime(first) is runtime
    assert utils.get_tracer_project() == "first-project"
    assert os.environ["LANGSMITH_API_KEY"] == "first-test-key"
    assert run_trees.get_cached_client() is client


async def test_matching_workspaces_share_tracing_during_concurrent_builds(
    server: ModuleType, tmp_path: Path
) -> None:
    from langsmith import run_trees

    environment = {"LANGSMITH_PROJECT": "shared-project"}
    first = _workspace(tmp_path / "first", environment)
    second = _workspace(tmp_path / "second", environment)
    await server._workspace_runtime(first)
    client = run_trees.get_cached_client()

    results = await asyncio.gather(
        server._workspace_runtime(first), server._workspace_runtime(second)
    )

    assert len(results) == 2
    assert results[0] is not results[1]
    assert run_trees.get_cached_client() is client
    assert os.environ["LANGSMITH_PROJECT"] == "shared-project"


async def test_concurrent_conflicting_builds_cannot_change_active_tracing(
    server: ModuleType, tmp_path: Path
) -> None:
    from langsmith import utils

    first = _workspace(tmp_path / "first", {"LANGSMITH_PROJECT": "first-project"})
    other = _workspace(tmp_path / "other", {"LANGSMITH_PROJECT": "other-project"})
    started, release = asyncio.Event(), asyncio.Event()

    async def build(**_kwargs: object) -> object:
        started.set()
        await release.wait()
        return server.ServerRuntime(object(), object(), object())

    with patch.object(server, "_make_graphs_in_environment", build):
        first_task = asyncio.create_task(server._workspace_runtime(first))
        await started.wait()
        other_task = asyncio.create_task(server._workspace_runtime(other))
        release.set()
        first_result, other_result = await asyncio.gather(
            first_task, other_task, return_exceptions=True
        )

    assert not isinstance(first_result, BaseException)
    assert isinstance(other_result, WorkspaceConflictError)
    assert utils.get_tracer_project() == "first-project"


async def test_failed_build_and_cache_eviction_keep_tracing_reservation(
    server: ModuleType, tmp_path: Path
) -> None:
    first = _workspace(tmp_path / "first", {"LANGSMITH_PROJECT": "first-project"})
    other = _workspace(tmp_path / "other", {"LANGSMITH_PROJECT": "other-project"})
    with (
        patch.object(
            server,
            "_make_graphs_in_environment",
            AsyncMock(side_effect=ValueError("build failed")),
        ),
        pytest.raises(ValueError, match="build failed"),
    ):
        await server._workspace_runtime(first)

    with pytest.raises(WorkspaceConflictError):
        await server._workspace_runtime(other)
    await server._workspace_runtime(first)
    server._workspace_runtimes.clear()
    with pytest.raises(WorkspaceConflictError):
        await server._workspace_runtime(other)


async def test_readiness_runtime_reserves_tracing(
    server: ModuleType, tmp_path: Path
) -> None:
    first = _workspace(tmp_path / "first", {"LANGSMITH_PROJECT": "first-project"})
    other = _workspace(tmp_path / "other", {"LANGSMITH_PROJECT": "other-project"})
    with patch.object(
        ServerConfig, "from_env", return_value=ServerConfig(cwd=first.cwd)
    ):
        runtime = await server.get_server_runtime()

    with pytest.raises(WorkspaceConflictError):
        await server._workspace_runtime(other)
    assert await server._workspace_runtime(first) is runtime
