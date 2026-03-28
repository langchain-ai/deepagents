"""Tests for server manager bootstrap behavior."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from deepagents_cli._server_config import ServerConfig
from deepagents_cli._server_constants import ENV_PREFIX
from deepagents_cli.project_utils import ProjectContext
from deepagents_cli.server_manager import (
    _apply_server_config,
    _write_pyproject,
    server_session,
    start_server_and_get_agent,
)


class TestServerConfigRoundTrip:
    """The env-var serialization contract between CLI and server graph."""

    def test_round_trip_preserves_all_fields(self) -> None:
        """to_env -> from_env should reconstruct the original config."""
        original = ServerConfig(
            model="anthropic:claude-sonnet-4-6",
            model_params={"temperature": 0.7},
            assistant_id="my-agent",
            system_prompt="Be helpful",
            auto_approve=True,
            interrupt_shell_only=True,
            interactive=False,
            enable_shell=False,
            enable_ask_user=True,
            enable_memory=False,
            enable_skills=False,
            sandbox_type="modal",
            sandbox_id="sb-12345",
            sandbox_setup="/home/user/setup.sh",
            cwd="/home/user/project",
            project_root="/home/user/project",
            mcp_config_path="/home/user/.mcp.json",
            no_mcp=True,
            trust_project_mcp=True,
        )
        env_dict = original.to_env()
        with patch.dict(os.environ, {}, clear=True):
            for suffix, value in env_dict.items():
                if value is not None:
                    os.environ[f"{ENV_PREFIX}{suffix}"] = value
            restored = ServerConfig.from_env()

        assert restored == original

    def test_defaults_round_trip(self) -> None:
        """Default config should survive a round trip."""
        original = ServerConfig()
        env_dict = original.to_env()
        with patch.dict(os.environ, {}, clear=True):
            for suffix, value in env_dict.items():
                if value is not None:
                    os.environ[f"{ENV_PREFIX}{suffix}"] = value
            restored = ServerConfig.from_env()

        assert restored == original

    def test_trust_project_mcp_none_round_trips(self) -> None:
        """None trust_project_mcp should survive a round trip."""
        original = ServerConfig(trust_project_mcp=None)
        env_dict = original.to_env()
        with patch.dict(os.environ, {}, clear=True):
            for suffix, value in env_dict.items():
                if value is not None:
                    os.environ[f"{ENV_PREFIX}{suffix}"] = value
            restored = ServerConfig.from_env()

        assert restored.trust_project_mcp is None


class TestApplyServerConfig:
    """Tests for env-var serialization via ServerConfig."""

    def test_normalizes_relative_mcp_path_from_project_context(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Relative MCP config paths should be made absolute before crossing."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        (project_root / ".git").mkdir()
        user_cwd = project_root / "src"
        user_cwd.mkdir()

        project_context = ProjectContext.from_user_cwd(user_cwd)

        config = ServerConfig.from_cli_args(
            project_context=project_context,
            model_name=None,
            model_params=None,
            assistant_id="agent",
            auto_approve=False,
            sandbox_type="none",
            sandbox_id=None,
            sandbox_setup=None,
            enable_shell=True,
            enable_ask_user=False,
            mcp_config_path="configs/mcp.json",
            no_mcp=False,
            trust_project_mcp=None,
            interactive=True,
        )

        with patch.dict(os.environ, {}, clear=False):
            for suffix in ("MCP_CONFIG_PATH", "CWD", "PROJECT_ROOT"):
                monkeypatch.delenv(f"{ENV_PREFIX}{suffix}", raising=False)

            _apply_server_config(config)

            assert os.environ[f"{ENV_PREFIX}MCP_CONFIG_PATH"] == str(
                (user_cwd / "configs" / "mcp.json").resolve()
            )
            assert os.environ[f"{ENV_PREFIX}CWD"] == str(user_cwd.resolve())
            assert os.environ[f"{ENV_PREFIX}PROJECT_ROOT"] == str(
                project_root.resolve()
            )


class TestStartServerAndGetAgent:
    """Tests for server bootstrap wiring."""

    async def test_uses_relative_graph_and_checkpointer_refs(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Generated LangGraph config should use relative paths (Windows compat)."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        monkeypatch.chdir(project_root)

        work_dir = tmp_path / "runtime"
        work_dir.mkdir()

        mock_server = MagicMock()
        mock_server.start = AsyncMock()
        mock_server.url = "http://127.0.0.1:2024"
        mock_agent = object()

        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "deepagents_cli.server_manager.tempfile.mkdtemp",
                return_value=str(work_dir),
            ),
            patch("deepagents_cli.server_manager.shutil.copy2"),
            patch("deepagents_cli.server_manager._write_checkpointer"),
            patch("deepagents_cli.server_manager._write_pyproject"),
            patch(
                "deepagents_cli.server.generate_langgraph_json"
            ) as mock_generate_langgraph_json,
            patch("deepagents_cli.server.ServerProcess", return_value=mock_server),
            patch("deepagents_cli.remote_client.RemoteAgent", return_value=mock_agent),
        ):
            agent, server, manager = await start_server_and_get_agent(
                assistant_id="agent",
                mcp_config_path=None,
            )

        assert agent is mock_agent
        assert server is mock_server
        assert manager is None

        kwargs = mock_generate_langgraph_json.call_args.kwargs
        assert kwargs["graph_ref"] == "./server_graph.py:graph"
        assert kwargs["checkpointer_path"] == "./checkpointer.py:create_checkpointer"

    def test_relative_paths_written_verbatim_to_langgraph_json(
        self, tmp_path: Path
    ) -> None:
        """Relative refs must appear verbatim in the generated config."""
        import json

        from deepagents_cli.server import generate_langgraph_json

        generate_langgraph_json(
            tmp_path,
            graph_ref="./server_graph.py:graph",
            checkpointer_path="./checkpointer.py:create_checkpointer",
        )
        config = json.loads((tmp_path / "langgraph.json").read_text())
        assert config["graphs"]["agent"] == "./server_graph.py:graph"
        assert config["checkpointer"]["path"] == "./checkpointer.py:create_checkpointer"


class TestWritePyproject:
    """Tests for the generated runtime pyproject."""

    def test_runtime_pyproject_relies_on_cli_dependency_only(
        self, tmp_path: Path
    ) -> None:
        """The runtime should inherit `langgraph-cli` from `deepagents-cli`."""
        _write_pyproject(tmp_path)

        content = (tmp_path / "pyproject.toml").read_text()

        assert '"deepagents-cli @ file://' in content
        assert "langgraph-cli[inmem]" not in content


class TestServerSession:
    """Tests for the server_session async context manager."""

    async def test_yields_agent_and_server(self) -> None:
        """server_session yields (agent, server_proc)."""
        mock_agent = MagicMock()
        mock_server = MagicMock()
        mock_server.stop = MagicMock()

        with patch(
            "deepagents_cli.server_manager.start_server_and_get_agent",
            new_callable=AsyncMock,
            return_value=(mock_agent, mock_server, None),
        ):
            async with server_session(assistant_id="agent") as (agent, server):
                assert agent is mock_agent
                assert server is mock_server

    async def test_stops_server_on_normal_exit(self) -> None:
        """Server is stopped when the context manager exits normally."""
        mock_server = MagicMock()
        mock_server.stop = MagicMock()

        with patch(
            "deepagents_cli.server_manager.start_server_and_get_agent",
            new_callable=AsyncMock,
            return_value=(MagicMock(), mock_server, None),
        ):
            async with server_session(assistant_id="agent"):
                pass

        mock_server.stop.assert_called_once()

    async def test_stops_server_on_exception(self) -> None:
        """Server is stopped even when body raises."""
        mock_server = MagicMock()
        mock_server.stop = MagicMock()

        with (  # noqa: PT012
            patch(
                "deepagents_cli.server_manager.start_server_and_get_agent",
                new_callable=AsyncMock,
                return_value=(MagicMock(), mock_server, None),
            ),
            pytest.raises(RuntimeError, match="boom"),
        ):
            async with server_session(assistant_id="agent"):
                msg = "boom"
                raise RuntimeError(msg)

        mock_server.stop.assert_called_once()

    async def test_cleans_up_mcp_session(self) -> None:
        """MCP session manager is cleaned up in finally block."""
        mock_server = MagicMock()
        mock_server.stop = MagicMock()
        mock_mcp = AsyncMock()

        with patch(
            "deepagents_cli.server_manager.start_server_and_get_agent",
            new_callable=AsyncMock,
            return_value=(MagicMock(), mock_server, mock_mcp),
        ):
            async with server_session(assistant_id="agent"):
                pass

        mock_mcp.cleanup.assert_awaited_once()
        mock_server.stop.assert_called_once()
