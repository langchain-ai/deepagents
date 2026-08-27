"""Tests for server manager bootstrap behavior."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, call, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from deepagents_code._env_vars import SERVER_ENV_PREFIX
from deepagents_code._server_config import ServerConfig
from deepagents_code.client.launch.server_manager import (
    _apply_server_config,
    _preflight_validate_mcp_config,
    _runtime_package_dependency,
    _write_pyproject,
    server_session,
    start_server_and_get_agent,
)
from deepagents_code.project_utils import ProjectContext


class TestServerConfigRoundTrip:
    """The env-var serialization contract between CLI and server graph."""

    def test_allow_fs_tools_list_round_trips(self) -> None:
        """An explicit allowlist survives the env round trip as a JSON list."""
        original = ServerConfig(allow_fs_tools=["ls", "read_file"])
        env_dict = original.to_env()
        with patch.dict(os.environ, {}, clear=True):
            for suffix, value in env_dict.items():
                if value is not None:
                    os.environ[f"{SERVER_ENV_PREFIX}{suffix}"] = value
            restored = ServerConfig.from_env()

        assert restored.allow_fs_tools == ["ls", "read_file"]

    def test_rejects_allow_fs_tools_without_read_file(self) -> None:
        """An explicit allowlist missing `read_file` fails at construction.

        `ServerConfig.__post_init__` owns this invariant so a tampered env value
        (which `_read_env_allow_fs_tools` intentionally does not check for
        `read_file`) fails closed here rather than a process boundary away in
        `FilesystemMiddleware`.
        """
        with pytest.raises(ValueError, match="allow_fs_tools must include"):
            ServerConfig(allow_fs_tools=["ls"])

    def test_from_env_rejects_invalid_allow_fs_tools_shape(self) -> None:
        """A tampered/skewed ALLOW_FS_TOOLS value fails closed rather than open.

        Well-formed JSON of an unexpected type must raise instead of falling
        through to an unrestricted filesystem — see `_read_env_allow_fs_tools`.
        Covers non-list scalars/objects, a list containing non-strings (the
        `all(isinstance(...))` guard), and the empty list (rejected directly so
        the fail-closed guarantee is self-contained, not SDK-dependent).
        """
        bad_values = (
            "null",  # explicit null is not the same as an absent variable
            '"all"',  # the "all" sentinel is collapsed to None before serialize
            '"read_file"',  # bare string, not a list
            "42",  # number
            "true",  # boolean
            "{}",  # object
            "[1, 2]",  # list of non-strings
            '["ls", null]',  # list with a null element
            "[]",  # empty list
        )
        for bad in bad_values:
            with (
                patch.dict(
                    os.environ,
                    {f"{SERVER_ENV_PREFIX}ALLOW_FS_TOOLS": bad},
                    clear=True,
                ),
                pytest.raises(ValueError, match="ALLOW_FS_TOOLS"),
            ):
                ServerConfig.from_env()


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
            sandbox_snapshot_name=None,
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
                monkeypatch.delenv(f"{SERVER_ENV_PREFIX}{suffix}", raising=False)

            _apply_server_config(config)

            assert os.environ[f"{SERVER_ENV_PREFIX}MCP_CONFIG_PATH"] == str(
                (user_cwd / "configs" / "mcp.json").resolve()
            )
            assert os.environ[f"{SERVER_ENV_PREFIX}CWD"] == str(user_cwd.resolve())
            assert os.environ[f"{SERVER_ENV_PREFIX}PROJECT_ROOT"] == str(
                project_root.resolve()
            )


class TestStartServerAndGetAgent:
    """Tests for server bootstrap wiring."""

    async def test_uses_package_graph_and_relative_checkpointer_refs(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Generated LangGraph config should import the package graph."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        monkeypatch.chdir(project_root)

        work_dir = tmp_path / "runtime"
        work_dir.mkdir()

        mock_server = MagicMock()
        mock_server.start = AsyncMock()
        mock_server.wait_for_graph_ready = AsyncMock()
        mock_server.url = "http://127.0.0.1:2024"
        mock_agent = object()

        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "deepagents_code.client.launch.server_manager.tempfile.mkdtemp",
                return_value=str(work_dir),
            ),
            patch("deepagents_code.client.launch.server_manager._write_checkpointer"),
            patch("deepagents_code.client.launch.server_manager._write_pyproject"),
            patch(
                "deepagents_code.client.launch.server.generate_langgraph_json"
            ) as mock_generate_langgraph_json,
            patch(
                "deepagents_code.client.launch.server.ServerProcess",
                return_value=mock_server,
            ),
            patch(
                "deepagents_code.client.remote_client.RemoteAgent",
                return_value=mock_agent,
            ),
        ):
            agent, server, manager = await start_server_and_get_agent(
                assistant_id="agent",
                mcp_config_path=None,
            )

        assert agent is mock_agent
        assert server is mock_server
        assert manager is None
        assert mock_server.wait_for_graph_ready.await_args_list == [call("agent")]

        kwargs = mock_generate_langgraph_json.call_args.kwargs
        assert kwargs["graph_ref"] == "deepagents_code.server_graph:make_graph"
        assert "additional_graphs" not in kwargs
        assert kwargs["checkpointer_path"] == "./checkpointer.py:create_checkpointer"

        # The graph is imported as a package module, so the scaffold must not
        # copy server_graph.py into the runtime workdir (a relic of the old
        # file-copy approach). `_scaffold_workspace` runs for real here — only
        # its file-writing helpers are patched — so a reintroduced copy would
        # surface as a stray file and fail this assertion.
        assert not (work_dir / "server_graph.py").exists()

    async def test_passes_scaffold_hook_to_server_process(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """ServerProcess should receive the scaffold hook for restart recovery."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        monkeypatch.chdir(project_root)

        work_dir = tmp_path / "runtime"
        work_dir.mkdir()

        mock_server = MagicMock()
        mock_server.start = AsyncMock()
        mock_server.wait_for_graph_ready = AsyncMock()
        mock_server.url = "http://127.0.0.1:2024"

        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "deepagents_code.client.launch.server_manager.tempfile.mkdtemp",
                return_value=str(work_dir),
            ),
            patch(
                "deepagents_code.client.launch.server_manager._scaffold_workspace"
            ) as mock_scaffold,
            patch(
                "deepagents_code.client.launch.server.ServerProcess",
                return_value=mock_server,
            ) as mock_server_process,
            patch(
                "deepagents_code.client.remote_client.RemoteAgent",
                return_value=object(),
            ),
        ):
            await start_server_and_get_agent(
                assistant_id="agent",
                mcp_config_path=None,
            )

        assert mock_server_process.call_args.kwargs["scaffold"] is mock_scaffold

    async def test_forwards_allow_fs_tools_into_server_config(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """`allow_fs_tools` reaches the `ServerConfig` written to the subprocess.

        The higher-level TUI/non-interactive forwarding tests mock this function
        out, so without this a dropped kwarg here would disable the feature for
        every server-backed session with no failing test.
        """
        project_root = tmp_path / "project"
        project_root.mkdir()
        monkeypatch.chdir(project_root)

        work_dir = tmp_path / "runtime"
        work_dir.mkdir()

        mock_server = MagicMock()
        mock_server.start = AsyncMock()
        mock_server.wait_for_graph_ready = AsyncMock()
        mock_server.url = "http://127.0.0.1:2024"

        captured: list[ServerConfig] = []

        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "deepagents_code.client.launch.server_manager.tempfile.mkdtemp",
                return_value=str(work_dir),
            ),
            patch("deepagents_code.client.launch.server_manager._write_checkpointer"),
            patch("deepagents_code.client.launch.server_manager._write_pyproject"),
            patch(
                "deepagents_code.client.launch.server_manager._apply_server_config",
                side_effect=captured.append,
            ),
            patch("deepagents_code.client.launch.server.generate_langgraph_json"),
            patch(
                "deepagents_code.client.launch.server.ServerProcess",
                return_value=mock_server,
            ),
            patch(
                "deepagents_code.client.remote_client.RemoteAgent",
                return_value=object(),
            ),
        ):
            await start_server_and_get_agent(
                assistant_id="agent",
                mcp_config_path=None,
                allow_fs_tools=["ls", "read_file"],
            )

        assert len(captured) == 1
        assert captured[0].allow_fs_tools == ["ls", "read_file"]

    async def test_stops_server_when_graph_readiness_fails(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Lazy graph initialization failures should fail startup before return."""
        project_root = tmp_path / "project"
        project_root.mkdir()
        monkeypatch.chdir(project_root)

        work_dir = tmp_path / "runtime"
        work_dir.mkdir()

        mock_server = MagicMock()
        mock_server.start = AsyncMock()
        mock_server.wait_for_graph_ready = AsyncMock(
            side_effect=RuntimeError("graph failed")
        )
        mock_server.stop = MagicMock()
        mock_server.url = "http://127.0.0.1:2024"

        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "deepagents_code.client.launch.server_manager.tempfile.mkdtemp",
                return_value=str(work_dir),
            ),
            patch("deepagents_code.client.launch.server_manager._write_checkpointer"),
            patch("deepagents_code.client.launch.server_manager._write_pyproject"),
            patch(
                "deepagents_code.client.launch.server.ServerProcess",
                return_value=mock_server,
            ),
            patch("deepagents_code.client.remote_client.RemoteAgent") as mock_agent,
            pytest.raises(RuntimeError, match="graph failed"),
        ):
            await start_server_and_get_agent(
                assistant_id="agent",
                mcp_config_path=None,
            )

        mock_server.start.assert_awaited_once()
        mock_server.wait_for_graph_ready.assert_awaited_once_with("agent")
        mock_server.stop.assert_called_once()
        mock_agent.assert_not_called()

    @pytest.mark.parametrize(
        "interrupt",
        [asyncio.CancelledError, KeyboardInterrupt, SystemExit],
    )
    async def test_stops_server_when_start_interrupted(
        self, interrupt: type[BaseException], tmp_path: Path, monkeypatch
    ) -> None:
        """A quit during startup must still reap the half-started server.

        The langgraph subprocess is spawned inside `ServerProcess.start()`
        before this function returns, so the caller has not yet stored a
        reference to it (`DeepAgentsApp._server_proc` is assigned only on
        successful return). When the background startup worker is interrupted
        mid-`start()` — e.g. the user presses Ctrl+D before the health check
        completes — this `except` clause is the only thing that can stop the
        orphaned subprocess. The interrupts covered here are all `BaseException`
        subclasses rather than `Exception`, so an `except Exception` guard would
        leak the process (regression: PR #4629).
        """
        project_root = tmp_path / "project"
        project_root.mkdir()
        monkeypatch.chdir(project_root)

        work_dir = tmp_path / "runtime"
        work_dir.mkdir()

        mock_server = MagicMock()
        mock_server.start = AsyncMock(side_effect=interrupt)
        mock_server.wait_for_graph_ready = AsyncMock()
        mock_server.stop = MagicMock()
        mock_server.url = "http://127.0.0.1:2024"

        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "deepagents_code.client.launch.server_manager.tempfile.mkdtemp",
                return_value=str(work_dir),
            ),
            patch("deepagents_code.client.launch.server_manager._write_checkpointer"),
            patch("deepagents_code.client.launch.server_manager._write_pyproject"),
            patch(
                "deepagents_code.client.launch.server.ServerProcess",
                return_value=mock_server,
            ),
            patch("deepagents_code.client.remote_client.RemoteAgent") as mock_agent,
            pytest.raises(interrupt),
        ):
            await start_server_and_get_agent(
                assistant_id="agent",
                mcp_config_path=None,
            )

        mock_server.start.assert_awaited_once()
        mock_server.stop.assert_called_once()
        # The interrupt must propagate: graph readiness is never reached, and
        # no client is handed back to a caller that is being torn down.
        mock_server.wait_for_graph_ready.assert_not_awaited()
        mock_agent.assert_not_called()

    async def test_start_cleanup_error_does_not_mask_interrupt(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A failure inside `stop()` must not replace the in-flight interrupt.

        Cleanup runs while a `BaseException` (here `CancelledError`) is
        propagating. If `stop()` itself raises, that error is swallowed and
        logged so the original cancellation stays the propagated exception,
        preserving cancellation semantics instead of surfacing the teardown
        error (regression: PR #4629).
        """
        project_root = tmp_path / "project"
        project_root.mkdir()
        monkeypatch.chdir(project_root)

        work_dir = tmp_path / "runtime"
        work_dir.mkdir()

        mock_server = MagicMock()
        mock_server.start = AsyncMock(side_effect=asyncio.CancelledError)
        mock_server.wait_for_graph_ready = AsyncMock()
        mock_server.stop = MagicMock(side_effect=RuntimeError("kill failed"))
        mock_server.url = "http://127.0.0.1:2024"

        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "deepagents_code.client.launch.server_manager.tempfile.mkdtemp",
                return_value=str(work_dir),
            ),
            patch("deepagents_code.client.launch.server_manager._write_checkpointer"),
            patch("deepagents_code.client.launch.server_manager._write_pyproject"),
            patch(
                "deepagents_code.client.launch.server.ServerProcess",
                return_value=mock_server,
            ),
            patch("deepagents_code.client.remote_client.RemoteAgent"),
            pytest.raises(asyncio.CancelledError),
        ):
            await start_server_and_get_agent(
                assistant_id="agent",
                mcp_config_path=None,
            )

        mock_server.stop.assert_called_once()

    def test_relative_paths_written_verbatim_to_langgraph_json(
        self, tmp_path: Path
    ) -> None:
        """Relative refs must appear verbatim in the generated config."""
        import json

        from deepagents_code.client.launch.server import generate_langgraph_json

        generate_langgraph_json(
            tmp_path,
            graph_ref="./server_graph.py:make_graph",
            checkpointer_path="./checkpointer.py:create_checkpointer",
        )
        config = json.loads((tmp_path / "langgraph.json").read_text())
        assert config["graphs"]["agent"] == "./server_graph.py:make_graph"
        assert config["checkpointer"]["path"] == "./checkpointer.py:create_checkpointer"

    def test_builtin_server_registers_only_the_agent_graph(
        self, tmp_path: Path
    ) -> None:
        """Operations use an authenticated route, not addressable siblings."""
        import json

        from deepagents_code.client.launch.server import generate_langgraph_json

        # The production default (see `server_manager`) resolves to the real
        # installed module.
        generate_langgraph_json(tmp_path)
        config = json.loads((tmp_path / "langgraph.json").read_text())
        assert config["graphs"] == {"agent": "deepagents_code.server_graph:make_graph"}
        assert config["http"] == {
            "app": "deepagents_code.offload_api:app",
            "enable_custom_route_auth": True,
        }

    async def test_forwards_agent_options_into_server_config(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Agent construction options reach the subprocess `ServerConfig`.

        The higher-level TUI/non-interactive forwarding tests mock this function
        out, so a dropped kwarg here would silently disable the option for every
        server-backed session.
        """
        project_root = tmp_path / "project"
        project_root.mkdir()
        monkeypatch.chdir(project_root)

        work_dir = tmp_path / "runtime"
        work_dir.mkdir()

        mock_server = MagicMock()
        mock_server.start = AsyncMock()
        mock_server.wait_for_graph_ready = AsyncMock()
        mock_server.url = "http://127.0.0.1:2024"

        captured: list[ServerConfig] = []

        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "deepagents_code.client.launch.server_manager.tempfile.mkdtemp",
                return_value=str(work_dir),
            ),
            patch("deepagents_code.client.launch.server_manager._write_checkpointer"),
            patch("deepagents_code.client.launch.server_manager._write_pyproject"),
            patch(
                "deepagents_code.client.launch.server_manager._apply_server_config",
                side_effect=captured.append,
            ),
            patch("deepagents_code.client.launch.server.generate_langgraph_json"),
            patch(
                "deepagents_code.client.launch.server.ServerProcess",
                return_value=mock_server,
            ),
            patch(
                "deepagents_code.client.remote_client.RemoteAgent",
                return_value=object(),
            ),
        ):
            await start_server_and_get_agent(
                assistant_id="agent",
                mcp_config_path=None,
                allow_fs_tools=["ls", "read_file"],
                summarization_model="openai:summary-model",
            )

        assert len(captured) == 1
        assert captured[0].allow_fs_tools == ["ls", "read_file"]
        assert captured[0].summarization_model == "openai:summary-model"


class TestWritePyproject:
    """Tests for the generated runtime pyproject."""

    def test_runtime_dependency_uses_source_checkout_dependency(
        self, tmp_path: Path
    ) -> None:
        """Source checkouts should keep using the local package path."""
        package_root = tmp_path / "package"
        package_root.mkdir()
        (package_root / "pyproject.toml").write_text("[project]\n")

        dependency = _runtime_package_dependency(package_root)

        assert dependency == f"deepagents-code @ {package_root.as_uri()}"

    def test_runtime_dependency_default_uses_package_project_root(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default root should not depend on `server_manager.py` depth."""
        from pathlib import Path

        import deepagents_code

        # Derive the expected project root independently, from this test file's
        # own location (libs/code/tests/unit_tests/ -> libs/code), rather than
        # reusing the implementation's package-anchored expression. Mirroring the
        # implementation would let a bug in that expression pass unnoticed.
        project_root = Path(__file__).resolve().parents[2]
        monkeypatch.setattr(
            deepagents_code,
            "__file__",
            str(project_root / "deepagents_code" / "__init__.py"),
        )

        dependency = _runtime_package_dependency()

        assert dependency == f"deepagents-code @ {project_root.as_uri()}"

    def test_runtime_pyproject_excludes_langgraph_cli_dependency(
        self, tmp_path: Path
    ) -> None:
        """The runtime project should rely on the app package dependency only."""
        with patch(
            "deepagents_code.client.launch.server_manager._runtime_package_dependency",
            return_value="deepagents-code==1.2.3",
        ):
            _write_pyproject(tmp_path)

        content = (tmp_path / "pyproject.toml").read_text()

        assert '"deepagents-code==1.2.3"' in content
        assert "langgraph-cli[inmem]" not in content


class TestServerSession:
    """Tests for the server_session async context manager."""

    async def test_forwards_summarization_model(self) -> None:
        """The context manager forwards the dedicated summary model.

        `start_server_and_get_agent` accepts `summarization_model`; a wrapper
        that drops it makes the option unreachable for `server_session`
        callers, and server startup is the only channel that configures
        server-owned `/offload` summaries.
        """
        mock_server = MagicMock()
        mock_server.stop = MagicMock()

        with patch(
            "deepagents_code.client.launch.server_manager.start_server_and_get_agent",
            new_callable=AsyncMock,
            return_value=(MagicMock(), mock_server, None),
        ) as start:
            async with server_session(
                assistant_id="agent",
                summarization_model="openai:summary-model",
            ):
                pass

        start.assert_awaited_once()
        await_args = start.await_args
        assert await_args is not None
        assert await_args.kwargs["summarization_model"] == "openai:summary-model"


class TestPreflightValidateMCPConfig:
    """Pre-flight validation of `--mcp-config` raises an actionable error."""
