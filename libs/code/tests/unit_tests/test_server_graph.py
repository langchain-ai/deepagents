"""Tests for server graph MCP loading behavior."""

from __future__ import annotations

import dataclasses
import importlib
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from blockbuster import blockbuster_ctx

from deepagents_code._env_vars import SERVER_ENV_PREFIX
from deepagents_code._server_config import ServerConfig

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _disable_extensions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep user extension code out of server graph unit tests."""
    monkeypatch.delenv("DEEPAGENTS_CODE_EXPERIMENTAL", raising=False)


def _import_fresh_server_graph() -> ModuleType:
    """Import `deepagents_code.server_graph` from a clean module state."""
    sys.modules.pop("deepagents_code.server_graph", None)
    return importlib.import_module("deepagents_code.server_graph")


def _module_with_attrs(name: str, **attrs: object) -> ModuleType:
    """Create a module stub with dynamically assigned attributes."""
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _backend_with_offload(default: object) -> SimpleNamespace:
    """Build a minimal backend carrying the server operation resource."""
    from deepagents_code.offload_middleware import OffloadOperation

    backend = SimpleNamespace(default=default)
    backend._dcode_offload_operation = OffloadOperation(MagicMock(), MagicMock())
    return backend


class TestServerGraph:
    """Tests for server-mode graph bootstrap."""

    async def test_make_graph_caches_first_constructed_graph(self) -> None:
        """Repeated factory access should preserve process-lifetime resources."""
        graph_obj = object()
        module = _import_fresh_server_graph()

        with patch.object(
            module,
            "_make_graphs",
            new=AsyncMock(
                return_value=module.ServerRuntime(graph_obj, object(), object())
            ),
        ) as make_graph:
            assert await module.make_graph() is graph_obj
            assert await module.make_graph() is graph_obj

        make_graph.assert_awaited_once_with()

    async def test_concurrent_resolution_builds_one_runtime(self) -> None:
        """Concurrent requests share the single graph runtime."""
        import asyncio

        module = _import_fresh_server_graph()
        graph_obj = object()
        calls = 0

        async def build() -> object:
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return module.ServerRuntime(graph_obj, object(), object())

        factory = module._build_graph_factory(build)
        results = await asyncio.gather(factory(), factory(), factory())

        assert calls == 1
        assert results == [graph_obj, graph_obj, graph_obj]

    def test_config_bootstrap_runs_off_the_blockbuster_loop(
        self, tmp_path: Path
    ) -> None:
        """Profile validation must not block the server event loop."""
        profile = tmp_path / "profile"
        profile.mkdir()
        env = os.environ.copy()
        env["DEEPAGENTS_HOME"] = str(profile)
        env.pop("DEEPAGENTS_HOME_IS_DEFAULT", None)
        code = """
import asyncio
from unittest.mock import AsyncMock, patch
from blockbuster import blockbuster_ctx
from deepagents_code._server_config import ServerConfig
import deepagents_code.server_graph as module

async def main():
    runtime = module.ServerRuntime(object(), object(), object())
    with patch.object(
        module,
        "_make_graphs_in_environment",
        new=AsyncMock(return_value=runtime),
    ):
        with blockbuster_ctx():
            assert await module._make_graphs(
                config_override=ServerConfig(no_mcp=True)
            ) is runtime

asyncio.run(main())
"""

        process = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

        assert process.returncode == 0, process.stderr

    def test_criteria_context_tools_use_identity_allowlist_in_tool_order(self) -> None:
        """Criteria tools should be known context objects in main-tool order."""
        module = _import_fresh_server_graph()
        from deepagents_code.tools import fetch_url, get_current_thread_id, web_search

        mcp_tool = SimpleNamespace(
            name="repository_search",
            metadata={"readOnlyHint": True, "destructiveHint": False},
        )
        mcp_lookalike = SimpleNamespace(name="repository_search")
        unknown_builtin = object()

        result = module._criteria_context_tools(
            [
                unknown_builtin,
                mcp_tool,
                get_current_thread_id,
                web_search,
                mcp_lookalike,
                fetch_url,
            ],
            [mcp_tool],
            [fetch_url, web_search],
        )

        assert len(result) == 3
        assert all(
            actual is expected
            for actual, expected in zip(
                result,
                [mcp_tool, web_search, fetch_url],
                strict=True,
            )
        )

    def test_criteria_context_tools_fail_closed_on_mcp_annotations(self) -> None:
        """Only unambiguously read-only MCP annotations grant criteria access."""
        from mcp.types import ToolAnnotations

        module = _import_fresh_server_graph()
        from deepagents_code.tools import fetch_url, web_search

        readonly_metadata = ToolAnnotations(readOnlyHint=True).model_dump()
        assert readonly_metadata["readOnlyHint"] is True
        readonly = SimpleNamespace(
            name="search",
            metadata=readonly_metadata,
        )
        mutating = SimpleNamespace(
            name="write",
            metadata={"readOnlyHint": False, "destructiveHint": True},
        )
        unannotated = SimpleNamespace(name="unknown", metadata=None)
        ambiguous = SimpleNamespace(
            name="contradictory",
            metadata={"readOnlyHint": True, "destructiveHint": True},
        )

        result = module._criteria_context_tools(
            [mutating, fetch_url, readonly, unannotated, web_search, ambiguous],
            [readonly, mutating, unannotated, ambiguous],
            [fetch_url, web_search],
        )

        assert result == [fetch_url, readonly, web_search]

    @pytest.mark.parametrize("read_only", [False, None, True])
    async def test_mcp_search_marker_cannot_bypass_read_only_gate(
        self, read_only: bool | None
    ) -> None:
        """Server-controlled annotation extras cannot grant criteria access."""
        from langchain_mcp_adapters.tools import convert_mcp_tool_to_langchain_tool
        from mcp.types import Tool, ToolAnnotations

        from deepagents_code.tools import fetch_url

        module = _import_fresh_server_graph()
        remote = convert_mcp_tool_to_langchain_tool(
            None,
            Tool(
                name="remote_tool",
                inputSchema={"type": "object", "properties": {}},
                annotations=ToolAnnotations.model_validate(
                    {
                        "readOnlyHint": read_only,
                        "destructiveHint": True,
                        "deepagents_web_search": True,
                    }
                ),
            ),
            connection={"transport": "stdio", "command": "unused", "args": []},
        )
        tools, _, _, read_only_builtins = await module._build_tools(
            ServerConfig(no_mcp=True), None, tavily_api_key=""
        )
        tools.append(remote)
        selected = module._criteria_context_tools(tools, [remote], read_only_builtins)

        assert remote not in selected
        assert fetch_url in selected
        assert any(getattr(tool, "name", None) == "web_search" for tool in selected)

    async def test_make_graph_emits_marker_and_exits_on_failure(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A construction failure must emit the startup marker, then exit non-zero."""
        from deepagents_code._startup_error import STARTUP_ERROR_MARKER

        module = _import_fresh_server_graph()

        with (
            patch.object(
                module,
                "_make_graphs",
                new=AsyncMock(side_effect=ValueError("boom: bad model")),
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            await module.make_graph()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert f"{STARTUP_ERROR_MARKER}ValueError: boom: bad model" in captured.err

    async def test_build_tools_binds_workspace_tavily_key(self) -> None:
        """Web search uses a workspace-specific tool instead of the singleton."""
        module = _import_fresh_server_graph()
        bound_tool = object()

        with patch(
            "deepagents_code.tools.create_web_search_tool",
            return_value=bound_tool,
        ) as create:
            tools, _, _, read_only_builtins = await module._build_tools(
                ServerConfig(no_mcp=True),
                None,
                tavily_api_key="workspace-key",
            )

        assert bound_tool in tools
        # The read-only allowlist is a security control, and the
        # `_criteria_context_tools` tests are handed it as an argument, so this
        # is the only place its contents are actually checked.
        from deepagents_code.tools import fetch_url, get_current_thread_id

        assert read_only_builtins == [fetch_url, bound_tool]
        assert get_current_thread_id not in read_only_builtins
        create.assert_called_once_with("workspace-key")

    async def test_build_tools_read_only_allowlist_without_web_search(self) -> None:
        """With no Tavily key the allowlist holds `fetch_url` alone."""
        module = _import_fresh_server_graph()
        from deepagents_code.tools import fetch_url, get_current_thread_id

        tools, _, mcp_tools, read_only_builtins = await module._build_tools(
            ServerConfig(no_mcp=True), None, tavily_api_key=None
        )
        selected = module._criteria_context_tools(tools, mcp_tools, read_only_builtins)

        assert selected == [fetch_url]
        assert get_current_thread_id in tools
        assert get_current_thread_id not in selected

    async def test_build_tools_skips_mcp_when_disabled(self) -> None:
        """`no_mcp=True` should not call the MCP resolver at all."""
        fetch_tool = object()
        thread_tool = object()
        resolve_mcp_tools = AsyncMock()
        config_module = _module_with_attrs(
            "deepagents_code.config",
            active_environment=dict,
            credentials=SimpleNamespace(has_tavily=False),
        )
        tools_module = _module_with_attrs(
            "deepagents_code.tools",
            create_web_search_tool=Mock(),
            fetch_url=fetch_tool,
            get_current_thread_id=thread_tool,
            web_search=object(),
        )
        mcp_module = _module_with_attrs(
            "deepagents_code.mcp_tools",
            resolve_and_load_mcp_tools=resolve_mcp_tools,
        )

        with patch.dict(
            sys.modules,
            {
                "deepagents_code.config": config_module,
                "deepagents_code.tools": tools_module,
                "deepagents_code.mcp_tools": mcp_module,
            },
        ):
            module = _import_fresh_server_graph()
            tools, mcp_server_info, mcp_tools, _ = await module._build_tools(
                ServerConfig(no_mcp=True),
                None,
                tavily_api_key=None,
            )

        assert tools == [fetch_tool, thread_tool]
        assert mcp_server_info is None
        assert mcp_tools == []
        resolve_mcp_tools.assert_not_awaited()

    async def test_interpreter_settings_apply_before_agent_construction(self) -> None:
        """Server PTC overrides should reach the interpreter snapshot."""
        from deepagents_code.config import _tracing_environment_values

        graph_obj = object()
        model_obj = object()
        observed: dict[str, object] = {}

        def create_cli_agent_side_effect(**kwargs: object) -> tuple[object, object]:
            from deepagents_code.configuration.interpreter import InterpreterConfig

            interpreter = kwargs["interpreter_config"]
            assert isinstance(interpreter, InterpreterConfig)
            observed["interpreter_ptc"] = interpreter.ptc
            observed["acknowledge"] = interpreter.ptc_acknowledge_unsafe
            observed["enable_interpreter"] = kwargs["enable_interpreter"]
            observed["auto_classifier_model"] = kwargs["auto_classifier_model"]
            return graph_obj, _backend_with_offload(object())

        settings_obj = SimpleNamespace(has_tavily=False, tavily_api_key=None)
        environment = dict(os.environ)
        config_module = _module_with_attrs(
            "deepagents_code.config",
            Credentials=SimpleNamespace(
                snapshot_from_environment=MagicMock(return_value=settings_obj)
            ),
            _ensure_bootstrap=MagicMock(),
            _preview_dotenv_environ=MagicMock(return_value=environment),
            active_environment=MagicMock(return_value=environment),
            use_environment=__import__("contextlib").nullcontext,
            _tracing_environment_values=_tracing_environment_values,
            is_langsmith_redaction_enabled=MagicMock(return_value=True),
            configure_langsmith_secret_redaction=MagicMock(),
            reconcile_tracing_environment=MagicMock(),
            create_model=MagicMock(
                return_value=SimpleNamespace(
                    model=model_obj,
                    provider="openai",
                    apply_to_runtime_state=MagicMock(),
                    model_retries=5,
                    cli_max_retries=None,
                ),
            ),
            is_memory_auto_save_enabled=MagicMock(return_value=True),
            resolve_auto_classifier_model_for_provider=MagicMock(
                return_value="openai:gpt-5.6-luna"
            ),
            credentials=settings_obj,
        )
        agent_module = _module_with_attrs(
            "deepagents_code.agent",
            create_cli_agent=MagicMock(side_effect=create_cli_agent_side_effect),
            load_async_subagents=MagicMock(return_value=None),
        )
        tools_module = _module_with_attrs(
            "deepagents_code.tools",
            create_web_search_tool=Mock(),
            fetch_url=object(),
            get_current_thread_id=object(),
            web_search=object(),
        )
        config = ServerConfig(
            no_mcp=True,
            enable_interpreter=True,
            interpreter_ptc=["js_eval"],
            interpreter_ptc_acknowledge_unsafe=True,
        )
        env_overrides = {
            f"{SERVER_ENV_PREFIX}{suffix}": value
            for suffix, value in config.to_env().items()
            if value is not None
        }

        with (
            patch.dict(os.environ, env_overrides, clear=False),
            patch.dict(
                sys.modules,
                {
                    "deepagents_code.agent": agent_module,
                    "deepagents_code.config": config_module,
                    "deepagents_code.tools": tools_module,
                },
            ),
            patch(
                "deepagents_code.project_utils.get_server_project_context",
                return_value=None,
            ),
        ):
            module = _import_fresh_server_graph()
            assert await module.make_graph() is graph_obj

        assert observed == {
            "interpreter_ptc": ["js_eval"],
            "acknowledge": True,
            "enable_interpreter": True,
            "auto_classifier_model": "openai:gpt-5.6-luna",
        }


class TestWorkspaceEnvironmentBinding:
    """The workspace snapshot must actually be bound around construction.

    Every other `_make_graphs` test stubs `deepagents_code.config`, including
    `use_environment`, while each consumer test patches `active_environment`
    directly. Both ends are mocked, so nothing exercises the wire between them:
    dropping the `with use_environment(...)` block leaves `active_environment()`
    falling back to `os.environ` with no error and no failing test, and sandbox
    setup would expand the server's own secrets.
    """

    async def test_consumers_read_the_workspace_env_during_construction(
        self, tmp_path, monkeypatch
    ) -> None:
        """The real config module binds the workspace `.env` for consumers."""
        import deepagents_code.agent as agent_mod
        import deepagents_code.config as config_mod
        import deepagents_code.integrations.sandbox_factory as sandbox_mod

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / ".env").write_text(
            "WORKSPACE_ONLY=from-workspace-dotenv\n", encoding="utf-8"
        )
        monkeypatch.delenv("WORKSPACE_ONLY", raising=False)
        monkeypatch.setenv("SERVER_ONLY", "from-server-process")
        monkeypatch.setattr(
            config_mod, "_GLOBAL_DOTENV_PATH", tmp_path / "missing-global.env"
        )

        seen: dict[str, object] = {}

        def _record(label: str) -> None:
            environment = config_mod.active_environment()
            seen[label] = environment.get("WORKSPACE_ONLY")
            seen[f"{label}_server_leak"] = environment.get("SERVER_ONLY")

        graph_obj = object()

        def _create_cli_agent(**_kwargs: object) -> tuple[object, object]:
            _record("agent")
            return graph_obj, _backend_with_offload(object())

        def _create_model(*_args: object, **_kwargs: object) -> object:
            _record("model")
            return SimpleNamespace(
                model=object(),
                provider="openai",
                apply_to_runtime_state=lambda: None,
                model_retries=5,
                cli_max_retries=None,
            )

        def _create_sandbox(*_args: object, **_kwargs: object) -> object:
            _record("sandbox")
            return MagicMock()

        module = _import_fresh_server_graph()
        config = ServerConfig(
            no_mcp=True,
            cwd=str(workspace),
            project_root=str(workspace),
            sandbox_type="vercel",
        )

        with (
            patch.object(agent_mod, "create_cli_agent", _create_cli_agent),
            patch.object(agent_mod, "load_async_subagents", lambda **_: None),
            patch.object(config_mod, "create_model", _create_model),
            patch.object(sandbox_mod, "create_sandbox", _create_sandbox),
            patch.object(
                module, "_criteria_context_tools", lambda *_args, **_kwargs: []
            ),
        ):
            runtime = await module._make_graphs(config_override=config)

        assert runtime.agent is graph_obj
        # Each consumer read the workspace `.env`, not the server process env.
        assert seen["model"] == "from-workspace-dotenv"
        assert seen["agent"] == "from-workspace-dotenv"
        assert seen["sandbox"] == "from-workspace-dotenv"
        # The server's own environment still shows through where unshadowed.
        assert seen["agent_server_leak"] == "from-server-process"
        # And the workspace value never reached the process.
        assert "WORKSPACE_ONLY" not in os.environ


def _bind(config: ServerConfig, cwd: Any) -> Any:  # noqa: ANN401
    """Resolve a workspace binding for `cwd`, creating the directory first."""
    from deepagents_code.workspace import resolve_workspace

    cwd.mkdir(exist_ok=True)
    identity = resolve_workspace(str(cwd))
    resolved = config.resolve_workspace(identity.cwd, identity.project_root)
    return resolve_workspace(
        identity.cwd,
        resolved.to_workspace_payload(),
        config_fingerprint=resolved.workspace_fingerprint(),
    )


class TestWorkspaceRuntime:
    """Workspace runtimes retain trusted server-only configuration."""

    async def test_cached_runtime_resolves_policy_off_event_loop(
        self, tmp_path
    ) -> None:
        module = _import_fresh_server_graph()
        config = ServerConfig()
        binding = _bind(config, tmp_path)
        runtime = module.ServerRuntime(object(), object(), object())
        module._remember_workspace_runtime(binding, runtime)

        with (
            patch.object(ServerConfig, "from_env", return_value=config),
            blockbuster_ctx(scanned_modules=module),
        ):
            assert await module._workspace_runtime(binding) is runtime

    async def test_uses_full_server_config_and_replaces_only_workspace_paths(
        self, tmp_path
    ) -> None:
        module = _import_fresh_server_graph()
        bound_config = ServerConfig(
            model="trusted:model",
            system_prompt="trusted prompt",
            model_params={"api_key": "secret"},
            auto_approve=True,
        )
        binding = _bind(bound_config, tmp_path)
        runtime = module.ServerRuntime(object(), object(), object())
        with (
            patch.object(ServerConfig, "from_env", return_value=bound_config),
            patch.object(
                module, "_make_graphs", new=AsyncMock(return_value=runtime)
            ) as make,
        ):
            assert await module._workspace_runtime(binding) is runtime

        call = make.await_args
        assert call is not None
        config = call.kwargs["config_override"]
        assert config.model == "trusted:model"
        assert config.system_prompt == "trusted prompt"
        assert config.model_params == {"api_key": "secret"}
        assert config.cwd == binding.cwd
        assert config.project_root == binding.project_root

    async def test_readiness_runtime_owns_sandbox_workspace(self, tmp_path) -> None:
        """The startup runtime must reserve its sandbox for the launch workspace."""
        from deepagents_code.workspace import WorkspaceConflictError

        module = _import_fresh_server_graph()
        launch_dir = tmp_path / "launch"
        config = ServerConfig(sandbox_type="daytona", cwd=str(launch_dir))
        launch = _bind(config, launch_dir)
        other = _bind(config, tmp_path / "other")
        readiness_runtime = module.ServerRuntime(object(), object(), object())
        make = AsyncMock(return_value=readiness_runtime)

        with (
            patch.object(ServerConfig, "from_env", return_value=config),
            patch.object(module, "_make_graphs", new=make),
        ):
            assert await module.get_server_runtime() is readiness_runtime
            assert await module._workspace_runtime(launch) is readiness_runtime
            with pytest.raises(WorkspaceConflictError, match="another workspace"):
                await module._workspace_runtime(other)

        make.assert_awaited_once_with()

    async def test_sandbox_refuses_second_workspace_and_keeps_first(
        self, tmp_path
    ) -> None:
        from deepagents_code.workspace import WorkspaceConflictError

        module = _import_fresh_server_graph()
        config = ServerConfig(sandbox_type="daytona")
        first = _bind(config, tmp_path / "first")
        second = _bind(config, tmp_path / "second")
        first_runtime = module.ServerRuntime(object(), object(), object())
        make = AsyncMock(return_value=first_runtime)

        with (
            patch.object(ServerConfig, "from_env", return_value=config),
            patch.object(module, "_make_graphs", new=make),
        ):
            assert await module._workspace_runtime(first) is first_runtime
            with pytest.raises(
                WorkspaceConflictError,
                match=(
                    "Cannot host this workspace because a runtime for another "
                    "workspace already exists and the configured sandbox is "
                    "process-wide"
                ),
            ):
                await module._workspace_runtime(second)
            assert await module._workspace_runtime(first) is first_runtime

        make.assert_awaited_once()

    async def test_failed_sandbox_runtime_keeps_workspace_ownership(
        self, tmp_path
    ) -> None:
        """A failed build must not let another workspace claim the sandbox."""
        from deepagents_code.workspace import WorkspaceConflictError

        module = _import_fresh_server_graph()
        config = ServerConfig(sandbox_type="daytona")
        first = _bind(config, tmp_path / "first")
        second = _bind(config, tmp_path / "second")
        first_runtime = module.ServerRuntime(object(), object(), object())
        make = AsyncMock(side_effect=[SystemExit(1), first_runtime])

        with (
            patch.object(ServerConfig, "from_env", return_value=config),
            patch.object(module, "_make_graphs", new=make),
        ):
            with pytest.raises(SystemExit):
                await module._workspace_runtime(first)
            with pytest.raises(WorkspaceConflictError, match="another workspace"):
                await module._workspace_runtime(second)
            assert await module._workspace_runtime(first) is first_runtime

        assert make.await_count == 2

    async def test_without_sandbox_builds_second_workspace(self, tmp_path) -> None:
        module = _import_fresh_server_graph()
        config = ServerConfig()
        bindings = [_bind(config, tmp_path / name) for name in ("first", "second")]
        runtimes = [
            module.ServerRuntime(object(), object(), object()),
            module.ServerRuntime(object(), object(), object()),
        ]
        make = AsyncMock(side_effect=runtimes)

        with (
            patch.object(ServerConfig, "from_env", return_value=config),
            patch.object(module, "_make_graphs", new=make),
        ):
            assert [
                await module._workspace_runtime(binding) for binding in bindings
            ] == (runtimes)

        assert make.await_count == 2

    async def test_rejects_resolved_project_policy_divergence(self, tmp_path) -> None:
        from deepagents_code.workspace import WorkspaceConflictError

        module = _import_fresh_server_graph()
        project = tmp_path / "project"
        bound_config = ServerConfig(
            cwd=str(project),
            project_root=str(project),
            trust_project_mcp=False,
        )
        binding = _bind(bound_config, project)
        changed = ServerConfig(
            cwd=str(project),
            project_root=str(project),
            trust_project_mcp=True,
        )
        with (
            patch.object(ServerConfig, "from_env", return_value=changed),
            patch.object(module, "_make_graphs", new=AsyncMock()) as make,
            pytest.raises(WorkspaceConflictError, match="project's resolved policy"),
        ):
            await module._workspace_runtime(binding)

        make.assert_not_awaited()

    async def test_cached_runtime_rejects_project_policy_divergence(
        self, tmp_path
    ) -> None:
        from deepagents_code.workspace import WorkspaceConflictError

        module = _import_fresh_server_graph()
        project = tmp_path / "project"
        bound_config = ServerConfig(
            cwd=str(project),
            project_root=str(project),
            trust_project_mcp=True,
        )
        binding = _bind(bound_config, project)
        runtime = module.ServerRuntime(object(), object(), object())
        make = AsyncMock(return_value=runtime)

        with (
            patch.object(ServerConfig, "from_env", return_value=bound_config),
            patch.object(module, "_make_graphs", new=make),
        ):
            assert await module._workspace_runtime(binding) is runtime

        changed = dataclasses.replace(bound_config, trust_project_mcp=False)
        with (
            patch.object(ServerConfig, "from_env", return_value=changed),
            pytest.raises(WorkspaceConflictError, match="project's resolved policy"),
        ):
            await module._workspace_runtime(binding)

        make.assert_awaited_once()

    async def test_cached_runtime_rejects_revoked_extension_trust(
        self, tmp_path
    ) -> None:
        """Revoking trust must invalidate a runtime built while it was granted.

        `trust_project_extensions` is the one project policy field not derived
        from the environment: `resolve_workspace` re-reads it from the
        persisted trust store. `ServerConfig.from_env()` is identical across
        both calls here, so the server-config fingerprint check cannot fire and
        only the project-policy comparison can catch the revocation. Without
        it, a runtime keeps executing project Python the user has untrusted.
        """
        from deepagents_code.workspace import WorkspaceConflictError

        module = _import_fresh_server_graph()
        launch = tmp_path / "launch"
        other = tmp_path / "other"
        launch.mkdir()
        other.mkdir()
        launch_config = ServerConfig(cwd=str(launch), project_root=str(launch))
        runtime = module.ServerRuntime(object(), object(), object())
        make = AsyncMock(return_value=runtime)
        trust = "deepagents_code.extensions.trust.is_project_extensions_trusted"

        with patch(trust, return_value=True):
            binding = _bind(launch_config, other)

        with (
            patch(trust, return_value=True),
            patch.object(ServerConfig, "from_env", return_value=launch_config),
            patch.object(module, "_make_graphs", new=make),
        ):
            assert await module._workspace_runtime(binding) is runtime

        # The user revokes trust. Nothing about the environment changes.
        with (
            patch(trust, return_value=False),
            patch.object(ServerConfig, "from_env", return_value=launch_config),
            pytest.raises(
                WorkspaceConflictError,
                match=r"project's resolved policy.*\(trust_project_extensions\)",
            ),
        ):
            await module._workspace_runtime(binding)

        make.assert_awaited_once()

    async def test_second_project_runtime_is_built_without_launch_grants(
        self, tmp_path
    ) -> None:
        """Second-project runtimes drop launch grants and retain session policy."""
        module = _import_fresh_server_graph()
        launch = tmp_path / "launch"
        other = tmp_path / "other"
        launch.mkdir()
        other.mkdir()
        launch_config = ServerConfig(
            cwd=str(launch),
            project_root=str(launch),
            mcp_config_path="/launch/.mcp.json",
            sandbox_setup="/launch/setup.sh",
            trust_project_mcp=True,
            trust_project_extensions=True,
            extension_paths=("/launch/ext.py",),
            no_mcp=True,
            auto_approve=True,
            allow_fs_tools=["read_file"],
        )
        runtime = module.ServerRuntime(object(), object(), object())
        make = AsyncMock(return_value=runtime)
        trust = "deepagents_code.extensions.trust.is_project_extensions_trusted"

        with patch(trust, return_value=False):
            binding = _bind(launch_config, other)

        with (
            patch(trust, return_value=False),
            patch.object(ServerConfig, "from_env", return_value=launch_config),
            patch.object(module, "_make_graphs", new=make),
        ):
            assert await module._workspace_runtime(binding) is runtime

        call = make.await_args
        assert call is not None
        built = call.kwargs["config_override"]
        assert built.mcp_config_path is None
        assert built.sandbox_setup is None
        assert built.trust_project_mcp is None
        assert built.extension_paths == ()
        assert built.trust_project_extensions is False
        # Session policy belongs to the command, not the project, and survives.
        assert built.no_mcp is True
        assert built.auto_approve is True
        assert built.allow_fs_tools == ["read_file"]
        assert built.cwd == binding.cwd

    async def test_launch_binding_uses_the_explicit_server_project_root(
        self, tmp_path
    ) -> None:
        """The binding must agree with the config `_get_runtime()` builds from.

        `resolve_workspace` derives the project root with `find_project_root`,
        but `get_server_project_context` prefers an explicit
        `DEEPAGENTS_CODE_SERVER_PROJECT_ROOT`. Where they disagreed, the launch
        binding recorded scrubbed project policy while the process-wide runtime
        kept the launch project's MCP servers and extensions live.
        """
        module = _import_fresh_server_graph()
        # `explicit` is not a project root by discovery, so `find_project_root`
        # cannot return it -- only the explicit setting can.
        workdir = tmp_path / "workdir"
        explicit = tmp_path / "explicit"
        workdir.mkdir()
        explicit.mkdir()
        config = ServerConfig(
            cwd=str(workdir),
            project_root=str(explicit),
            mcp_config_path="/launch/.mcp.json",
            sandbox_setup="/launch/setup.sh",
        )

        with blockbuster_ctx(scanned_modules=module):
            binding = await module._default_workspace_binding(config)

        assert binding is not None
        policy = binding.workspace_config()
        assert policy["mcp_config_path"] == "/launch/.mcp.json"
        assert policy["sandbox_setup"] == "/launch/setup.sh"
        assert binding.config_fingerprint == config.workspace_fingerprint()

    async def test_cached_runtime_survives_a_granted_extension_trust(
        self, tmp_path
    ) -> None:
        """Granting trust elsewhere must not brick an already-bound thread.

        Trust is resolved from a mutable on-disk store, so a grant in another
        session looked exactly like drift -- and because the check runs before
        the cache lookup, every request on the thread refused with no way to
        recover. A grant only adds privilege, so the bound value is pinned and
        takes effect on the next binding instead.
        """
        module = _import_fresh_server_graph()
        launch = tmp_path / "launch"
        other = tmp_path / "other"
        launch.mkdir()
        other.mkdir()
        launch_config = ServerConfig(cwd=str(launch), project_root=str(launch))
        runtime = module.ServerRuntime(object(), object(), object())
        make = AsyncMock(return_value=runtime)
        trust = "deepagents_code.extensions.trust.is_project_extensions_trusted"

        with patch(trust, return_value=False):
            binding = _bind(launch_config, other)

        with (
            patch(trust, return_value=False),
            patch.object(ServerConfig, "from_env", return_value=launch_config),
            patch.object(module, "_make_graphs", new=make),
        ):
            assert await module._workspace_runtime(binding) is runtime

        # The user grants trust for this project in another session.
        with (
            patch(trust, return_value=True),
            patch.object(ServerConfig, "from_env", return_value=launch_config),
            patch.object(module, "_make_graphs", new=make),
        ):
            assert await module._workspace_runtime(binding) is runtime

        # The thread keeps the trust it was bound with, not the new grant.
        make.assert_awaited_once()
        call = make.await_args
        assert call is not None
        assert call.kwargs["config_override"].trust_project_extensions is False

    async def test_rejects_server_config_drift(self, tmp_path) -> None:
        from deepagents_code.workspace import WorkspaceConflictError

        module = _import_fresh_server_graph()
        bound_config = ServerConfig(model="trusted:model")
        binding = _bind(bound_config, tmp_path)
        with (
            patch.object(
                ServerConfig,
                "from_env",
                return_value=ServerConfig(model="changed:model"),
            ),
            patch.object(module, "_make_graphs", new=AsyncMock()) as make,
            pytest.raises(WorkspaceConflictError, match="configuration changed"),
        ):
            await module._workspace_runtime(binding)

        make.assert_not_awaited()

    async def test_unusable_launch_cwd_emits_startup_marker(
        self, tmp_path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Resolving the launch binding runs outside `_get_runtime`'s barrier.

        A launch cwd that cannot be canonicalized must still produce the marker
        the parent app process scrapes, not a bare `ValueError`.
        """
        from deepagents_code._startup_error import STARTUP_ERROR_MARKER

        module = _import_fresh_server_graph()
        missing = tmp_path / "gone"
        config = ServerConfig(cwd=str(missing))

        with (
            patch.object(ServerConfig, "from_env", return_value=config),
            patch.object(module, "_make_graphs", new=AsyncMock()) as make,
            pytest.raises(SystemExit) as exc_info,
        ):
            await module.get_server_runtime()

        assert exc_info.value.code == 1
        assert STARTUP_ERROR_MARKER in capsys.readouterr().err
        make.assert_not_awaited()


class TestStartupErrorMarker:
    """`emit_startup_failure` must produce the parser marker on stderr.

    The marker is the contract `wait_for_server_healthy` parses to surface
    a one-line summary instead of "Server process exited with code N".
    """


class TestGraphFactorySignature:
    """`make_graph` must stay loadable as a LangGraph server graph factory.

    The server does not call the factory to learn what it wants. It resolves
    the factory's annotations with `typing.get_type_hints` at graph-load time
    and builds a keyword dispatch from them. An annotation that names a symbol
    which exists only for type checkers fails to resolve, and the server then
    rejects the graph before it serves a request. Calling `make_graph`
    directly cannot detect this, because Python never evaluates annotations.
    """

    def test_factory_annotations_resolve_at_runtime(self) -> None:
        """Every `make_graph` annotation must resolve outside TYPE_CHECKING."""
        import typing

        module = _import_fresh_server_graph()

        hints = typing.get_type_hints(module.make_graph)

        assert "runtime" in hints
        assert "config" in hints

    def test_server_classifies_factory_as_config_and_runtime(self) -> None:
        """The server must map both parameters, not reject the factory."""
        from langgraph_api._factory_utils import _classify_factory

        module = _import_fresh_server_graph()

        # `_classify_factory` returns the keyword dispatch the server uses for
        # every graph load. The public `classify_factory` caches into a process
        # global, so it is deliberately not used here.
        dispatch = _classify_factory(module.make_graph)

        assert dispatch is not None
        # Sentinels: the dispatch only routes these values by keyword.
        config: Any = object()
        runtime: Any = object()
        assert dispatch(config, runtime) == {"config": config, "runtime": runtime}
