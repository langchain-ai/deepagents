"""Tests for server graph MCP loading behavior."""

from __future__ import annotations

import importlib
import os
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from blockbuster import BlockBuster

from deepagents_code._env_vars import SERVER_ENV_PREFIX
from deepagents_code._server_config import ServerConfig
from deepagents_code.integrations import sandbox_factory


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
        )

        assert result == [fetch_url, readonly, web_search]

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

    async def test_build_tools_skips_mcp_when_disabled(self) -> None:
        """`no_mcp=True` should not call the MCP resolver at all."""
        fetch_tool = object()
        thread_tool = object()
        resolve_mcp_tools = AsyncMock()
        config_module = _module_with_attrs(
            "deepagents_code.config",
            credentials=SimpleNamespace(has_tavily=False),
        )
        tools_module = _module_with_attrs(
            "deepagents_code.tools",
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
            tools, mcp_server_info, mcp_tools = await module._build_tools(
                ServerConfig(no_mcp=True),
                None,
            )

        assert tools == [fetch_tool, thread_tool]
        assert mcp_server_info is None
        assert mcp_tools == []
        resolve_mcp_tools.assert_not_awaited()

    async def test_interpreter_settings_apply_before_agent_construction(self) -> None:
        """Server PTC overrides should reach the interpreter snapshot."""
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
            return graph_obj, _backend_with_offload(object())

        settings_obj = SimpleNamespace(has_tavily=False)
        config_module = _module_with_attrs(
            "deepagents_code.config",
            configure_langsmith_secret_redaction=MagicMock(),
            create_model=MagicMock(
                return_value=SimpleNamespace(
                    model=model_obj,
                    apply_to_runtime_state=MagicMock(),
                    model_retries=5,
                    cli_max_retries=None,
                ),
            ),
            is_memory_auto_save_enabled=MagicMock(return_value=True),
            credentials=settings_obj,
        )
        agent_module = _module_with_attrs(
            "deepagents_code.agent",
            create_cli_agent=MagicMock(side_effect=create_cli_agent_side_effect),
            load_async_subagents=MagicMock(return_value=None),
        )
        tools_module = _module_with_attrs(
            "deepagents_code.tools",
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
        }

    async def test_sandbox_creation_does_not_trip_blockbuster_guard(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Sandbox creation must not run sync blocking I/O on the event loop.

        `langgraph dev` arms the blockbuster guard
        (`langgraph_runtime_inmem/queue.py` -> `_enable_blockbuster`), which
        raises `BlockingError` when a patched blocking call runs on the
        asyncio loop. `_make_graphs` creates the sandbox synchronously, so
        `dcode --sandbox <provider>` fails the server readiness check with
        "Blocking call to socket.socket.connect" (reproduced on langsmith,
        agentcore, and daytona). This test pins the desired behavior: the
        provider's sync `get_or_create` must not run directly on the loop.
        """
        graph_obj = object()
        model_obj = object()

        def create_cli_agent_side_effect(**_kwargs: object) -> tuple[object, object]:
            return graph_obj, _backend_with_offload(object())

        # The sync sandbox SDKs (langsmith `SandboxClient`, daytona, ...) do
        # real blocking I/O such as socket connects. Model that with a sleep:
        # blockbuster flags it identically on the event loop, and it stays
        # deterministic under pytest-socket's `--disable-socket`.
        def blocking_get_or_create(**_kwargs: object) -> object:
            time.sleep(0.001)
            return MagicMock()

        provider = MagicMock()
        provider.get_or_create.side_effect = blocking_get_or_create
        registry = MagicMock()
        registry.get_metadata.return_value = None
        registry.get_params.return_value = {}

        settings_obj = SimpleNamespace(has_tavily=False)
        config_module = _module_with_attrs(
            "deepagents_code.config",
            configure_langsmith_secret_redaction=MagicMock(),
            create_model=MagicMock(
                return_value=SimpleNamespace(
                    model=model_obj,
                    apply_to_runtime_state=MagicMock(),
                    model_retries=5,
                    cli_max_retries=None,
                ),
            ),
            is_memory_auto_save_enabled=MagicMock(return_value=False),
            credentials=settings_obj,
        )
        agent_module = _module_with_attrs(
            "deepagents_code.agent",
            create_cli_agent=MagicMock(side_effect=create_cli_agent_side_effect),
            load_async_subagents=MagicMock(return_value=None),
        )
        tools_module = _module_with_attrs(
            "deepagents_code.tools",
            fetch_url=object(),
            get_current_thread_id=object(),
            web_search=object(),
        )
        config = ServerConfig(no_mcp=True, sandbox_type="langsmith")
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
            patch.object(
                sandbox_factory,
                "_get_provider",
                return_value=provider,
            ),
            patch.object(
                sandbox_factory,
                "_get_registry",
                return_value=registry,
            ),
        ):
            module = _import_fresh_server_graph()
            bb = BlockBuster()
            bb.activate()
            try:
                try:
                    result = await module.make_graph()
                except SystemExit as exc:
                    captured = capsys.readouterr()
                    pytest.fail(
                        "sandbox creation tripped the blockbuster "
                        f"blocking-I/O guard: SystemExit({exc.code}) -- "
                        f"startup error: {captured.err}"
                    )
            finally:
                # Explicit activate/deactivate rather than `blockbuster_ctx`:
                # blockbuster <1.5.27 lacks the try/finally in that helper, so
                # the guard leaks into later tests when the body raises.
                bb.deactivate()

        assert result is graph_obj


class TestWorkspaceRuntime:
    """Workspace runtimes retain trusted server-only configuration."""

    async def test_uses_full_server_config_and_replaces_only_workspace_paths(
        self, tmp_path
    ) -> None:
        from deepagents_code.workspace import resolve_workspace

        module = _import_fresh_server_graph()
        bound_config = ServerConfig(
            model="trusted:model",
            system_prompt="trusted prompt",
            model_params={"api_key": "secret"},
            auto_approve=True,
        )
        binding = resolve_workspace(
            str(tmp_path),
            bound_config.to_workspace_payload(),
            config_fingerprint=bound_config.workspace_fingerprint(),
        )
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

    async def test_rejects_server_config_drift(self, tmp_path) -> None:
        from deepagents_code.workspace import resolve_workspace

        module = _import_fresh_server_graph()
        bound_config = ServerConfig(model="trusted:model")
        binding = resolve_workspace(
            str(tmp_path),
            bound_config.to_workspace_payload(),
            config_fingerprint=bound_config.workspace_fingerprint(),
        )
        with (
            patch.object(
                ServerConfig,
                "from_env",
                return_value=ServerConfig(model="changed:model"),
            ),
            patch.object(module, "_make_graphs", new=AsyncMock()) as make,
            pytest.raises(RuntimeError, match="configuration changed"),
        ):
            await module._workspace_runtime(binding)

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
