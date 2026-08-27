"""Tests for server graph MCP loading behavior."""

from __future__ import annotations

import importlib
import os
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from deepagents_code._env_vars import SERVER_ENV_PREFIX
from deepagents_code._server_config import ServerConfig


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

    async def test_auto_discovery_loads_mcp_without_explicit_config(self) -> None:
        """Server mode should auto-discover MCP configs when the graph is built."""
        graph_obj = object()
        model_obj = object()
        fetch_tool = object()
        thread_tool = object()
        web_tool = object()
        mcp_tool = SimpleNamespace(
            metadata={"readOnlyHint": True, "destructiveHint": False}
        )
        mcp_server_info = [SimpleNamespace(name="docs")]
        loop_thread_id = threading.get_ident()
        create_cli_agent_thread_ids: list[int] = []
        create_model_thread_ids: list[int] = []
        setup_thread_ids: list[int] = []
        redaction_thread_ids: list[int] = []
        repository_backend = object()
        user_cwd = object()
        project_context = SimpleNamespace(user_cwd=user_cwd, project_root=object())
        reload_from_environment = MagicMock()

        def create_cli_agent_side_effect(**_: object) -> tuple[object, object]:
            create_cli_agent_thread_ids.append(threading.get_ident())
            return graph_obj, _backend_with_offload(repository_backend)

        def create_model_side_effect(*_: object, **__: object) -> object:
            create_model_thread_ids.append(threading.get_ident())
            return model_result

        def get_server_project_context_side_effect(*_: object, **__: object) -> object:
            setup_thread_ids.append(threading.get_ident())
            return project_context

        def reload_from_environment_side_effect(
            *_: object, **kwargs: object
        ) -> list[str]:
            setup_thread_ids.append(threading.get_ident())
            assert kwargs.get("start_path") is user_cwd
            return []

        def configure_redaction_side_effect() -> None:
            redaction_thread_ids.append(threading.get_ident())

        reload_from_environment.side_effect = reload_from_environment_side_effect

        create_cli_agent = MagicMock(side_effect=create_cli_agent_side_effect)
        agent_module = _module_with_attrs(
            "deepagents_code.agent",
            DEFAULT_AGENT_NAME="agent",
            create_cli_agent=create_cli_agent,
            load_async_subagents=MagicMock(return_value=None),
        )

        model_result = SimpleNamespace(
            model=model_obj,
            apply_to_runtime_state=MagicMock(),
            model_retries=5,
            cli_max_retries=3,
        )
        configure_redaction = MagicMock(side_effect=configure_redaction_side_effect)
        create_model = MagicMock(side_effect=create_model_side_effect)
        config_module = _module_with_attrs(
            "deepagents_code.config",
            configure_langsmith_secret_redaction=configure_redaction,
            create_model=create_model,
            is_memory_auto_save_enabled=MagicMock(return_value=True),
            credentials=SimpleNamespace(
                has_tavily=True,
                reload_from_environment=reload_from_environment,
            ),
        )

        tools_module = _module_with_attrs(
            "deepagents_code.tools",
            fetch_url=fetch_tool,
            get_current_thread_id=thread_tool,
            web_search=web_tool,
        )

        class FakeSessionManager:
            async def cleanup(self) -> None:
                return None

        resolve_mcp_tools = AsyncMock(return_value=([mcp_tool], None, mcp_server_info))
        mcp_module = _module_with_attrs(
            "deepagents_code.mcp_tools",
            MCPSessionManager=FakeSessionManager,
            resolve_and_load_mcp_tools=resolve_mcp_tools,
        )

        config = ServerConfig(
            no_mcp=False,
            profile_overrides={"max_input_tokens": 32000},
            # Non-default allowlist so the `fs_tools=` assertion below is
            # load-bearing: it round-trips through `to_env()`/`from_env()` and
            # must reach `create_cli_agent`. With the `None` default this
            # assertion passed whether or not `_make_graphs` read
            # `config.allow_fs_tools`, so a dropped read would go unnoticed.
            allow_fs_tools=["ls", "read_file"],
            # Non-default budget for the same reason: it must survive the
            # `to_env()`/`from_env()` round trip and reach `create_model`.
            # With the `None` default, dropping the `cli_max_retries=` argument
            # in `_make_graphs` left every server-mode run on
            # `DEFAULT_MODEL_RETRIES` with the whole suite still green.
            cli_max_retries=3,
            summarization_model="openai:summary-model",
        )
        env_overrides = {}
        for suffix, value in config.to_env().items():
            if value is not None:
                env_overrides[f"{SERVER_ENV_PREFIX}{suffix}"] = value

        with (
            patch.dict(os.environ, env_overrides, clear=False),
            patch.dict(
                sys.modules,
                {
                    "deepagents_code.agent": agent_module,
                    "deepagents_code.config": config_module,
                    "deepagents_code.tools": tools_module,
                    "deepagents_code.mcp_tools": mcp_module,
                },
            ),
            patch(
                "deepagents_code.project_utils.get_server_project_context",
                side_effect=get_server_project_context_side_effect,
            ),
            patch(
                "deepagents_code.plugins.adapters.mcp.discover_plugin_mcp_configs",
                return_value=(),
            ),
        ):
            # Drop every ambient `DEEPAGENTS_CODE_SERVER_*` the round-trip did
            # not set. With `clear=False` an exported value (e.g.
            # `AUTO_CLASSIFIER_MODEL` from a developer shell) would survive
            # into `ServerConfig.from_env()` and break the strict
            # `create_cli_agent` kwargs assertion below, even though the test
            # never configured it.
            for key in list(os.environ):
                if key.startswith(SERVER_ENV_PREFIX) and key not in env_overrides:
                    os.environ.pop(key)

            module = _import_fresh_server_graph()
            resolve_mcp_tools.assert_not_awaited()
            assert await module.make_graph() is graph_obj

        configure_redaction.assert_called_once_with()
        reload_from_environment.assert_called_once_with(start_path=user_cwd)
        resolve_mcp_tools.assert_awaited_once()
        assert create_cli_agent_thread_ids
        assert create_cli_agent_thread_ids[0] != loop_thread_id
        # Project-context resolution and the settings bootstrap must run off the
        # loop: on Windows they call `os.getcwd()` via `Path.resolve()` /
        # `Path.cwd()`, which `blockbuster` rejects on the server event loop.
        assert setup_thread_ids
        assert all(thread_id != loop_thread_id for thread_id in setup_thread_ids)
        # Redaction fail-closed must run on the server task so
        # `configure(enabled=False)` updates this task's tracing ContextVar,
        # not only a worker-thread copy left behind by `asyncio.to_thread`.
        assert redaction_thread_ids == [loop_thread_id]
        # `create_model` must run off the loop thread: it does blocking disk IO
        # for some providers (e.g. the `openai_codex` token store calls
        # `os.mkdir`), which `blockbuster` rejects on the server event loop.
        assert create_model_thread_ids
        assert create_model_thread_ids[0] != loop_thread_id
        assert create_model.call_args.kwargs["profile_overrides"] == {
            "max_input_tokens": 32000
        }
        assert create_model.call_args.kwargs["cli_max_retries"] == 3
        kwargs = resolve_mcp_tools.await_args_list[0].kwargs
        assert kwargs["explicit_config_path"] is None
        assert kwargs["no_mcp"] is False
        assert kwargs["trust_project_mcp"] is None
        assert kwargs["project_context"] is project_context
        assert kwargs["stateless"] is True
        assert isinstance(kwargs["session_manager"], FakeSessionManager)
        create_cli_agent.assert_called_once_with(
            model=model_obj,
            assistant_id="agent",
            tools=[fetch_tool, thread_tool, web_tool, mcp_tool],
            mcp_tools=[mcp_tool],
            sandbox=None,
            sandbox_type=None,
            system_prompt=None,
            interactive=True,
            auto_approve=False,
            auto_mode_enabled=True,
            interrupt_shell_only=False,
            shell_allow_list=None,
            fs_tools=["ls", "read_file"],
            enable_ask_user=False,
            enable_memory=True,
            memory_auto_save=True,
            enable_skills=True,
            enable_shell=True,
            enable_interpreter=False,
            interpreter_config=None,
            rubric_model=None,
            rubric_max_iterations=None,
            auto_classifier_model=None,
            recursion_limit=None,
            mcp_server_info=mcp_server_info,
            cwd=user_cwd,
            project_context=project_context,
            async_subagents=None,
            goal_criteria_tools=[fetch_tool, web_tool, mcp_tool],
            rubric_grader_tools=[fetch_tool, web_tool, mcp_tool],
            model_retries=5,
            cli_max_retries=3,
            summarization_model="openai:summary-model",
            extension_registry=None,
        )

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


class TestStartupErrorMarker:
    """`emit_startup_failure` must produce the parser marker on stderr.

    The marker is the contract `wait_for_server_healthy` parses to surface
    a one-line summary instead of "Server process exited with code N".
    """
