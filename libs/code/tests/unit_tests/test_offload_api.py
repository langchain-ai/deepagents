"""Tests for the server-owned offload HTTP boundary."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import threading
import time
from collections import OrderedDict
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, create_autospec, patch

import pytest
from blockbuster import blockbuster_ctx

from deepagents_code.offload_middleware import OffloadExecution, OffloadResult

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from pathlib import Path

    from deepagents_code.offload_middleware import _PendingArchive


@pytest.fixture(autouse=True)
def _reset_offload_globals() -> Iterator[None]:
    """Clear cached clients and operation state between tests.

    `offload_api` caches one `httpx`-backed client per process (building one per
    request leaks a connection pool). Tests patch `get_client`, so the cache has
    to be dropped or the first test's mock would serve every later one.
    """
    from deepagents_code import offload_api

    offload_api._client = None
    offload_api._active_operations.clear()
    offload_api._operation_outcomes.clear()
    try:
        yield
    finally:
        offload_api._client = None
        offload_api._active_operations.clear()
        offload_api._operation_outcomes.clear()


@contextlib.asynccontextmanager
async def _workspace_route_client(
    runtime_error: BaseException,
) -> AsyncIterator[tuple[Any, MagicMock]]:
    """Serve the workspace route with a runtime build that fails.

    Binding succeeds, so every failure the caller asserts on comes from the
    runtime preflight rather than from request validation.

    Yields:
        The ASGI client and the patched thread-client factory, which must stay
        uncalled whenever the preflight refuses.
    """
    from httpx import ASGITransport, AsyncClient

    from deepagents_code import offload_api
    from deepagents_code._server_config import ServerConfig

    with (
        patch.object(ServerConfig, "from_env", return_value=ServerConfig()),
        patch.object(
            offload_api,
            "bind_thread_workspace",
            new=AsyncMock(return_value=object()),
        ),
        patch.object(
            offload_api,
            "get_server_runtime",
            new=AsyncMock(side_effect=runtime_error),
        ),
        patch.object(offload_api, "_thread_client") as thread_client,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=offload_api.app),
            base_url="http://test",
        ) as client:
            yield client, thread_client


class TestWorkspaceRoute:
    @pytest.fixture(autouse=True)
    def workspace_database(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "DEEPAGENTS_CODE_SERVER_DB_PATH", str(tmp_path / "sessions.db")
        )

    @pytest.mark.parametrize("cached", [False, True])
    @pytest.mark.parametrize("git_workspace", [False, True])
    async def test_explicit_launch_root_preserves_policy_and_runtime(
        self, tmp_path: Path, cached: bool, git_workspace: bool
    ) -> None:
        from httpx import ASGITransport, AsyncClient

        from deepagents_code import offload_api, server_graph
        from deepagents_code._server_config import ServerConfig
        from deepagents_code.workspace import require_thread_workspace

        root = tmp_path / "project"
        workdir = root / "workdir"
        workdir.mkdir(parents=True)
        if git_workspace:
            (workdir / ".git").mkdir()
        config = ServerConfig(
            cwd=str(workdir),
            project_root=str(root),
            mcp_config_path=str(root / ".mcp.json"),
            sandbox_setup=str(root / "setup.sh"),
            trust_project_mcp=True,
            trust_project_extensions=True,
            extension_paths=(str(root / "ext.py"),),
        )
        runtime = create_autospec(server_graph.ServerRuntime, instance=True)
        threads = SimpleNamespace(create=AsyncMock(), update=AsyncMock())
        with (
            patch.object(ServerConfig, "from_env", return_value=config),
            patch.object(server_graph, "_workspace_runtimes", OrderedDict()),
            patch.object(
                server_graph, "_get_runtime", new=AsyncMock(return_value=runtime)
            ),
            patch.object(
                server_graph, "_make_graphs", new=AsyncMock(return_value=runtime)
            ) as make,
            patch.object(
                offload_api, "get_server_runtime", server_graph._workspace_runtime
            ),
            patch.object(
                offload_api,
                "_thread_client",
                return_value=SimpleNamespace(threads=threads),
            ),
        ):
            if cached:
                assert await server_graph.get_server_runtime() is runtime
            async with AsyncClient(
                transport=ASGITransport(app=offload_api.app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/dcode/threads/thread-1/workspace", json={"cwd": str(workdir)}
                )
            assert response.status_code == 200, response.text
            binding = await require_thread_workspace(
                "thread-1", response.json()["workspace"]
            )
            assert binding.workspace_config() == config.to_workspace_payload()
            assert await server_graph._workspace_runtime(binding) is runtime
            if cached:
                make.assert_not_awaited()
            else:
                make.assert_awaited_once()
                assert make.await_args is not None
                built = make.await_args.kwargs["config_override"]
                context = make.await_args.kwargs["project_context_override"]
                assert built.project_root == str(root)
                assert context.project_root == root

    @pytest.mark.parametrize("initial_trust", [False, True])
    async def test_reconnect_after_extension_trust_changes(
        self, tmp_path, initial_trust: bool
    ) -> None:
        from httpx import ASGITransport, AsyncClient

        from deepagents_code import offload_api
        from deepagents_code._server_config import ServerConfig
        from deepagents_code.workspace import get_thread_workspace

        launch = tmp_path / "launch"
        other = tmp_path / "other"
        launch.mkdir()
        other.mkdir()
        config = ServerConfig(cwd=str(launch), project_root=str(launch))
        threads = SimpleNamespace(create=AsyncMock(), update=AsyncMock())
        with (
            patch.object(ServerConfig, "from_env", return_value=config),
            patch.object(offload_api, "get_server_runtime", new=AsyncMock()),
            patch.object(
                offload_api,
                "_thread_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            patch(
                "deepagents_code.extensions.trust.is_project_extensions_trusted",
                return_value=initial_trust,
            ) as trust,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=offload_api.app), base_url="http://test"
            ) as client:
                url = "/dcode/threads/thread-1/workspace"
                first = await client.post(url, json={"cwd": str(other)})
                assert first.status_code == 200
                original = await get_thread_workspace("thread-1")

                trust.return_value = not initial_trust
                resumed = await client.post(url, json={"cwd": str(other)})
                assert resumed.status_code == (409 if initial_trust else 200)
                if not initial_trust:
                    assert resumed.json() == first.json()
                assert await get_thread_workspace("thread-1") == original

                fresh = await client.post(
                    "/dcode/threads/thread-2/workspace", json={"cwd": str(other)}
                )
                assert fresh.status_code == 200
                binding = await get_thread_workspace("thread-2")
                assert binding is not None
                assert binding.workspace_config()["trust_project_extensions"] is (
                    not initial_trust
                )

    async def test_server_supplies_policy_when_client_omits_claim(
        self, tmp_path
    ) -> None:
        from deepagents_code import offload_api
        from deepagents_code._server_config import ServerConfig

        request = SimpleNamespace(
            path_params={"thread_id": "thread-1"},
            json=AsyncMock(return_value={"cwd": str(tmp_path)}),
        )
        launch = tmp_path / "launch"
        launch.mkdir()
        server_config = ServerConfig(
            auto_approve=True,
            cwd=str(launch),
            project_root=str(launch),
            mcp_config_path="/launch/.mcp.json",
            sandbox_setup="/launch/setup.sh",
            trust_project_mcp=True,
            extension_paths=("/launch/ext.py",),
        )
        binding = SimpleNamespace(
            cwd=str(tmp_path),
            workspace_id="workspace-1",
            generation=1,
            to_payload=lambda: {"workspace_id": "workspace-1"},
        )
        threads = SimpleNamespace(create=AsyncMock(), update=AsyncMock())
        runtime = AsyncMock()
        with (
            patch.object(ServerConfig, "from_env", return_value=server_config),
            patch.object(
                offload_api,
                "bind_thread_workspace",
                new=AsyncMock(return_value=binding),
            ) as bind,
            patch.object(offload_api, "get_server_runtime", new=runtime),
            patch.object(
                offload_api,
                "_thread_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            blockbuster_ctx(scanned_modules=offload_api),
        ):
            response = await offload_api.workspace(cast("Any", request))

        assert response.status_code == 200
        # Assert the literal policy, not `resolve_workspace(...)` re-run here:
        # comparing against the method under test passes even if it stops
        # stripping anything.
        bind.assert_awaited_once()
        bind_call = bind.await_args
        assert bind_call is not None
        bound_policy = bind_call.args[2]
        assert bound_policy["mcp_config_path"] is None
        assert bound_policy["sandbox_setup"] is None
        assert bound_policy["trust_project_mcp"] is None
        assert bound_policy["extension_paths"] == []
        # Session policy is the client's own and survives.
        assert bound_policy["auto_approve"] is True
        runtime.assert_awaited_once_with(binding)

    async def test_runtime_conflict_returns_409_before_thread_creation(
        self, tmp_path
    ) -> None:
        """Workspace preflight reports a conflict before a streamed run starts."""
        from deepagents_code.workspace import WorkspaceConflictError

        detail = "Cannot host this workspace because the sandbox is already owned."
        async with _workspace_route_client(WorkspaceConflictError(detail)) as (
            client,
            thread_client,
        ):
            response = await client.post(
                "/dcode/threads/thread-1/workspace",
                json={"cwd": str(tmp_path)},
            )

        assert response.status_code == 409
        assert response.json() == {"detail": detail}
        thread_client.assert_not_called()

    async def test_runtime_build_exit_is_contained_as_503(self, tmp_path) -> None:
        """`_make_graphs` exits on sandbox failure; the route must contain it.

        Without containment the `SystemExit` escapes the handler and takes the
        server down mid-request, since it is a `BaseException`.
        """
        async with _workspace_route_client(SystemExit(1)) as (client, thread_client):
            response = await client.post(
                "/dcode/threads/thread-1/workspace",
                json={"cwd": str(tmp_path)},
            )

        assert response.status_code == 503
        assert "could not build its agent runtime" in response.json()["detail"]
        thread_client.assert_not_called()

    async def test_runtime_build_error_is_not_reported_as_client_error(
        self, tmp_path
    ) -> None:
        """A build `ValueError` is server misconfiguration, never a 422."""
        error = ValueError("unknown model provider")
        async with _workspace_route_client(error) as (client, thread_client):
            # `ASGITransport` re-raises app exceptions, so the build failure
            # surfaces here rather than as the 422 the validation arm above
            # would have produced when it shared this `try`.
            with pytest.raises(ValueError, match="unknown model provider"):
                await client.post(
                    "/dcode/threads/thread-1/workspace",
                    json={"cwd": str(tmp_path)},
                )

        thread_client.assert_not_called()

    async def test_rejects_client_policy_mismatch(self, tmp_path) -> None:
        """A caller cannot choose privileged runtime configuration."""
        from deepagents_code import offload_api
        from deepagents_code._server_config import ServerConfig

        request = SimpleNamespace(
            path_params={"thread_id": "thread-1"},
            json=AsyncMock(
                return_value={
                    "cwd": str(tmp_path),
                    "workspace_config": {"auto_approve": True},
                    "config_fingerprint": "attacker",
                }
            ),
        )
        server_config = ServerConfig(auto_approve=False)
        with patch.object(
            ServerConfig, "from_env", return_value=server_config
        ) as from_env:
            response = await offload_api.workspace(cast("Any", request))

        assert response.status_code == 409
        from_env.assert_called_once_with()

    async def test_rejects_client_project_policy_claim(self, tmp_path) -> None:
        from deepagents_code import offload_api
        from deepagents_code._server_config import ServerConfig

        config = ServerConfig()
        claim = config.to_session_workspace_claim()
        claim["trust_project_mcp"] = True
        request = SimpleNamespace(
            path_params={"thread_id": "thread-1"},
            json=AsyncMock(
                return_value={
                    "cwd": str(tmp_path),
                    "workspace_config": claim,
                    "config_fingerprint": config.session_workspace_fingerprint(),
                }
            ),
        )

        with patch.object(ServerConfig, "from_env", return_value=config):
            response = await offload_api.workspace(cast("Any", request))

        assert response.status_code == 409
        assert response.body == (
            b'{"detail":"clients cannot claim project workspace policy"}'
        )

    async def test_rejects_unknown_workspace_request_field(self, tmp_path) -> None:
        from deepagents_code import offload_api

        request = SimpleNamespace(
            path_params={"thread_id": "thread-1"},
            json=AsyncMock(
                return_value={"cwd": str(tmp_path), "trust_project_mcp": True}
            ),
        )

        response = await offload_api.workspace(cast("Any", request))

        assert response.status_code == 422
        assert b"trust_project_mcp" in response.body

    async def test_malformed_policy_is_a_client_error(self, tmp_path) -> None:
        """A non-object policy returns 422 instead of escaping as a 500."""
        from deepagents_code import offload_api
        from deepagents_code._server_config import ServerConfig

        request = SimpleNamespace(
            path_params={"thread_id": "thread-1"},
            json=AsyncMock(
                return_value={
                    "cwd": str(tmp_path),
                    "workspace_config": ["not", "an", "object"],
                    "config_fingerprint": "attacker",
                }
            ),
        )
        with patch.object(ServerConfig, "from_env", return_value=ServerConfig()):
            response = await offload_api.workspace(cast("Any", request))

        assert response.status_code == 422


class TestOperationPayload:
    """Malformed client requests fail with a field-naming 422 at the boundary."""

    @pytest.mark.parametrize(
        "key",
        [
            "base_url",
            "api_base",
            "openai_api_base",
            "anthropic_api_url",
            "azure_endpoint",
            "azure_openai_api_base",
            "api_endpoint",
            "openai_proxy",
            "anthropic_proxy",
            "proxy",
            "proxies",
            "http_client",
            "http_async_client",
            "transport",
            "default_headers",
            "custom_headers",
        ],
    )
    def test_transport_model_params_are_stripped(self, key: str) -> None:
        """The boundary drops endpoint/transport keys from `model_params`.

        These keys would route the server's credentialed provider calls to a
        client-chosen destination.
        """
        from deepagents_code.offload_api import _operation_payload

        _, context, _ = _operation_payload(
            {
                "operation_id": "op-1",
                "context": {
                    "model": "openai:gpt-5",
                    "model_params": {
                        key: "http://attacker.example/",
                        "temperature": 0.2,
                    },
                },
            }
        )

        assert context["model_params"] == {"temperature": 0.2}

    def test_stripping_is_logged_with_key_names_only(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A dropped transport key must leave a trace naming the key.

        Silently ignoring it leaves a user whose gateway config is being skipped
        with nothing to find. The value is an endpoint or header, so only the
        key name is logged.
        """
        import logging

        from deepagents_code.offload_api import _operation_payload

        with caplog.at_level(logging.WARNING):
            _operation_payload(
                {
                    "operation_id": "op-1",
                    "context": {
                        "model_params": {
                            "base_url": "http://gateway.internal/v1",
                            "temperature": 0.2,
                        }
                    },
                }
            )

        assert "base_url" in caplog.text
        assert "gateway.internal" not in caplog.text


def _thread_state(checkpoint_id: str = "checkpoint-1") -> dict[str, object]:
    """Build an idle LangGraph thread-state response."""
    return {
        "values": {
            "messages": [
                {
                    "role": "user",
                    "content": "hello",
                    "id": "message-1",
                }
            ],
            "_session_cost_usd": 1.0,
            "_model_spec": "provider:checkpointed-model",
            "_model_params": {
                "base_url": "https://trusted.example/v1",
                "temperature": 0.1,
            },
        },
        "next": [],
        "tasks": [],
        "interrupts": [],
        "checkpoint": {
            "thread_id": "thread-1",
            "checkpoint_ns": "",
            "checkpoint_id": checkpoint_id,
            "checkpoint_map": {},
        },
    }


def _result(
    archive_path: str | None = "/conversation_history/thread-1.md",
) -> OffloadResult:
    """Build a complete operation result."""
    return {
        "status": "compacted",
        "messages_offloaded": 1,
        "messages_kept": 1,
        "tokens_before": 20,
        "tokens_after": 10,
        "archive_path": archive_path,
        "archive_ephemeral": False,
        "error": None,
    }


class TestExecuteOffload:
    async def test_missing_workspace_binding_is_rejected(self) -> None:
        """Offload never falls back to the process-global runtime."""
        from deepagents_code import offload_api
        from deepagents_code.workspace import WorkspaceConflictError

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(return_value=_thread_state()),
            update_state=AsyncMock(),
        )
        runtime = AsyncMock()
        with (
            patch.object(
                offload_api,
                "get_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            patch.object(
                offload_api,
                "require_thread_workspace",
                new=AsyncMock(side_effect=WorkspaceConflictError("not bound")),
            ),
            patch.object(offload_api, "get_server_runtime", new=runtime),
            pytest.raises(offload_api._OffloadConflictError, match="not bound"),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        runtime.assert_not_awaited()
        threads.update_state.assert_not_awaited()

    async def test_runtime_workspace_conflict_is_rejected(self) -> None:
        """Runtime conflicts use the same offload 409 path as binding conflicts."""
        from deepagents_code import offload_api
        from deepagents_code.workspace import WorkspaceConflictError

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(return_value=_thread_state()),
            update_state=AsyncMock(),
        )
        detail = (
            "Cannot host this workspace because a runtime for another workspace "
            "already exists and the configured sandbox is process-wide."
        )
        with (
            patch.object(
                offload_api,
                "get_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            patch.object(
                offload_api,
                "require_thread_workspace",
                new=AsyncMock(return_value=object()),
            ),
            patch.object(
                offload_api,
                "get_server_runtime",
                new=AsyncMock(side_effect=WorkspaceConflictError(detail)),
            ),
        ):
            response = await offload_api.offload(
                SimpleNamespace(  # ty: ignore[invalid-argument-type]
                    path_params={"thread_id": "thread-1"},
                    json=AsyncMock(
                        return_value={
                            "operation_id": "operation-1",
                            "context": {"workspace": {}},
                        }
                    ),
                )
            )

        assert response.status_code == 409
        import json

        assert json.loads(bytes(response.body)) == {"detail": detail}
        threads.update_state.assert_not_awaited()

    async def test_missing_workspace_context_is_rejected(self) -> None:
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(return_value=_thread_state()),
            update_state=AsyncMock(),
        )
        runtime = AsyncMock()
        with (
            patch.object(
                offload_api,
                "get_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            patch.object(
                offload_api,
                "require_thread_workspace",
                new=AsyncMock(side_effect=TypeError("workspace context is required")),
            ),
            patch.object(offload_api, "get_server_runtime", new=runtime),
            pytest.raises(
                offload_api._OffloadConflictError,
                match="workspace context is required",
            ),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        runtime.assert_not_awaited()
        threads.update_state.assert_not_awaited()

    """The route owns state hydration, validation, and atomic persistence."""

    def test_hydrates_persisted_summary_message(self) -> None:
        """A subsequent offload receives a message object in its prior event."""
        from langchain_core.messages import HumanMessage

        from deepagents_code.offload_api import _hydrate_state

        state = _hydrate_state(
            {
                "messages": [
                    {"role": "user", "content": "new message", "id": "message-1"}
                ],
                "_summarization_event": {
                    "cutoff_index": 1,
                    "summary_message": {
                        "type": "human",
                        "content": "Prior summary.",
                        "id": "summary-1",
                    },
                },
            }
        )

        event = state["_summarization_event"]
        assert isinstance(event, dict)
        assert isinstance(event["summary_message"], HumanMessage)
        assert event["summary_message"].content == "Prior summary."

    async def test_commits_event_and_cost_without_messages(self) -> None:
        from deepagents_code import offload_api

        before = _thread_state()
        calls: list[str] = []

        async def update_state(  # noqa: RUF029  # AsyncMock side effect contract
            *_args: object, **_kwargs: object
        ) -> None:
            calls.append("checkpoint")

        append = SimpleNamespace(
            path="/conversation_history/thread-1.md", rollback=AsyncMock()
        )
        archive = SimpleNamespace(
            session_id="archive-1",
            write=AsyncMock(side_effect=lambda: calls.append("archive") or append),
            update=lambda path: {
                "_summarization_event": {"cutoff_index": 1, "file_path": path}
            },
        )
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(side_effect=[before, before]),
            update_state=AsyncMock(side_effect=update_state),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution(
                    {
                        "_summarization_event": {
                            "cutoff_index": 1,
                            "file_path": None,
                        },
                        "_summarization_session_id": "archive-1",
                    },
                    _result(archive_path=None),
                    cast("_PendingArchive", archive),
                )
            )
        )
        prepared = SimpleNamespace(
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
        )

        with (
            patch.object(
                offload_api,
                "get_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            patch.object(
                offload_api,
                "require_thread_workspace",
                new=AsyncMock(return_value=object()),
            ),
            patch.object(
                offload_api,
                "get_server_runtime",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        agent=SimpleNamespace(store=None), offload=operation
                    )
                ),
            ),
            patch.object(offload_api, "prepare_operation_cost", return_value=prepared),
        ):
            response = await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={"model": "test:model"},
                hook_responses={},
            )

        assert response == {"status": "complete", "result": _result()}
        state = operation.execute.await_args.args[0]
        assert state["messages"][0].id == "message-1"
        runtime = operation.execute.await_args.args[1]
        assert runtime.context["thread_id"] == "thread-1"
        assert runtime.context["model"] == "provider:checkpointed-model"
        assert runtime.context["model_params"] == {
            "base_url": "https://trusted.example/v1",
            "temperature": 0.1,
        }
        assert threads.update_state.await_count == 2
        args = threads.update_state.await_args_list[0]
        assert args.args[:2] == (
            "thread-1",
            {
                "_summarization_event": {
                    "cutoff_index": 1,
                    "file_path": None,
                },
                "_summarization_session_id": "archive-1",
                "_session_cost_usd": 0.25,
            },
        )
        assert "messages" not in args.args[1]
        assert "checkpoint" not in args.kwargs
        assert threads.update_state.await_args_list[1].args == (
            "thread-1",
            {
                "_summarization_event": {
                    "cutoff_index": 1,
                    "file_path": "/conversation_history/thread-1.md",
                }
            },
        )
        assert calls == ["checkpoint", "archive", "checkpoint"]
        prepared.rollback.assert_not_called()

    async def test_failed_archive_link_restores_the_append(self) -> None:
        """A failed follow-up checkpoint cannot leave duplicate history."""
        from deepagents_code import offload_api

        before = _thread_state()
        unlinked = _thread_state("reserved")
        unlinked_values = cast("dict[str, object]", unlinked["values"])
        unlinked_values["_summarization_event"] = {
            "cutoff_index": 1,
            "file_path": None,
        }
        append = SimpleNamespace(
            path="/conversation_history/thread-1.md", rollback=AsyncMock()
        )
        archive = SimpleNamespace(
            session_id="archive-1",
            write=AsyncMock(return_value=append),
            update=lambda path: {
                "_summarization_event": {"cutoff_index": 1, "file_path": path}
            },
        )
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(side_effect=[before, before, unlinked]),
            update_state=AsyncMock(
                side_effect=[None, RuntimeError("archive link unavailable")]
            ),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution(
                    {
                        "_summarization_event": {
                            "cutoff_index": 1,
                            "file_path": None,
                        },
                        "_summarization_session_id": "archive-1",
                    },
                    _result(archive_path=None),
                    cast("_PendingArchive", archive),
                )
            )
        )
        prepared = SimpleNamespace(
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
            records=[],
        )

        with self._patched(offload_api, threads, operation, prepared):
            response = await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        assert response["status"] == "complete"
        assert response["result"]["archive_path"] is None
        append.rollback.assert_awaited_once()
        prepared.rollback.assert_not_called()

    async def test_cancellation_waits_for_checkpoint_archive_settlement(self) -> None:
        """A reserved commit must settle before cancellation becomes terminal."""
        from deepagents_code import offload_api

        before = _thread_state()
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(side_effect=[before, before]),
            update_state=AsyncMock(),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution(
                    {"_summarization_event": {"cutoff_index": 1}},
                    _result(),  # ty: ignore[invalid-argument-type]
                )
            )
        )
        prepared = SimpleNamespace(
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
        )
        settlement_started = asyncio.Event()
        finish_settlement = asyncio.Event()

        async def settle(*_args: object, **_kwargs: object) -> None:
            settlement_started.set()
            await finish_settlement.wait()

        with (
            self._patched(offload_api, threads, operation, prepared),
            patch.object(
                offload_api,
                "_commit_deferred_archive",
                new=AsyncMock(side_effect=settle),
            ) as commit,
        ):
            task = asyncio.create_task(
                offload_api._execute_offload(
                    "thread-1",
                    operation_id="operation-1",
                    context={},
                    hook_responses={},
                )
            )
            await asyncio.wait_for(settlement_started.wait(), timeout=1)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            finish_settlement.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        commit.assert_awaited_once()

    async def test_request_transport_cannot_replace_checkpointed_model(self) -> None:
        """Offload uses the model settings from the target thread checkpoint."""
        from deepagents_code import offload_api

        before = _thread_state()
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(side_effect=[before, before]),
            update_state=AsyncMock(),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution(
                    {},
                    _result(),  # ty: ignore[invalid-argument-type]
                )
            )
        )
        prepared = SimpleNamespace(update={}, rollback=MagicMock(), commit=MagicMock())

        with self._patched(offload_api, threads, operation, prepared):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={
                    "model": "attacker:model",
                    "model_params": {"base_url": "https://attacker.example"},
                },
                hook_responses={},
            )

        runtime = operation.execute.await_args.args[1]
        assert runtime.context["model"] == "provider:checkpointed-model"
        assert runtime.context["model_params"]["base_url"] == (
            "https://trusted.example/v1"
        )

    async def test_legacy_thread_reuses_startup_summarizer(self) -> None:
        """A thread without model metadata ignores request model selection."""
        from deepagents_code import offload_api

        before = _thread_state()
        values = cast("dict[str, object]", before["values"])
        assert isinstance(values, dict)
        values.pop("_model_spec")
        values.pop("_model_params")
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(side_effect=[before, before]),
            update_state=AsyncMock(),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution(
                    {},
                    _result(),  # ty: ignore[invalid-argument-type]
                )
            )
        )
        prepared = SimpleNamespace(update={}, rollback=MagicMock(), commit=MagicMock())

        with self._patched(offload_api, threads, operation, prepared):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={"model": "request:model", "model_params": {"x": 1}},
                hook_responses={},
            )

        runtime = operation.execute.await_args.args[1]
        assert "model" not in runtime.context
        assert "model_params" not in runtime.context

    async def test_checkpoint_change_fails_without_state_commit(self) -> None:
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(
                side_effect=[_thread_state("before"), _thread_state("changed")]
            ),
            update_state=AsyncMock(),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution(
                    {"_summarization_event": {"cutoff_index": 1}},
                    _result(),  # ty: ignore[invalid-argument-type]
                    cast(
                        "_PendingArchive",
                        SimpleNamespace(session_id="archive-1", write=AsyncMock()),
                    ),
                )
            )
        )

        with (
            patch.object(
                offload_api,
                "get_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            patch.object(
                offload_api,
                "require_thread_workspace",
                new=AsyncMock(return_value=object()),
            ),
            patch.object(
                offload_api,
                "get_server_runtime",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        agent=SimpleNamespace(store=None), offload=operation
                    )
                ),
            ),
            patch.object(offload_api, "prepare_operation_cost") as prepare,
            pytest.raises(offload_api._OffloadConflictError, match="thread changed"),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        threads.update_state.assert_not_awaited()
        operation.execute.return_value.archive.write.assert_not_awaited()
        prepare.assert_not_called()

    @pytest.mark.parametrize("status", ["busy", "interrupted"])
    async def test_thread_with_work_in_flight_is_rejected_before_operation(
        self, status: str
    ) -> None:
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": status}),
            get_state=AsyncMock(),
            update_state=AsyncMock(),
        )
        runtime = AsyncMock()
        with (
            patch.object(
                offload_api,
                "get_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            patch.object(offload_api, "get_server_runtime", new=runtime),
            pytest.raises(offload_api._OffloadConflictError, match="active"),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        threads.get_state.assert_not_awaited()
        runtime.assert_not_awaited()

    async def test_errored_thread_is_still_offloadable(self) -> None:
        """A failed turn must not lock the user out of `/offload`.

        A run that raises leaves the thread row on `error` until the next run
        completes, which is exactly when a user reaches for `/offload` to
        recover. Reaching the state read proves the status gate let it past;
        in-flight work is caught separately by the `next`/`tasks`/`interrupts`
        check against the checkpoint.
        """
        from deepagents_code import offload_api

        class _ReachedStateReadError(Exception):
            """Sentinel proving control passed the thread-status gate."""

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "error"}),
            get_state=AsyncMock(side_effect=_ReachedStateReadError),
            update_state=AsyncMock(),
        )
        with (
            patch.object(
                offload_api,
                "get_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            patch.object(offload_api, "get_server_runtime", new=AsyncMock()),
            pytest.raises(_ReachedStateReadError),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        threads.update_state.assert_not_awaited()

    @staticmethod
    @contextlib.contextmanager
    def _patched(
        offload_api: object,
        threads: SimpleNamespace,
        operation: SimpleNamespace,
        prepared: object,
    ) -> Iterator[None]:
        """Patch the client, workspace, runtime, and cost seams used here."""
        with (
            patch.object(
                offload_api,
                "get_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            patch.object(
                offload_api,
                "require_thread_workspace",
                new=AsyncMock(return_value=object()),
            ),
            patch.object(
                offload_api,
                "get_server_runtime",
                new=AsyncMock(
                    return_value=SimpleNamespace(
                        agent=SimpleNamespace(store=None), offload=operation
                    )
                ),
            ),
            patch.object(offload_api, "prepare_operation_cost", return_value=prepared),
        ):
            yield

    @pytest.mark.parametrize("channel", ["messages", "todos"])
    async def test_unpermitted_update_is_refused_and_cost_rolled_back(
        self, channel: str
    ) -> None:
        """A channel outside `OffloadStateUpdate` cannot reach the checkpoint.

        Unlike asserting on a mocked update that never contains the channel
        (which passes whether the guard exists or not), this drives an update
        that actually carries one. `todos` covers the allowlist itself: a
        `messages`-only check would let every other channel through.
        """
        from deepagents_code import offload_api

        before = _thread_state()
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(side_effect=[before, before]),
            update_state=AsyncMock(),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution(
                    # Deliberately violates `OffloadStateUpdate` -- that is the
                    # point of the test: the runtime guard is the backstop for
                    # `Any`-typed values the SDK hands back.
                    {channel: ["smuggled"]},  # ty: ignore[invalid-key,invalid-argument-type]
                    _result(),  # ty: ignore[invalid-argument-type]
                )
            )
        )
        prepared = SimpleNamespace(update={}, rollback=MagicMock(), commit=MagicMock())

        with (
            self._patched(offload_api, threads, operation, prepared),
            pytest.raises(RuntimeError, match=f"may not write .*{channel}"),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        threads.update_state.assert_not_awaited()
        prepared.rollback.assert_called_once()

    async def test_empty_update_still_releases_claimed_cost_records(self) -> None:
        """A prepare with nothing to write must not silently eat its records."""
        from deepagents_code import offload_api

        before = _thread_state()
        noop_result = {**_result(), "status": "noop"}
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(side_effect=[before, before]),
            update_state=AsyncMock(),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution({}, noop_result)  # ty: ignore[invalid-argument-type]
            )
        )
        prepared = SimpleNamespace(update={}, rollback=MagicMock(), commit=MagicMock())

        with self._patched(offload_api, threads, operation, prepared):
            response = await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        assert response == {"status": "complete", "result": noop_result}
        threads.update_state.assert_not_awaited()
        prepared.rollback.assert_called_once()

    async def test_write_failure_without_advance_restores_cost(self) -> None:
        """A write that provably did not land returns its records to the recorder."""
        from deepagents_code import offload_api

        before = _thread_state()
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            # Third read is `_write_landed`: same checkpoint => did not land.
            get_state=AsyncMock(side_effect=[before, before, before]),
            update_state=AsyncMock(side_effect=RuntimeError("boom")),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution(
                    {"_summarization_event": {"cutoff_index": 1}},
                    _result(),  # ty: ignore[invalid-argument-type]
                )
            )
        )
        prepared = SimpleNamespace(
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
            records=[],
        )

        with (
            self._patched(offload_api, threads, operation, prepared),
            pytest.raises(RuntimeError, match="boom"),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        prepared.rollback.assert_called_once()

    async def test_write_failure_after_advance_keeps_cost_claimed(self) -> None:
        """An indeterminate write must not re-queue records and double-charge."""
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(
                side_effect=[
                    _thread_state("before"),
                    _thread_state("before"),
                    # `_write_landed`: the thread advanced, so the write likely
                    # applied despite the transport error.
                    _thread_state("after"),
                ]
            ),
            update_state=AsyncMock(side_effect=RuntimeError("connection reset")),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution(
                    {"_summarization_event": {"cutoff_index": 1}},
                    _result(),  # ty: ignore[invalid-argument-type]
                )
            )
        )
        prepared = SimpleNamespace(
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
            records=[],
        )

        with (
            self._patched(offload_api, threads, operation, prepared),
            pytest.raises(
                offload_api._OffloadIndeterminateError, match="could not confirm"
            ),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        prepared.rollback.assert_not_called()

    async def test_unreadable_thread_does_not_claim_the_thread_advanced(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """An unreadable readback must not be logged as an observed advance.

        Both outcomes keep the records claimed, but only `advanced` has evidence
        the write landed. Reporting a thread advance that was never observed
        would tell anyone auditing a missing charge that the spend was
        accounted for.
        """
        import logging

        from deepagents_code import offload_api

        before = _thread_state()
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(
                side_effect=[before, before, RuntimeError("thread store offline")]
            ),
            update_state=AsyncMock(side_effect=RuntimeError("connection reset")),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution(
                    {"_summarization_event": {"cutoff_index": 1}},
                    _result(),  # ty: ignore[invalid-argument-type]
                )
            )
        )
        prepared = SimpleNamespace(
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
            records=[],
        )

        with (
            self._patched(offload_api, threads, operation, prepared),
            caplog.at_level(logging.ERROR),
            pytest.raises(offload_api._OffloadIndeterminateError),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        assert "could not be read back" in caplog.text
        assert "may be lost from the thread total" in caplog.text
        assert "advanced past checkpoint" not in caplog.text
        prepared.rollback.assert_not_called()
        prepared.commit.assert_called_once()

    async def test_cancelled_probe_still_settles_the_cost_records(self) -> None:
        """A cancel inside the write-landed probe must not skip settlement.

        `prepare_operation_cost` drains the recorder destructively, so a prepare
        that is neither committed nor rolled back deletes that spend from the
        thread's lifetime total permanently. The probe runs inside the
        settlement handler, so an escape from it -- a disconnect or a shutdown
        re-delivering cancellation while the handler unwinds -- would take both
        branches off the table and lose the records with no trace.
        """
        from deepagents_code import offload_api

        before = _thread_state()
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(side_effect=[before, before, asyncio.CancelledError()]),
            update_state=AsyncMock(side_effect=RuntimeError("boom")),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution(
                    {"_summarization_event": {"cutoff_index": 1}},
                    _result(),  # ty: ignore[invalid-argument-type]
                )
            )
        )
        prepared = SimpleNamespace(
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
            records=[],
        )

        with (
            self._patched(offload_api, threads, operation, prepared),
            pytest.raises(offload_api._OffloadIndeterminateError),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        # Unreadable means indeterminate, so the records stay claimed rather
        # than being restored -- but a decision was reached either way.
        prepared.rollback.assert_not_called()

    async def test_a_cancelled_write_is_not_converted_to_a_runtime_error(
        self,
    ) -> None:
        """Cancellation must propagate so the task actually observes it."""
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(
                side_effect=[
                    _thread_state("before"),
                    _thread_state("before"),
                    _thread_state("after"),
                ]
            ),
            update_state=AsyncMock(side_effect=asyncio.CancelledError()),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(
                return_value=OffloadExecution(
                    {"_summarization_event": {"cutoff_index": 1}},
                    _result(),  # ty: ignore[invalid-argument-type]
                )
            )
        )
        prepared = SimpleNamespace(
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
            records=[],
        )

        with (
            self._patched(offload_api, threads, operation, prepared),
            pytest.raises(asyncio.CancelledError),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        prepared.rollback.assert_not_called()

    @staticmethod
    def _hook_request() -> object:
        """Build a real server-owned hook invocation request."""
        from datetime import UTC, datetime
        from pathlib import Path
        from uuid import uuid4

        from deepagents_code.hooks.models.domain import (
            ApprovalMode,
            HookContext,
            HookEvent,
            PreToolUseEvent,
            ToolCallData,
        )
        from deepagents_code.hooks.models.transport import (
            HookInvocation,
            HookInvocationRequest,
        )

        return HookInvocationRequest(
            protocol_version=1,
            invocation_id=uuid4(),
            snapshot_id="snapshot-1",
            run_id="run-1",
            invocation=HookInvocation(
                context=HookContext(
                    thread_id="thread-1",
                    cwd=Path("/tmp"),
                    approval_mode=ApprovalMode.MANUAL,
                ),
                event=PreToolUseEvent(
                    event=HookEvent.PRE_TOOL_USE,
                    call=ToolCallData(
                        id="call-1", name="compact_conversation", args={"force": True}
                    ),
                ),
            ),
            deadline=datetime(2026, 7, 23, tzinfo=UTC),
        )

    async def test_a_hook_request_becomes_an_interrupt_response(self) -> None:
        """An unanswered hook must leave the route as a resumable interrupt.

        `HookTransportInterruptError` derives from `BaseException` so the
        compaction chain cannot swallow it -- which also means the route's own
        `except Exception` cannot catch it. Without the dedicated handler it
        escapes to Starlette as a raw 500 for every user with a `PreCompact` or
        `PreToolUse` hook configured, and nothing else in the suite notices.
        """
        from deepagents_code import offload_api
        from deepagents_code.hooks.interrupt import is_hook_interrupt_payload
        from deepagents_code.hooks.server_middleware import (
            HookTransportInterruptError,
        )

        request = self._hook_request()
        before = _thread_state()
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(return_value=before),
            update_state=AsyncMock(),
        )
        operation = SimpleNamespace(
            execute=AsyncMock(side_effect=HookTransportInterruptError(request))  # ty: ignore[invalid-argument-type]
        )
        prepared = SimpleNamespace(
            update={}, rollback=MagicMock(), commit=MagicMock(), records=[]
        )

        with self._patched(offload_api, threads, operation, prepared):
            response = await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        assert response["status"] == "interrupt"
        assert is_hook_interrupt_payload(response["request"])
        assert response["request"]["request"]["invocation_id"] == str(
            request.invocation_id  # ty: ignore[unresolved-attribute]
        )
        # Nothing may be committed while a hook is still unanswered.
        threads.update_state.assert_not_awaited()

    async def test_accumulated_hook_responses_reach_the_operation(self) -> None:
        """The resume round must hand the replies back to the hook transport.

        `operation_hook_responses` is the single line that makes a multi-round
        resume terminate: `_invoke_hook` replays an already-answered invocation
        from that mapping instead of raising again. Passing an empty mapping
        would re-raise the same invocation forever and the client would die at
        its round limit, so assert the mapping is actually installed.
        """
        from deepagents_code import offload_api
        from deepagents_code.hooks.server_middleware import operation_hook_responses

        seen: list[object] = []

        async def execute(  # noqa: RUF029 -- must satisfy the async execute signature
            *_args: object, **_kwargs: object
        ) -> OffloadExecution:
            # Read the context var the way the hook transport does.
            from deepagents_code.hooks import server_middleware

            seen.append(server_middleware._HOOK_RESPONSES.get())
            return OffloadExecution(
                {"_summarization_event": {"cutoff_index": 1}},
                _result(),  # ty: ignore[invalid-argument-type]
            )

        before = _thread_state()
        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(side_effect=[before, before]),
            update_state=AsyncMock(),
        )
        operation = SimpleNamespace(execute=AsyncMock(side_effect=execute))
        prepared = SimpleNamespace(
            update={"_session_cost_usd": 0.25},
            rollback=MagicMock(),
            commit=MagicMock(),
            delta_usd=0.25,
            records=[],
        )

        replies: dict[str, object] = {"hook-1": {"decision": "allow"}}
        with self._patched(offload_api, threads, operation, prepared):
            response = await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses=replies,
            )

        assert response["status"] == "complete"
        assert seen == [replies]
        # Outside the operation the var is back to graph mode.
        assert operation_hook_responses is not None

    async def test_unregistered_thread_is_rejected_with_an_actionable_conflict(
        self,
    ) -> None:
        """A 404 from the live thread store must not become an opaque 500.

        The dev server keeps checkpoint persistence and thread registration
        separate, so a resumed thread can 404 here while holding state on disk.
        `NotFoundError` is an ordinary `Exception`, so without this mapping it
        reaches the route's generic handler and the user is told to read the
        server log.
        """
        import httpx
        from langgraph_sdk.errors import NotFoundError

        from deepagents_code import offload_api

        request = httpx.Request("GET", "http://localhost/threads/thread-1")
        not_found = NotFoundError(
            "missing", response=httpx.Response(404, request=request), body=None
        )
        threads = SimpleNamespace(
            get=AsyncMock(side_effect=not_found),
            get_state=AsyncMock(),
            update_state=AsyncMock(),
        )
        runtime = AsyncMock()
        with (
            patch.object(
                offload_api,
                "get_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            patch.object(offload_api, "get_server_runtime", new=runtime),
            pytest.raises(
                offload_api._OffloadConflictError, match="not registered on the server"
            ),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        threads.get_state.assert_not_awaited()
        threads.update_state.assert_not_awaited()
        runtime.assert_not_awaited()

    async def test_empty_thread_reports_nothing_to_offload(self) -> None:
        """An empty thread is an unchanged outcome, not a failure.

        `_checkpoint_id` rejects a thread with no checkpoint, so answering the
        empty case at the boundary is what keeps `OffloadOperation.execute`'s
        graceful `empty` branch reachable over HTTP. Without it the user is told
        the operation failed for a thread that simply has nothing to compact.
        """
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(return_value={"values": {}, "checkpoint": {}}),
            update_state=AsyncMock(),
        )
        runtime = AsyncMock()
        with (
            patch.object(
                offload_api,
                "get_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            patch.object(offload_api, "get_server_runtime", new=runtime),
        ):
            response = await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )

        assert response["status"] == "complete"
        assert response["result"]["status"] == "empty"
        assert response["result"]["messages_kept"] == 0
        # Nothing is compacted, so nothing is written and no agent is built.
        threads.update_state.assert_not_awaited()
        runtime.assert_not_awaited()

    async def test_missing_checkpoint_with_messages_is_rejected(self) -> None:
        """State with messages but no checkpoint cannot be written against."""
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "idle"}),
            get_state=AsyncMock(
                return_value={**_thread_state(), "checkpoint": {}},
            ),
            update_state=AsyncMock(),
        )
        with (
            patch.object(
                offload_api,
                "get_client",
                return_value=SimpleNamespace(threads=threads),
            ),
            patch.object(offload_api, "get_server_runtime", new=AsyncMock()),
            pytest.raises(offload_api._OffloadConflictError, match="no checkpoint"),
        ):
            await offload_api._execute_offload(
                "thread-1",
                operation_id="operation-1",
                context={},
                hook_responses={},
            )


_ORPHAN_JOIN_TIMEOUT = 5.0
"""Seconds a test waits for a released flush thread before giving up."""


def _flush_client() -> MagicMock:
    """Build an autospecced LangSmith client for shutdown tests."""
    from langsmith import Client

    return create_autospec(Client, instance=True)


def _new_flush_thread(existing: set[threading.Thread]) -> threading.Thread:
    """Return the flush thread started after the supplied snapshot."""
    started = [
        thread
        for thread in threading.enumerate()
        if thread not in existing and thread.name == "langsmith-shutdown-flush"
    ]
    assert len(started) == 1, started
    return started[0]


def _join_flush_threads(existing: set[threading.Thread]) -> None:
    """Wait for flush threads started after the supplied snapshot."""
    for thread in threading.enumerate():
        if thread not in existing and thread.name == "langsmith-shutdown-flush":
            thread.join(_ORPHAN_JOIN_TIMEOUT)


class TestCheckpointModelContext:
    """The trust boundary between a request's context and the models it picks.

    `/offload` is server-owned and the dev server accepts connections from any
    local process, so every model spec the operation resolves must be
    server-sourced. A field that is merely *narrowed* downstream still reached
    `create_model` first.
    """

    def test_request_model_selection_is_replaced_by_checkpointed_values(self) -> None:
        from deepagents_code.offload_api import _checkpoint_model_context

        trusted = _checkpoint_model_context(
            {"model": "evil:model", "model_params": {"base_url": "http://attacker"}},
            {"_model_spec": "openai:gpt-5.4", "_model_params": {"temperature": 0}},
        )

        assert trusted["model"] == "openai:gpt-5.4"
        assert trusted["model_params"] == {"temperature": 0}

    def test_client_supplied_summarization_model_is_discarded(self) -> None:
        """A bare spec carries no endpoint, but it still names a provider.

        The server holds credentials for every configured provider, so honoring
        this key would let any local client route the thread's conversation
        history to a provider its owner never chose. There is no checkpointed
        summary spec to substitute, so the operation falls back to the server's
        own launch configuration.
        """
        from deepagents_code.offload_api import _checkpoint_model_context

        trusted = _checkpoint_model_context(
            {"summarization_model": "attacker-provider:model"},
            {"_model_spec": "openai:gpt-5.4"},
        )

        assert "summarization_model" not in trusted

    def test_summarization_model_is_dropped_without_a_model_checkpoint(self) -> None:
        """Threads predating model checkpointing must not become a bypass."""
        from deepagents_code.offload_api import _checkpoint_model_context

        trusted = _checkpoint_model_context(
            {"summarization_model": "attacker-provider:model"}, {}
        )

        assert "summarization_model" not in trusted
        assert "model" not in trusted

    def test_unrelated_context_survives(self) -> None:
        from deepagents_code.offload_api import _checkpoint_model_context

        trusted = _checkpoint_model_context(
            {"thread_id": "t1", "profile_overrides": {"max_input_tokens": 1}},
            {"_model_spec": "openai:gpt-5.4"},
        )

        assert trusted["thread_id"] == "t1"
        assert trusted["profile_overrides"] == {"max_input_tokens": 1}


class TestValidateContext:
    """Type checks on the context fields the offload operation consumes."""

    def test_summarization_model_must_be_a_string_or_null(self) -> None:
        """Reject a wrong-typed value instead of silently ignoring it.

        `_runtime_model_config` narrows this field with `isinstance`, so
        without a type check the client gets a field that quietly does nothing
        rather than an error naming it.
        """
        from deepagents_code.offload_api import _validate_context

        with pytest.raises(TypeError, match=re.escape("context.summarization_model")):
            _validate_context({"summarization_model": 7})

    @pytest.mark.parametrize("value", [None, "provider:model"])
    def test_accepts_a_string_or_null(self, value: str | None) -> None:
        from deepagents_code.offload_api import _validate_context

        _validate_context({"summarization_model": value})


class TestLifespan:
    """Shutdown flushes buffered LangSmith traces."""

    async def test_flushes_existing_tracing_client(self) -> None:
        """The flush happens on shutdown, not on startup."""
        from langsmith import run_trees

        from deepagents_code import offload_api

        client = _flush_client()
        with patch.object(run_trees, "_CLIENT", client):
            async with offload_api.app.router.lifespan_context(offload_api.app):
                client.flush.assert_not_called()

        client.flush.assert_called_once_with()

    async def test_flushes_when_the_app_body_raises(self) -> None:
        """A crashing app still flushes without replacing its exception."""
        from langsmith import run_trees

        from deepagents_code import offload_api

        client = _flush_client()

        async def run_and_raise() -> None:
            async with offload_api.app.router.lifespan_context(offload_api.app):
                msg = "app exploded"
                raise RuntimeError(msg)

        with (
            patch.object(run_trees, "_CLIENT", client),
            pytest.raises(RuntimeError, match="app exploded"),
        ):
            await run_and_raise()

        client.flush.assert_called_once_with()

    async def test_flushes_off_the_event_loop(self) -> None:
        """The blocking flush runs on a daemon worker, not the event loop."""
        from langsmith import run_trees

        from deepagents_code import offload_api

        flush_threads: list[int] = []
        client = _flush_client()
        client.flush.side_effect = lambda: flush_threads.append(threading.get_ident())

        with patch.object(run_trees, "_CLIENT", client):
            async with offload_api.app.router.lifespan_context(offload_api.app):
                pass

        assert len(flush_threads) == 1
        assert flush_threads[0] != threading.get_ident()

    async def test_does_not_create_tracing_client(self) -> None:
        """Tracing disabled remains a no-op rather than constructing a client."""
        import langsmith
        from langsmith import run_trees

        from deepagents_code import offload_api

        def forbidden(*_args: object, **_kwargs: object) -> None:
            msg = "shutdown must not construct a LangSmith client"
            raise AssertionError(msg)

        with (
            patch.object(run_trees, "_CLIENT", None),
            patch.object(langsmith.Client, "__init__", forbidden),
        ):
            async with offload_api.app.router.lifespan_context(offload_api.app):
                pass

    async def test_hung_flush_does_not_block_shutdown(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A hung flush is abandoned after the external deadline."""
        from langsmith import run_trees

        from deepagents_code import offload_api

        release = threading.Event()
        client = _flush_client()
        client.flush.side_effect = release.wait
        existing = set(threading.enumerate())
        try:
            with (
                patch.object(run_trees, "_CLIENT", client),
                patch.object(offload_api, "_TRACE_FLUSH_TIMEOUT", 0.01),
                patch.object(offload_api, "_TRACE_FLUSH_POLL_INTERVAL", 0.001),
                caplog.at_level(logging.WARNING, logger=offload_api.__name__),
            ):
                started = time.monotonic()
                async with offload_api.app.router.lifespan_context(offload_api.app):
                    pass
                elapsed = time.monotonic() - started

            assert "some traces may be lost" in caplog.text
            assert elapsed < 0.5
            orphan = _new_flush_thread(existing)
            assert orphan.daemon
            assert orphan.is_alive()
        finally:
            release.set()
            _join_flush_threads(existing)

    async def test_flush_failure_is_logged_and_swallowed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A telemetry failure cannot turn into a failed shutdown."""
        from langsmith import run_trees

        from deepagents_code import offload_api

        client = _flush_client()
        client.flush.side_effect = RuntimeError("flush failed")
        with (
            patch.object(run_trees, "_CLIENT", client),
            caplog.at_level(logging.ERROR, logger=offload_api.__name__),
        ):
            async with offload_api.app.router.lifespan_context(offload_api.app):
                pass

        assert "Failed to flush LangSmith traces" in caplog.text
        assert "flush failed" in caplog.text

    async def test_base_exception_is_logged_and_swallowed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A non-Exception failure still releases the shutdown waiter."""
        from langsmith import run_trees

        from deepagents_code import offload_api

        client = _flush_client()
        client.flush.side_effect = SystemExit(1)
        with (
            patch.object(run_trees, "_CLIENT", client),
            caplog.at_level(logging.ERROR, logger=offload_api.__name__),
        ):
            async with offload_api.app.router.lifespan_context(offload_api.app):
                pass

        assert "Failed to flush LangSmith traces" in caplog.text
        assert "SystemExit" in caplog.text


class TestRouteRegistration:
    """The Starlette app exposes the paths and methods the client calls.

    Every other route test fabricates a request with hand-written
    `path_params`, so a path or converter rename -- `{thread_id:str}` to
    `{tid:str}`, say -- would leave the whole unit suite green while the real
    handler raised `KeyError` (neither `TypeError` nor `ValueError`, so it
    escapes the 422 block as a bare 500).
    """

    def test_offload_and_cancel_paths_are_registered(self) -> None:
        from starlette.testclient import TestClient

        from deepagents_code import offload_api
        from deepagents_code.offload_middleware import unchanged_offload_result

        calls: list[tuple[str, str]] = []

        async def fake_execute(  # noqa: RUF029  # replaces an async callee
            thread_id: str,
            *,
            operation_id: str,
            context: dict[str, object],  # noqa: ARG001
            hook_responses: dict[str, object],  # noqa: ARG001
        ) -> dict[str, object]:
            calls.append((thread_id, operation_id))
            return {
                "status": "complete",
                "result": unchanged_offload_result("noop", messages=1, tokens=5),
            }

        with (
            patch.object(offload_api, "_execute_offload", new=fake_execute),
            TestClient(offload_api.app) as client,
        ):
            response = client.post(
                "/dcode/threads/thread-42/offload",
                json={
                    "operation_id": "op-1",
                    "context": {},
                    "hook_responses": {},
                },
            )

        assert response.status_code == 200, response.text
        # The handler read the id out of the real path params, so the route's
        # converter name and the key it indexes agree.
        assert calls == [("thread-42", "op-1")]

    def test_cancel_path_is_registered(self) -> None:
        from starlette.testclient import TestClient

        from deepagents_code import offload_api

        with TestClient(offload_api.app) as client:
            response = client.post(
                "/dcode/threads/thread-42/offload/op-1/cancel",
            )

        # No such operation is active, but the route resolved and its handler
        # answered rather than 404-ing on an unmatched path.
        assert response.status_code == 200, response.text


class TestThreadLock:
    """Concurrent offloads of one thread are serialized in-process.

    The whole design is read-check-execute-recheck-write against a checkpoint.
    The per-thread lock is what makes the recheck meaningful for two requests in
    the same process: without it both can pass the status and checkpoint gates
    before either writes.
    """

    def test_each_thread_gets_its_own_lock(self) -> None:
        from deepagents_code import offload_api

        first = offload_api._thread_lock("thread-1")

        assert offload_api._thread_lock("thread-1") is first
        assert offload_api._thread_lock("thread-2") is not first

    async def test_execute_waits_for_the_threads_lock(self) -> None:
        """An offload must not touch thread state while the lock is held.

        Holding the lock externally proves the `async with` is on the path:
        remove it, or key it on `operation_id` instead of `thread_id`, and the
        operation reads state immediately.
        """
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "busy"}),
            get_state=AsyncMock(),
            update_state=AsyncMock(),
        )

        async def run() -> None:
            with contextlib.suppress(offload_api._OffloadConflictError):
                await offload_api._execute_offload(
                    "thread-1",
                    operation_id="op-2",
                    context={},
                    hook_responses={},
                )

        with patch.object(
            offload_api,
            "get_client",
            return_value=SimpleNamespace(threads=threads),
        ):
            async with offload_api._thread_lock("thread-1"):
                blocked = asyncio.create_task(run())
                for _ in range(5):
                    await asyncio.sleep(0)
                threads.get.assert_not_awaited()

            await asyncio.wait_for(blocked, timeout=5)

        threads.get.assert_awaited_once()

    async def test_a_different_thread_is_not_blocked(self) -> None:
        """The lock is per thread, so unrelated threads must not serialize."""
        from deepagents_code import offload_api

        threads = SimpleNamespace(
            get=AsyncMock(return_value={"status": "busy"}),
            get_state=AsyncMock(),
            update_state=AsyncMock(),
        )

        with patch.object(
            offload_api,
            "get_client",
            return_value=SimpleNamespace(threads=threads),
        ):
            async with offload_api._thread_lock("thread-1"):
                with pytest.raises(offload_api._OffloadConflictError):
                    await asyncio.wait_for(
                        offload_api._execute_offload(
                            "thread-2",
                            operation_id="op-1",
                            context={},
                            hook_responses={},
                        ),
                        timeout=5,
                    )

        threads.get.assert_awaited_once()


class TestOffloadRoute:
    """The HTTP layer maps operation outcomes onto distinct status codes."""

    @staticmethod
    def _request(payload: object) -> SimpleNamespace:
        """Build a minimal Starlette-like request for the route handler."""
        return SimpleNamespace(
            path_params={"thread_id": "thread-1"},
            json=AsyncMock(return_value=payload),
        )

    @staticmethod
    def _cancel_request(operation_id: str = "op-1") -> SimpleNamespace:
        """Build a minimal request for the cancellation route."""
        return SimpleNamespace(
            path_params={"thread_id": "thread-1", "operation_id": operation_id}
        )

    async def test_cancel_stops_and_joins_an_active_operation(self) -> None:
        """The cancel response is sent only after the operation task exits."""
        import json

        from deepagents_code import offload_api

        started = asyncio.Event()
        stopped = asyncio.Event()

        async def execute(*_args: object, **_kwargs: object) -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        with patch.object(
            offload_api, "_execute_offload", new=AsyncMock(side_effect=execute)
        ):
            operation = asyncio.create_task(
                offload_api.offload(
                    self._request({"operation_id": "op-1", "context": {}})  # ty: ignore[invalid-argument-type]
                )
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            response = await offload_api.cancel_offload(self._cancel_request())  # ty: ignore[invalid-argument-type]

        assert response.status_code == 200
        assert json.loads(bytes(response.body)) == {"status": "cancelled"}
        assert stopped.is_set()
        with pytest.raises(asyncio.CancelledError):
            await operation

    async def test_cancel_before_request_prevents_operation_start(self) -> None:
        """A reordered cancel closes the disconnect-before-register race."""
        import json

        from deepagents_code import offload_api

        cancel = await offload_api.cancel_offload(self._cancel_request())  # ty: ignore[invalid-argument-type]
        execute = AsyncMock()
        with patch.object(offload_api, "_execute_offload", new=execute):
            response = await offload_api.offload(
                self._request({"operation_id": "op-1", "context": {}})  # ty: ignore[invalid-argument-type]
            )

        assert json.loads(bytes(cancel.body)) == {"status": "cancelled"}
        assert response.status_code == 409
        assert "cancelled" in json.loads(bytes(response.body))["detail"]
        execute.assert_not_awaited()

    async def test_unbuildable_runtime_is_503_and_does_not_exit(self) -> None:
        """The startup barrier must not kill the process from a request handler.

        `get_server_runtime` answers a construction failure with `sys.exit(1)`,
        which is correct for the `langgraph.json` graph factory and fatal here:
        `SystemExit` is a `BaseException`, so without containment it escapes the
        route entirely and takes the server down mid-request.
        """
        import json

        from deepagents_code import offload_api

        with patch.object(
            offload_api,
            "_execute_offload",
            new=AsyncMock(
                side_effect=offload_api._OffloadUnavailableError("runtime failed")
            ),
        ):
            response = await offload_api.offload(
                self._request({"operation_id": "op-1", "context": {}})  # ty: ignore[invalid-argument-type]
            )

        assert response.status_code == 503
        assert json.loads(bytes(response.body))["detail"] == "runtime failed"
