"""Server metadata resolution and client failure isolation."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from starlette.requests import Request

from deepagents_code.app import DeepAgentsApp
from deepagents_code.client.remote_client import RemoteAgent
from deepagents_code.config import runtime_state
from deepagents_code.model_api import model_metadata
from deepagents_code.model_config import ModelConfigError
from deepagents_code.model_metadata import ModelMetadata
from deepagents_code.workspace import WorkspaceConflictError


def _request(payload: object) -> Request:
    receive = AsyncMock(
        return_value={"type": "http.request", "body": json.dumps(payload).encode()}
    )

    return Request({"type": "http", "path_params": {"thread_id": "thread"}}, receive)


async def test_startup_metadata_works_despite_thread_workspace_conflicts() -> None:
    import httpx

    from deepagents_code.offload_api import app

    reason = "bound_elsewhere"
    metadata = ModelMetadata("test", "custom", 4096, frozenset({"video"}))
    with (
        patch(
            "deepagents_code.server_graph.get_server_runtime",
            AsyncMock(return_value=SimpleNamespace(model_metadata=metadata)),
        ),
        patch(
            "deepagents_code.model_api.require_thread_workspace",
            AsyncMock(side_effect=WorkspaceConflictError(reason)),
        ),
        patch("deepagents_code.config.create_model") as create,
    ):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            startup = await client.get("/dcode/model")
            switch = await client.post(
                "/dcode/threads/thread/model",
                json={"workspace": {}, "model_spec": "custom:new"},
            )
    assert startup.status_code == 200
    assert startup.json() == metadata.to_payload()
    assert switch.status_code == 409
    assert reason in switch.json()["detail"]
    create.assert_not_called()


@pytest.mark.parametrize("error", [None, RuntimeError("unavailable"), SystemExit(1)])
async def test_startup_metadata_unavailable(error: BaseException | None) -> None:
    from deepagents_code.model_api import startup_model_metadata

    with patch(
        "deepagents_code.server_graph.get_server_runtime",
        AsyncMock(return_value=SimpleNamespace(model_metadata=None), side_effect=error),
    ):
        response = await startup_model_metadata(_request({}))
    assert response.status_code == 503


@pytest.mark.parametrize("spec", [None, "custom:test"])
async def test_server_returns_only_metadata(spec: str | None) -> None:
    metadata = ModelMetadata("test", "custom", 4096, frozenset({"video"}))
    binding = SimpleNamespace(cwd="/workspace")
    result = SimpleNamespace(
        model_name="test",
        provider="custom",
        context_limit=4096,
        unsupported_modalities=frozenset({"video"}),
        model=object(),
    )
    config = SimpleNamespace(
        profile_overrides={"max_input_tokens": 4096}, cli_max_retries=2
    )
    with (
        patch(
            "deepagents_code.model_api.require_thread_workspace",
            AsyncMock(return_value=binding),
        ),
        patch(
            "deepagents_code.server_graph._workspace_runtime",
            AsyncMock(
                return_value=SimpleNamespace(
                    model_metadata=metadata, model_environment={}
                )
            ),
        ),
        patch(
            "deepagents_code.server_graph._resolve_bound_workspace_config",
            AsyncMock(return_value=config),
        ),
        patch("deepagents_code.config.create_model", return_value=result) as create,
    ):
        response = await model_metadata(
            _request(
                {
                    "workspace": {},
                    "model_spec": spec,
                    "extra_kwargs": {"temperature": 0.5} if spec else None,
                }
            )
        )
    assert response.status_code == 200
    assert json.loads(bytes(response.body)) == metadata.to_payload()
    if spec is None:
        create.assert_not_called()
    else:
        create.assert_called_once_with(
            spec,
            extra_kwargs={"temperature": 0.5},
            profile_overrides=config.profile_overrides,
            cli_max_retries=2,
        )


@pytest.mark.parametrize("environment_available", [False, True])
async def test_switch_metadata_uses_runtime_environment(
    environment_available: bool,
) -> None:
    from deepagents_code.config import active_environment

    runtime = SimpleNamespace(
        model_environment={"MODEL_VERSION": "original"}
        if environment_available
        else None
    )

    def resolve(_spec: str, **_kwargs: object) -> ModelMetadata:
        return ModelMetadata(active_environment()["MODEL_VERSION"], "custom")

    with (
        patch(
            "deepagents_code.model_api.require_thread_workspace",
            AsyncMock(return_value=SimpleNamespace(cwd="/workspace")),
        ),
        patch(
            "deepagents_code.server_graph._workspace_runtime",
            AsyncMock(return_value=runtime),
        ),
        patch(
            "deepagents_code.server_graph._resolve_bound_workspace_config",
            AsyncMock(
                return_value=SimpleNamespace(
                    profile_overrides=None, cli_max_retries=None
                )
            ),
        ),
        patch(
            "deepagents_code.config._preview_dotenv_environ",
            return_value={"MODEL_VERSION": "edited"},
        ),
        patch.dict("os.environ", {"MODEL_VERSION": "process"}),
        patch("deepagents_code.config.create_model", side_effect=resolve) as create,
    ):
        response = await model_metadata(_request({"model_spec": "custom:test"}))

    if environment_available:
        assert response.status_code == 200
        assert json.loads(bytes(response.body))["model_name"] == "original"
    else:
        assert response.status_code == 503
        create.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [[], {"model_spec": 2}, {"extra_kwargs": []}, {"class_path": "untrusted.Model"}],
)
async def test_server_rejects_malformed_requests_before_model_creation(
    payload: object,
) -> None:
    with patch("deepagents_code.config.create_model") as create:
        response = await model_metadata(_request(payload))
    assert response.status_code == 422
    create.assert_not_called()


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ModelConfigError("provider package unavailable"), 422),
        (SystemExit(1), 503),
    ],
)
async def test_server_resolution_failures_are_contained(
    error: BaseException, status: int
) -> None:
    with (
        patch(
            "deepagents_code.model_api.require_thread_workspace",
            AsyncMock(return_value=SimpleNamespace(cwd="/workspace")),
        ),
        patch(
            "deepagents_code.server_graph._workspace_runtime",
            AsyncMock(side_effect=error),
        ),
    ):
        response = await model_metadata(_request({"workspace": {}}))
    assert response.status_code == status


async def test_remote_roundtrip_never_constructs_a_local_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = RemoteAgent("http://test")
    metadata = ModelMetadata("test", "custom", 4096, frozenset({"video"}))
    post = AsyncMock(return_value=metadata.to_payload())
    monkeypatch.setattr(
        remote, "_workspace_for_thread", AsyncMock(return_value={"id": "bound"})
    )
    monkeypatch.setattr(
        remote,
        "_get_graph",
        Mock(
            return_value=SimpleNamespace(
                client=SimpleNamespace(http=SimpleNamespace(post=post))
            )
        ),
    )
    with patch("deepagents_code.config.create_model") as create:
        result = await remote.aresolve_model(
            {"configurable": {"thread_id": "thread"}}, "custom:test"
        )
    assert result == metadata
    create.assert_not_called()
    post.assert_awaited_once_with(
        "/dcode/threads/thread/model",
        json={
            "workspace": {"id": "bound"},
            "model_spec": "custom:test",
            "extra_kwargs": None,
        },
    )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {
            "model_name": "test",
            "provider": "custom",
            "context_limit": True,
            "unsupported_modalities": [],
        },
        {
            "model_name": "test",
            "provider": "custom",
            "context_limit": 10,
            "unsupported_modalities": "video",
        },
    ],
)
def test_malformed_metadata_cannot_be_applied(payload: object) -> None:
    with pytest.raises(TypeError):
        ModelMetadata.from_payload(payload)


@pytest.mark.parametrize("same_model", [False, True])
async def test_failed_switch_preserves_all_active_state(
    same_model: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    app._agent = RemoteAgent("http://test")
    app._model_override = "custom:old"
    app._model_params_override = {"temperature": 0.2}
    monkeypatch.setattr(runtime_state, "model_name", "old")
    monkeypatch.setattr(runtime_state, "model_provider", "custom")
    monkeypatch.setattr(runtime_state, "model_context_limit", 4096)
    monkeypatch.setattr(
        runtime_state, "model_unsupported_modalities", frozenset({"video"})
    )
    monkeypatch.setattr(app, "_mount_message", AsyncMock())
    with (
        patch.object(
            RemoteAgent,
            "aresolve_model",
            AsyncMock(side_effect=ConnectionError("server unavailable")),
        ),
        patch(
            "deepagents_code.model_config.get_provider_auth_status", return_value=None
        ),
        patch("deepagents_code.model_config.save_recent_model") as save,
        patch("deepagents_code.config.create_model") as create,
    ):
        await app._switch_model(
            "custom:old" if same_model else "custom:new",
            extra_kwargs={"temperature": 0.8},
        )
    assert app._model_override == "custom:old"
    assert app._model_params_override == {"temperature": 0.2}
    assert runtime_state.model_name == "old"
    assert runtime_state.model_provider == "custom"
    assert runtime_state.model_context_limit == 4096
    assert runtime_state.model_unsupported_modalities == frozenset({"video"})
    assert not app._model_switching
    save.assert_not_called()
    create.assert_not_called()


@pytest.mark.parametrize("status", [404, 422])
async def test_remote_surfaces_server_failure(
    status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx
    from langgraph_sdk.errors import APIStatusError

    remote = RemoteAgent("http://test")
    body = {"detail": "provider is not allowed"} if status == 422 else {}
    error = APIStatusError(
        "request failed",
        response=httpx.Response(status, request=httpx.Request("POST", "http://test")),
        body=body,
    )
    monkeypatch.setattr(remote, "_workspace_for_thread", AsyncMock(return_value={}))
    monkeypatch.setattr(
        remote,
        "_get_graph",
        Mock(
            return_value=SimpleNamespace(
                client=SimpleNamespace(
                    http=SimpleNamespace(post=AsyncMock(side_effect=error))
                )
            )
        ),
    )
    with pytest.raises(
        RuntimeError,
        match="provider is not allowed" if status == 422 else "Update or restart",
    ):
        await remote.aresolve_model(
            {"configurable": {"thread_id": "thread"}}, "custom:test"
        )


async def test_switch_does_not_read_client_provider_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DeepAgentsApp()
    app._agent = RemoteAgent("http://test")
    monkeypatch.setattr(app, "_mount_message", AsyncMock())
    monkeypatch.setattr(app, "_restore_effort_override", AsyncMock())
    monkeypatch.setattr(runtime_state, "model_name", "old")
    monkeypatch.setattr(runtime_state, "model_provider", "custom")
    monkeypatch.setattr(runtime_state, "model_context_limit", None)
    monkeypatch.setattr(runtime_state, "model_unsupported_modalities", frozenset())
    with (
        patch.object(
            RemoteAgent,
            "aresolve_model",
            AsyncMock(return_value=ModelMetadata("test", "openai", 4096)),
        ),
        patch(
            "deepagents_code.model_config.get_provider_auth_status",
            side_effect=AssertionError("client provider import"),
        ),
        patch(
            "deepagents_code.config.create_model",
            side_effect=AssertionError("client model construction"),
        ),
    ):
        await app._switch_model("openai:test", persist=False)
    assert app._model_override == "openai:test"
    assert runtime_state.model_context_limit == 4096


@pytest.mark.parametrize(
    ("model_name", "client_provider", "server_provider"),
    [
        ("claude-test", "anthropic", "google_anthropic_vertex"),
        ("gemini-test", "google_genai", "google_vertexai"),
    ],
)
async def test_bare_switch_adopts_server_provider(
    model_name: str,
    client_provider: str,
    server_provider: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = DeepAgentsApp()
    remote = RemoteAgent("http://test")
    app._agent = remote
    monkeypatch.setattr(app, "_mount_message", AsyncMock())
    monkeypatch.setattr(app, "_restore_effort_override", AsyncMock())
    monkeypatch.setattr(runtime_state, "model_name", model_name)
    monkeypatch.setattr(runtime_state, "model_provider", client_provider)
    monkeypatch.setattr(runtime_state, "model_context_limit", None)
    monkeypatch.setattr(runtime_state, "model_unsupported_modalities", frozenset())

    def resolve(_config: object, spec: str, **_kwargs: object) -> ModelMetadata:
        if spec != model_name:
            msg = "Only the server's inferred provider is available."
            raise ModelConfigError(msg)
        return ModelMetadata(model_name, server_provider, 4096)

    monkeypatch.setattr(remote, "aresolve_model", AsyncMock(side_effect=resolve))
    with (
        patch("deepagents_code.config.detect_provider", return_value=client_provider),
        patch(
            "deepagents_code.model_config.save_recent_model", return_value=True
        ) as save,
        patch("deepagents_code.model_config.touch_recent_model"),
    ):
        await app._switch_model(model_name)

    resolved_spec = f"{server_provider}:{model_name}"
    assert app._model_override == resolved_spec
    assert runtime_state.model_provider == server_provider
    assert runtime_state.model_context_limit == 4096
    save.assert_called_once_with(resolved_spec)
