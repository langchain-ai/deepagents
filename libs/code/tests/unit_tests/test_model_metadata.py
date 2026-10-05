"""Server metadata resolution and client failure isolation."""

import json
import tomllib
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from starlette.requests import Request

from deepagents_code.app import DeepAgentsApp
from deepagents_code.client.remote_client import RemoteAgent
from deepagents_code.config import runtime_state
from deepagents_code.model_api import model_metadata
from deepagents_code.model_config import ModelConfigError
from deepagents_code.model_metadata import ModelMetadata, ModelPurpose
from deepagents_code.workspace import WorkspaceConflictError


@pytest.fixture(autouse=True)
def clear_model_config_cache() -> Iterator[None]:
    """Keep policy loaded from temporary configs from leaking between tests."""
    from deepagents_code.model_config import clear_caches

    clear_caches()
    yield
    clear_caches()


def _request(payload: object) -> Request:
    receive = AsyncMock(
        return_value={"type": "http.request", "body": json.dumps(payload).encode()}
    )

    return Request({"type": "http", "path_params": {"thread_id": "thread"}}, receive)


@pytest.fixture
def model_runtime(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    runtime = SimpleNamespace(
        model_metadata=ModelMetadata("test", "custom", 4096, frozenset({"video"})),
        model_environment={},
    )
    monkeypatch.setattr(
        "deepagents_code.model_api.require_thread_workspace",
        AsyncMock(return_value=SimpleNamespace(cwd="/workspace")),
    )
    monkeypatch.setattr(
        "deepagents_code.server_graph._workspace_runtime",
        AsyncMock(return_value=runtime),
    )
    monkeypatch.setattr(
        "deepagents_code.server_graph._resolve_bound_workspace_config",
        AsyncMock(
            return_value=SimpleNamespace(
                profile_overrides={"max_input_tokens": 4096}, cli_max_retries=2
            )
        ),
    )
    return runtime


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
async def test_server_returns_only_metadata(
    spec: str | None, model_runtime: SimpleNamespace
) -> None:
    result = SimpleNamespace(
        model_name="test",
        provider="custom",
        context_limit=4096,
        unsupported_modalities=frozenset({"video"}),
        model=SimpleNamespace(profile=None),
    )
    with patch("deepagents_code.config.create_model", return_value=result) as create:
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
    assert json.loads(bytes(response.body)) == model_runtime.model_metadata.to_payload()
    if spec is None:
        create.assert_not_called()
    else:
        create.assert_called_once_with(
            spec,
            extra_kwargs={"temperature": 0.5},
            profile_overrides={"max_input_tokens": 4096},
            cli_max_retries=2,
        )


@pytest.mark.parametrize("environment_available", [False, True])
async def test_switch_metadata_uses_runtime_environment(
    environment_available: bool, model_runtime: SimpleNamespace
) -> None:
    from deepagents_code.config import active_environment

    model_runtime.model_environment = (
        {"MODEL_VERSION": "original"} if environment_available else None
    )

    def resolve(_spec: str, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            model_name=active_environment()["MODEL_VERSION"],
            provider="custom",
            context_limit=None,
            unsupported_modalities=frozenset(),
            model=SimpleNamespace(profile=None),
        )

    with (
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
    [
        [],
        {"model_spec": 2},
        {"extra_kwargs": []},
        {"class_path": "untrusted.Model"},
        {"model_spec": "custom:test", "purpose": []},
        {"model_spec": "custom:test", "purpose": "unknown"},
        {"purpose": "auxiliary"},
        {"model_spec": "custom:test", "purpose": "auxiliary", "extra_kwargs": {}},
    ],
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
            "deepagents_code.model_config.get_provider_auth_status",
            side_effect=AssertionError("client provider import"),
        ),
        patch(
            "deepagents_code.config.create_model",
            side_effect=AssertionError("client model construction"),
        ),
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


@pytest.mark.parametrize(
    ("command", "attribute"),
    [
        ("/summarization-model", "_summarization_model_override"),
        ("/auto model", "_auto_classifier_model"),
        ("/goal model", "_rubric_model"),
        ("/rubric model", "_rubric_model"),
    ],
)
async def test_auxiliary_selection_uses_server_environment_and_profile(
    command: str,
    attribute: str,
    model_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Selectors consume server metadata without retargeting the main model."""
    from deepagents_code.config import active_environment

    app = DeepAgentsApp()
    remote = RemoteAgent("http://test")
    app._agent = remote
    app._lc_thread_id = "thread"
    app._model_override = "custom:main"
    model_runtime.model_environment = {"MODEL_VERSION": "server"}
    monkeypatch.setattr(runtime_state, "model_name", "main")
    monkeypatch.setattr(runtime_state, "model_provider", "custom")
    monkeypatch.setattr(runtime_state, "model_context_limit", 8192)
    monkeypatch.setattr(runtime_state, "model_unsupported_modalities", frozenset())
    messages = AsyncMock()
    monkeypatch.setattr(app, "_mount_message", messages)
    monkeypatch.setattr(app, "_persist_goal_rubric_state", AsyncMock(return_value=True))
    monkeypatch.setattr(remote, "_workspace_for_thread", AsyncMock(return_value={}))

    async def post(_path: str, *, json: dict[str, object]) -> dict[str, object]:
        response = await model_metadata(_request(json))
        assert response.status_code == 200
        import json as json_module

        return json_module.loads(bytes(response.body))

    monkeypatch.setattr(
        remote,
        "_get_graph",
        Mock(
            return_value=SimpleNamespace(
                client=SimpleNamespace(http=SimpleNamespace(post=post))
            )
        ),
    )

    def resolve(spec: str, **kwargs: object) -> SimpleNamespace:
        assert spec == "alias"
        assert active_environment()["MODEL_VERSION"] == "server"
        assert kwargs["profile_overrides"] is None
        assert kwargs["cli_max_retries"] == 2
        return SimpleNamespace(
            model_name="resolved",
            provider="server_provider",
            context_limit=4096,
            unsupported_modalities=frozenset({"video"}),
            model=SimpleNamespace(profile={"structured_output": False}),
        )

    monkeypatch.setattr("deepagents_code.config.create_model", resolve)
    monkeypatch.setattr(
        "deepagents_code.config.detect_provider",
        Mock(side_effect=AssertionError("client provider inference")),
    )
    monkeypatch.setattr(
        "deepagents_code.model_config.get_provider_auth_status",
        Mock(side_effect=AssertionError("client authentication check")),
    )
    await app._handle_command(f"{command} alias")

    expected: dict[str, str | None] = {
        "_summarization_model_override": None,
        "_auto_classifier_model": None,
        "_rubric_model": None,
    }
    expected[attribute] = "server_provider:resolved"
    assert {name: getattr(app, name) for name in expected} == expected
    assert app._model_override == "custom:main"
    assert runtime_state.model_name == "main"
    assert runtime_state.model_provider == "custom"
    assert runtime_state.model_context_limit == 8192
    assert runtime_state.model_unsupported_modalities == frozenset()
    if command == "/auto model":
        assert any(
            "does not advertise structured output" in str(call.args[0]._content)
            for call in messages.await_args_list
        )


@pytest.mark.parametrize(
    "command",
    [
        "/summarization-model",
        "/auto model",
        "/goal model",
        "/rubric model",
    ],
)
async def test_auxiliary_resolution_failure_preserves_selections(
    command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No selector confirms or persists a choice the server cannot validate."""
    from deepagents_code.tui.widgets.messages import ErrorMessage

    app = DeepAgentsApp()
    remote = RemoteAgent("http://test")
    app._agent = remote
    app._lc_thread_id = "thread"
    app._summarization_model_override = "custom:summary"
    app._auto_classifier_model = "custom:classifier"
    app._rubric_model = "custom:grader"
    app._rubric_model_recorded = True
    app._server_kwargs = {"auto_classifier_model": "custom:classifier"}
    messages = AsyncMock()
    persist = AsyncMock()
    monkeypatch.setattr(app, "_mount_message", messages)
    monkeypatch.setattr(app, "_persist_goal_rubric_state", persist)
    monkeypatch.setattr(
        remote,
        "aresolve_model",
        AsyncMock(side_effect=RuntimeError("Server provider unavailable")),
    )

    await app._handle_command(f"{command} custom:unavailable")

    assert app._summarization_model_override == "custom:summary"
    assert app._auto_classifier_model == "custom:classifier"
    assert app._rubric_model == "custom:grader"
    assert app._rubric_model_recorded is True
    assert app._server_kwargs == {"auto_classifier_model": "custom:classifier"}
    persist.assert_not_awaited()
    assert any(
        isinstance(call.args[0], ErrorMessage) for call in messages.await_args_list
    )


@pytest.mark.parametrize("connecting", [False, True])
async def test_default_can_be_saved_before_server_connects(
    connecting: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = DeepAgentsApp()
    app._connecting = connecting
    monkeypatch.setattr(app, "_mount_message", AsyncMock())
    with patch(
        "deepagents_code.config.create_model",
        side_effect=AssertionError("Saving config must not construct a model"),
    ):
        await app._handle_command("/model --default :gpt-test")

    with (tmp_path / "config.toml").open("rb") as handle:
        assert tomllib.load(handle)["models"]["default"] == "openai:gpt-test"
    assert app._model_override is None


@pytest.mark.parametrize(
    ("command", "attribute"),
    [
        ("/summarization-model", "_summarization_model_override"),
        ("/auto model", "_auto_classifier_model"),
    ],
)
@pytest.mark.parametrize("allowed", [False, True])
async def test_auxiliary_choices_without_server_enforce_policy(
    command: str,
    attribute: str,
    allowed: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.tui.widgets.messages import ErrorMessage

    (tmp_path / "config.toml").write_text('[models]\nallowed = ["custom:allowed"]\n')
    app = DeepAgentsApp()
    messages = AsyncMock()
    monkeypatch.setattr(app, "_mount_message", messages)
    spec = "custom:allowed" if allowed else "custom:blocked"
    with patch(
        "deepagents_code.config.create_model",
        side_effect=AssertionError("Saving config must not construct a model"),
    ):
        await app._handle_command(f"{command} {spec}")

    assert getattr(app, attribute) == (spec if allowed else None)
    assert (
        any(isinstance(call.args[0], ErrorMessage) for call in messages.await_args_list)
        is not allowed
    )


@pytest.mark.parametrize("spec", ["custom:blocked", "unknown-model", "custom:"])
async def test_invalid_default_without_server_preserves_config(
    spec: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.tui.widgets.messages import ErrorMessage

    config = tmp_path / "config.toml"
    original = '[models]\ndefault = "custom:allowed"\nallowed = ["custom:allowed"]\n'
    config.write_text(original)
    app = DeepAgentsApp()
    messages = AsyncMock()
    monkeypatch.setattr(app, "_mount_message", messages)

    await app._handle_command(f"/model --default {spec}")

    assert config.read_text() == original
    assert any(
        isinstance(call.args[0], ErrorMessage) for call in messages.await_args_list
    )


async def test_picker_saves_default_without_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from deepagents_code.model_catalog import CatalogProvider, ModelCatalog
    from deepagents_code.model_config import ProviderAuthState

    app = DeepAgentsApp()
    monkeypatch.setattr(
        app,
        "_load_model_catalog",
        AsyncMock(
            return_value=ModelCatalog(
                models=["custom:test"],
                profiles={},
                providers={"custom": CatalogProvider(state=ProviderAuthState.MISSING)},
            )
        ),
    )
    async with app.run_test() as pilot:
        app.push_screen(app._build_model_selector_screen())
        await pilot.pause()
        await pilot.press("ctrl+s")
        await pilot.pause()

        with (tmp_path / "config.toml").open("rb") as handle:
            assert tomllib.load(handle)["models"]["default"] == "custom:test"
        assert app._model_override is None


def test_invalid_structured_output_metadata_is_rejected() -> None:
    payload = ModelMetadata("test", "custom").to_payload()
    payload["structured_output"] = "false"
    with pytest.raises(TypeError, match="structured output"):
        ModelMetadata.from_payload(payload)


@pytest.mark.parametrize("purpose", ["main", "auxiliary"])
async def test_catalog_uses_workspace_environment_and_filters_profile_extensions(
    purpose: ModelPurpose,
    model_runtime: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepagents_code.config import active_environment
    from deepagents_code.model_api import model_catalog
    from deepagents_code.model_config import (
        ModelConfig,
        ProviderAuthState,
        ProviderAuthStatus,
    )

    model_runtime.model_environment = {"MODEL_VERSION": "server"}
    monkeypatch.setattr(
        "deepagents_code.model_catalog.ModelConfig.load",
        Mock(return_value=ModelConfig()),
    )

    def available() -> dict[str, list[str]]:
        return {active_environment()["MODEL_VERSION"]: ["test"]}

    def profiles(*, cli_override: dict[str, object] | None) -> dict[str, object]:
        return {
            "server:test": {
                "profile": {
                    "max_input_tokens": cli_override["max_input_tokens"]
                    if cli_override
                    else 2048,
                    "private_extension": "must stay on the server",
                },
                "overridden_keys": frozenset({"max_input_tokens"})
                if cli_override
                else frozenset(),
            }
        }

    monkeypatch.setattr("deepagents_code.model_catalog.get_available_models", available)
    monkeypatch.setattr("deepagents_code.model_catalog.get_model_profiles", profiles)
    monkeypatch.setattr(
        "deepagents_code.model_catalog.get_provider_auth_status",
        lambda provider: ProviderAuthStatus(
            state=ProviderAuthState.NOT_REQUIRED,
            provider=provider,
        ),
    )
    remote = RemoteAgent("http://test")
    monkeypatch.setattr(remote, "_workspace_for_thread", AsyncMock(return_value={}))

    async def post(_path: str, *, json: dict[str, object]) -> dict[str, object]:
        response = await model_catalog(_request(json))
        assert response.status_code == 200
        import json as json_module

        return json_module.loads(bytes(response.body))

    monkeypatch.setattr(
        remote,
        "_get_graph",
        Mock(
            return_value=SimpleNamespace(
                client=SimpleNamespace(http=SimpleNamespace(post=post))
            )
        ),
    )
    catalog = await remote.aget_model_catalog(
        {"configurable": {"thread_id": "thread"}},
        purpose=purpose,
    )
    assert catalog.models == ["server:test"]
    assert catalog.profiles["server:test"].profile == {
        "max_input_tokens": 4096 if purpose == "main" else 2048,
    }


@pytest.mark.parametrize(
    "payload", [[], {"purpose": []}, {"recommended_models": [3]}, {"current_spec": 4}]
)
async def test_catalog_rejects_invalid_requests(payload: object) -> None:
    from deepagents_code.model_api import model_catalog

    response = await model_catalog(_request(payload))
    assert response.status_code == 422
