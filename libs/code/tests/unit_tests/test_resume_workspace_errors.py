"""Network-free resume model adoption failures."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from langgraph_sdk.errors import ConflictError

from deepagents_code.app import DeepAgentsApp
from deepagents_code.client.remote_client import RemoteAgent
from deepagents_code.config import runtime_state


@pytest.mark.parametrize("same_model", [True, False])
@pytest.mark.parametrize(
    "diagnostics",
    [
        None,
        {"invalid": "secret"},
        {
            "category": "config_drift",
            "reason": "secret reason",
            "snapshot_status": "current",
            "changes": [
                {
                    "name": "interpreter_ptc",
                    "state": "changed",
                    "bound": "secret bound value",
                    "current": "secret current value",
                },
                {"name": "secret_field", "state": "values_unavailable"},
            ],
        },
    ],
)
async def test_resume_workspace_refusal(
    monkeypatch: pytest.MonkeyPatch,
    diagnostics: object,
    same_model: bool,
) -> None:
    """Binding failures remain workspace failures even with the same model."""
    current = "saved" if same_model else "fallback"
    monkeypatch.setattr(runtime_state, "model_name", current)
    monkeypatch.setattr(runtime_state, "model_provider", "openai")
    remote = RemoteAgent("http://test", local_ownership=False)
    remote.set_workspace("/project")
    response = httpx.Response(409, request=httpx.Request("POST", "http://test"))
    conflict = ConflictError(
        "secret response",
        response=response,
        body={"detail": "secret detail", "diagnostics": diagnostics},
    )
    post = AsyncMock(side_effect=conflict)
    graph = SimpleNamespace(client=SimpleNamespace(http=SimpleNamespace(post=post)))
    monkeypatch.setattr(remote, "_get_graph", lambda: graph)
    app = DeepAgentsApp(thread_id="resumed")
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    mount = AsyncMock()
    monkeypatch.setattr(app, "_mount_message", mount)
    app._should_adopt_resumed_model = True

    await app._adopt_resumed_model_if_needed(model_spec="openai:saved")

    assert mount.await_args is not None
    assert post.await_args is not None
    text = str(mount.await_args.args[0].render())
    assert "Couldn't restore this thread's workspace" in text
    assert "Model restoration was not attempted" in text
    assert "continuing on" not in text
    assert "initialized" not in text
    assert "secret" not in text
    if isinstance(diagnostics, dict) and "changes" in diagnostics:
        assert "Restore interpreter_ptc to its original value" in text
    assert post.await_count == 1
    assert post.await_args.args[0].endswith("/workspace")
    assert app._model_override is None
    assert runtime_state.model_name == current
    assert app._model_switching is False


async def test_resume_model_failure_preserves_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real model endpoint failure still reports the selected fallback."""
    monkeypatch.setattr(runtime_state, "model_name", "fallback")
    monkeypatch.setattr(runtime_state, "model_provider", "openai")
    remote = RemoteAgent("http://test", local_ownership=False)
    remote.set_workspace("/project")
    post = AsyncMock(
        side_effect=[{"workspace": {"cwd": "/project"}}, RuntimeError("model failed")]
    )
    graph = SimpleNamespace(client=SimpleNamespace(http=SimpleNamespace(post=post)))
    monkeypatch.setattr(remote, "_get_graph", lambda: graph)
    app = DeepAgentsApp(thread_id="resumed")
    monkeypatch.setattr(app, "_remote_agent", lambda: remote)
    mount = AsyncMock()
    monkeypatch.setattr(app, "_mount_message", mount)
    app._should_adopt_resumed_model = True

    await app._adopt_resumed_model_if_needed(model_spec="openai:saved")

    assert mount.await_args is not None
    assert post.await_args is not None
    text = str(mount.await_args.args[0].render())
    assert "Couldn't restore this thread's model openai:saved" in text
    assert "the model could not be initialized" in text
    assert "continuing on openai:fallback" in text
    assert post.await_count == 2
    assert post.await_args.args[0].endswith("/model")
    assert app._model_override is None
