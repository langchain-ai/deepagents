"""Tests for Harbor's in-process Switchyard library configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest

from deepagents_harbor.langgraph_project import switchyard_library

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.language_models import BaseChatModel


class _RecordingAlgorithm:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    async def run(
        self,
        request: object,
        headers: object = None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        self.calls.append((request, headers))
        return ([{"selected_model": "weak"}], {"model": "weak", "outputs": []})


class _RecordingAlgorithms:
    def __init__(self) -> None:
        self.native = _RecordingAlgorithm()
        self.passthrough_target: object | None = None
        self.escalation: tuple[tuple[object, ...], dict[str, object]] | None = None

    def passthrough(self, target: object) -> _RecordingAlgorithm:
        self.passthrough_target = target
        return self.native

    def llm_escalation(
        self,
        *targets: object,
        **kwargs: object,
    ) -> _RecordingAlgorithm:
        self.escalation = (targets, kwargs)
        return self.native


@dataclass
class _RecordingMiddleware:
    algorithm: object


def _write_route(runtime: Path, name: str, content: str) -> None:
    routes = runtime / "routes"
    routes.mkdir(parents=True)
    (routes / f"routes-{name}.toml").write_text(content)


def _bindings(
    algorithms: _RecordingAlgorithms,
) -> switchyard_library._Bindings:
    return switchyard_library._Bindings(
        llm_target=lambda name, client: {"name": name, "client": client},
        client=lambda model: {"model": model},
        middleware=lambda algorithm: cast(
            "AgentMiddleware[Any, Any, Any]", _RecordingMiddleware(algorithm)
        ),
        algorithms=cast("switchyard_library._Algorithms", algorithms),
    )


async def test_passthrough_builds_openai_target_and_forwards_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_route(
        tmp_path,
        "nano",
        """
[llm_clients.nvidia]
format = "openai_chat"
base_url = "https://nvidia.example/v1"
api_key_env = "NVIDIA_API_KEY"

[targets.nano]
id = "private/nvidia/nano"
llm_client = "nvidia"
extra_body = { chat_template_kwargs = { enable_thinking = true } }

[routes.switchyard]
type = "passthrough"
target = "nano"
""",
    )
    algorithms = _RecordingAlgorithms()
    models: list[tuple[str, dict[str, object], object]] = []

    def fake_model(model: str, **kwargs: object) -> BaseChatModel:
        built = object()
        models.append((model, kwargs, built))
        return cast("BaseChatModel", built)

    monkeypatch.setenv("NVIDIA_API_KEY", "fake-nvidia-key")
    monkeypatch.setattr(switchyard_library, "init_chat_model", fake_model)
    monkeypatch.setattr(
        switchyard_library,
        "_load_bindings",
        lambda _runtime: _bindings(algorithms),
    )

    components = switchyard_library.build_switchyard_components(
        "nano",
        "trial-1__env",
        runtime_dir=tmp_path,
    )

    assert components.model is models[0][2]
    assert models[0][:2] == (
        "openai:private/nvidia/nano",
        {
            "api_key": "fake-nvidia-key",
            "base_url": "https://nvidia.example/v1",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
            "use_responses_api": False,
        },
    )
    assert algorithms.passthrough_target == {
        "name": "nano",
        "client": {"model": models[0][2]},
    }
    middleware = cast("_RecordingMiddleware", components.middleware)
    await cast("switchyard_library._SessionAlgorithm", middleware.algorithm).run(
        {"messages": []}
    )
    assert algorithms.native.calls == [
        ({"messages": []}, {"x-switchyard-session-id": "trial-1__env"})
    ]


async def test_escalation_builds_mixed_provider_targets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_route(
        tmp_path,
        "opus-nano",
        """
[llm_clients.nvidia]
format = "openai_chat"
base_url = "https://nvidia.example/v1"
api_key_env = "NVIDIA_API_KEY"

[llm_clients.anthropic]
format = "anthropic_messages"
base_url = "https://anthropic.example"
api_key_env = "ANTHROPIC_API_KEY"

[llm_clients.google]
format = "openai_chat"
base_url = "https://google.example/v1"
api_key_env = "GOOGLE_API_KEY"

[targets.weak]
id = "private/nvidia/nano"
llm_client = "nvidia"

[targets.strong]
id = "claude-opus-test"
llm_client = "anthropic"

[targets.judge]
id = "gemini-flash-test"
llm_client = "google"

[routes.switchyard]
type = "llm_classifier"
mode = "escalation"
classifier_target = "judge"
weak_target = "weak"
strong_target = "strong"

[routes.switchyard.escalation]
confirmations = 3
recent_turn_window = 20
window_message_chars = 600
max_output_tokens = 1024
""",
    )
    algorithms = _RecordingAlgorithms()
    models: list[tuple[str, dict[str, object], object]] = []

    def fake_model(model: str, **kwargs: object) -> BaseChatModel:
        built = object()
        models.append((model, kwargs, built))
        return cast("BaseChatModel", built)

    monkeypatch.setenv("NVIDIA_API_KEY", "fake-nvidia-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
    monkeypatch.setattr(switchyard_library, "init_chat_model", fake_model)
    monkeypatch.setattr(
        switchyard_library,
        "_load_bindings",
        lambda _runtime: _bindings(algorithms),
    )

    components = switchyard_library.build_switchyard_components(
        "opus-nano",
        "trial-2__env",
        runtime_dir=tmp_path,
    )

    assert [model for model, _kwargs, _built in models] == [
        "openai:gemini-flash-test",
        "openai:private/nvidia/nano",
        "anthropic:claude-opus-test",
    ]
    assert components.model is models[1][2]
    assert models[2][1] == {
        "api_key": "fake-anthropic-key",
        "base_url": "https://anthropic.example",
    }
    assert algorithms.escalation is not None
    targets, kwargs = algorithms.escalation
    assert [cast("dict[str, object]", target)["name"] for target in targets] == [
        "judge",
        "weak",
        "strong",
    ]
    assert kwargs == {
        "confirmations": 3,
        "recent_turn_window": 20,
        "window_message_chars": 600,
        "max_output_tokens": 1024,
    }
    middleware = cast("_RecordingMiddleware", components.middleware)
    await cast("switchyard_library._SessionAlgorithm", middleware.algorithm).run(
        {"messages": []}
    )
    assert algorithms.native.calls[0][1] == {
        "x-switchyard-session-id": "trial-2__env"
    }


@pytest.mark.parametrize("name", ["../nano", "/nano", "nano.toml", ""])
def test_config_name_rejects_path_traversal(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError, match="Invalid Switchyard config name"):
        switchyard_library.build_switchyard_components(
            name,
            "trial-session",
            runtime_dir=tmp_path,
        )


def test_session_id_is_required_before_runtime_loading(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="HARBOR_SESSION_ID"):
        switchyard_library.build_switchyard_components(
            "nano",
            "",
            runtime_dir=tmp_path,
        )


def test_missing_provider_key_fails_without_model_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_route(
        tmp_path,
        "nano",
        """
[llm_clients.nvidia]
format = "openai_chat"
base_url = "https://nvidia.example/v1"
api_key_env = "NVIDIA_API_KEY"

[targets.nano]
id = "private/nvidia/nano"
llm_client = "nvidia"

[routes.switchyard]
type = "passthrough"
target = "nano"
""",
    )
    algorithms = _RecordingAlgorithms()
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(
        switchyard_library,
        "_load_bindings",
        lambda _runtime: _bindings(algorithms),
    )

    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        switchyard_library.build_switchyard_components(
            "nano",
            "trial-session",
            runtime_dir=tmp_path,
        )
