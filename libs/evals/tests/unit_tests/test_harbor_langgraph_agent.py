"""Tests for running Deep Agents through Harbor's built-in LangGraph agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from deepagents_harbor.langgraph_project import langgraph_agent

if TYPE_CHECKING:
    import pytest


def test_langgraph_config_points_to_deepagent_factory() -> None:
    project_path = Path("deepagents_harbor/langgraph_project")
    config_path = project_path / "langgraph.json"

    config = json.loads(config_path.read_text())

    assert config["graphs"] == {
        "deepagent": "./langgraph_agent.py:make_graph",
        "bare_deepagent": "./langgraph_agent.py:make_bare_graph",
    }
    assert not (project_path / "langsmith.py").exists()


def test_langgraph_config_uses_harbor_env_for_fireworks_prereleases() -> None:
    config_path = Path("deepagents_harbor/langgraph_project/langgraph.json")

    dependencies = json.loads(config_path.read_text())["dependencies"]

    assert "langchain-fireworks>=1.4.2,<1.5.0" in dependencies
    assert not any(dependency.startswith("fireworks-ai") for dependency in dependencies)


def test_make_graph_scrubs_credentials_from_shell_backend_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph = object()
    captured_env: list[dict[str, str]] = []

    def fake_create_cli_agent(**_kwargs: object) -> tuple[object, object]:
        captured_env.append(dict(langgraph_agent.os.environ))
        return graph, object()

    monkeypatch.setattr(langgraph_agent, "init_chat_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(langgraph_agent, "create_cli_agent", fake_create_cli_agent)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("LANGSMITH_API_KEY", "secret")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("HARBOR_MODEL", "anthropic:test-model")
    monkeypatch.setenv("HARBOR_SESSION_ID", "trial-session")

    result = langgraph_agent.make_graph({"configurable": {"cwd": str(tmp_path)}})

    assert result is graph
    assert captured_env
    assert "ANTHROPIC_API_KEY" not in captured_env[0]
    assert "LANGSMITH_API_KEY" not in captured_env[0]
    assert "LANGSMITH_TRACING" not in captured_env[0]
    assert langgraph_agent.os.environ["ANTHROPIC_API_KEY"] == "secret"
    assert langgraph_agent.os.environ["LANGSMITH_API_KEY"] == "secret"
    assert langgraph_agent.os.environ["LANGSMITH_TRACING"] == "true"


def test_make_graph_builds_headless_local_deepagent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_init: list[dict[str, object]] = []
    captured_create: list[dict[str, object]] = []
    graph = object()

    def fake_init_chat_model(model: str, **kwargs: object) -> object:
        captured_init.append({"model": model, "kwargs": kwargs})
        return "chat-model"

    def fake_create_cli_agent(**kwargs: object) -> tuple[object, object]:
        captured_create.append(kwargs)
        return graph, object()

    monkeypatch.setattr(langgraph_agent, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(langgraph_agent, "create_cli_agent", fake_create_cli_agent)
    monkeypatch.setenv("HARBOR_SESSION_ID", "trial-session")

    result = langgraph_agent.make_graph(
        {
            "configurable": {
                "model": "test-provider:test-model",
                "cwd": str(tmp_path),
                "model_kwargs": {"temperature": 0.0},
            }
        }
    )

    assert result is graph
    assert captured_init == [
        {
            "model": "test-provider:test-model",
            "kwargs": {"temperature": 0.0},
        }
    ]
    assert captured_create
    assert captured_create[0]["model"] == "chat-model"
    assert captured_create[0]["assistant_id"] == "trial-session"
    assert captured_create[0]["cwd"] == tmp_path
    assert captured_create[0]["sandbox"] is None
    assert captured_create[0]["sandbox_type"] == "harbor"
    assert captured_create[0]["interactive"] is False
    assert captured_create[0]["auto_approve"] is True
    assert captured_create[0]["enable_memory"] is False
    assert captured_create[0]["enable_skills"] is False
    assert captured_create[0]["enable_shell"] is True
    # ask_user is fatal in this headless eval: a call raises GraphInterrupt with
    # no human to answer, killing the trial with empty output. It must be off.
    assert captured_create[0]["enable_ask_user"] is False
    # Finalize + anti-ramble are no longer create_cli_agent flags; they are
    # attached via the GLM-5.2 harness profile, so they must not be passed here.
    assert "enable_finalize" not in captured_create[0]
    assert "enable_anti_ramble" not in captured_create[0]
    assert isinstance(captured_create[0]["system_prompt"], str)
    assert "autonomous coding agent" in captured_create[0]["system_prompt"]


def test_make_graph_defaults_to_app_workdir(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_create: list[dict[str, object]] = []

    monkeypatch.setattr(langgraph_agent, "init_chat_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        langgraph_agent,
        "create_cli_agent",
        lambda **kwargs: (captured_create.append(kwargs) or object(), object()),
    )
    monkeypatch.delenv("HARBOR_SESSION_ID", raising=False)

    langgraph_agent.make_graph(
        {
            "configurable": {
                "model": "test-provider:test-model",
            }
        }
    )

    assert captured_create[0]["cwd"] == Path("/app")
    assert captured_create[0]["assistant_id"]


def test_make_graph_sanitizes_dotted_session_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_create: list[dict[str, object]] = []

    monkeypatch.setattr(langgraph_agent, "init_chat_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        langgraph_agent,
        "create_cli_agent",
        lambda **kwargs: (captured_create.append(kwargs) or object(), object()),
    )
    # Harbor sets HARBOR_SESSION_ID to the trial name; dotted names (e.g.
    # install-windows-3.11) are rejected by dcode's agent-name validator.
    monkeypatch.setenv("HARBOR_SESSION_ID", "install-windows-3.11__6atwQrL")

    langgraph_agent.make_graph({"configurable": {"model": "p:m", "cwd": str(tmp_path)}})

    assert captured_create[0]["assistant_id"] == "install-windows-3_11__6atwQrL"


def test_make_bare_graph_builds_sdk_deepagent_with_local_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_init: list[dict[str, object]] = []
    captured_backend: list[dict[str, object]] = []
    captured_create: list[dict[str, object]] = []
    backend = object()
    graph = object()

    def fake_init_chat_model(model: str, **kwargs: object) -> object:
        captured_init.append({"model": model, "kwargs": kwargs})
        return "chat-model"

    def fake_local_shell_backend(**kwargs: object) -> object:
        captured_backend.append(kwargs)
        return backend

    def fake_create_deep_agent(**kwargs: object) -> object:
        captured_create.append(kwargs)
        return graph

    monkeypatch.setattr(langgraph_agent, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(langgraph_agent, "LocalShellBackend", fake_local_shell_backend)
    monkeypatch.setattr(langgraph_agent, "create_deep_agent", fake_create_deep_agent)

    result = langgraph_agent.make_bare_graph(
        {
            "configurable": {
                "model": "test-provider:test-model",
                "cwd": str(tmp_path),
                "model_kwargs": {"temperature": 0.0},
            }
        }
    )

    assert result is graph
    assert captured_init == [
        {
            "model": "test-provider:test-model",
            "kwargs": {"temperature": 0.0},
        }
    ]
    assert captured_backend == [{"root_dir": tmp_path, "inherit_env": False}]
    assert captured_create
    assert captured_create[0]["model"] == "chat-model"
    assert captured_create[0]["backend"] is backend
    assert isinstance(captured_create[0]["system_prompt"], str)
    assert "autonomous coding agent" in captured_create[0]["system_prompt"]


_BASETEN_NEMOTRON = "baseten:nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B"
_ENABLE_THINKING = {"extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}


def test_resolve_init_kwargs_enables_thinking_for_baseten_nemotron() -> None:
    # Baseten serves Nemotron 3 Ultra reasoning-OFF by default; only
    # extra_body.chat_template_kwargs.enable_thinking turns it on.
    assert langgraph_agent._resolve_init_kwargs(_BASETEN_NEMOTRON, {}) == _ENABLE_THINKING


def test_resolve_init_kwargs_leaves_other_models_untouched() -> None:
    out = langgraph_agent._resolve_init_kwargs(
        "openrouter:nvidia/nemotron-3-ultra-550b-a55b", {"temperature": 0}
    )
    assert out == {"temperature": 0}


def test_resolve_init_kwargs_merges_caller_extra_body_caller_wins() -> None:
    # An unrelated caller extra_body key is preserved alongside enable_thinking.
    out = langgraph_agent._resolve_init_kwargs(
        _BASETEN_NEMOTRON, {"extra_body": {"foo": 1}, "temperature": 0}
    )
    assert out == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "foo": 1},
        "temperature": 0,
    }
    # An explicit caller chat_template_kwargs overrides the injected default.
    out2 = langgraph_agent._resolve_init_kwargs(
        _BASETEN_NEMOTRON, {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    )
    assert out2 == {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def test_resolve_init_kwargs_does_not_mutate_caller_kwargs() -> None:
    caller = {"extra_body": {"foo": 1}}
    langgraph_agent._resolve_init_kwargs(_BASETEN_NEMOTRON, caller)
    assert caller == {"extra_body": {"foo": 1}}


def test_make_graph_enables_thinking_for_baseten_nemotron(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_init: list[dict[str, object]] = []

    def fake_init_chat_model(model: str, **kwargs: object) -> object:
        captured_init.append({"model": model, "kwargs": kwargs})
        return "chat-model"

    monkeypatch.setattr(langgraph_agent, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(langgraph_agent, "create_cli_agent", lambda **_kwargs: (object(), object()))
    monkeypatch.setenv("HARBOR_SESSION_ID", "trial-session")

    langgraph_agent.make_graph({"configurable": {"model": _BASETEN_NEMOTRON, "cwd": str(tmp_path)}})

    assert captured_init[0]["model"] == _BASETEN_NEMOTRON
    assert captured_init[0]["kwargs"] == _ENABLE_THINKING
