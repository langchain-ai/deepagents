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
    # Grader-agnostic: the prompt must not reveal the agent is inside an eval.
    assert "grader" not in captured_create[0]["system_prompt"]


def test_make_graph_defaults_to_app_workdir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_create: list[dict[str, object]] = []

    # When no cwd is passed, the default workdir is used — as long as it exists. Point the
    # default at an existing tmp dir so the assertion is deterministic on any host.
    monkeypatch.setattr(langgraph_agent, "_DEFAULT_WORKDIR", tmp_path)
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

    assert captured_create[0]["cwd"] == tmp_path
    assert captured_create[0]["assistant_id"]


def test_workdir_returns_existing_configured_cwd(tmp_path: Path) -> None:
    assert langgraph_agent._workdir({"cwd": str(tmp_path)}) == tmp_path


def test_workdir_falls_back_when_configured_cwd_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # A configured cwd that does not exist in the task image would make every shell command
    # fail with FileNotFoundError; fall back to the first existing dir instead.
    monkeypatch.setattr(langgraph_agent, "_WORKDIR_FALLBACKS", (tmp_path,))
    out = langgraph_agent._workdir({"cwd": "/no/such/dir-xyz"})
    assert out == tmp_path
    assert out.is_dir()


def test_workdir_default_falls_back_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(langgraph_agent, "_DEFAULT_WORKDIR", Path("/no/such/default-xyz"))
    monkeypatch.setattr(langgraph_agent, "_WORKDIR_FALLBACKS", (tmp_path,))
    assert langgraph_agent._workdir({}) == tmp_path


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
_FIREWORKS_NEMOTRON = "fireworks:accounts/fireworks/models/nemotron-3-ultra-nvfp4"
# Nemotron 3 Ultra served by a Fireworks dedicated deployment; gets the same sampling.
_FIREWORKS_NEMOTRON_DEPLOYMENT = "fireworks:accounts/langchain-fireworks/deployments/nemotron-tb-test"
# NVIDIA's Nemotron 3 Ultra agentic-coding cookbook sampling; top_p is not a
# first-class ChatFireworks field so it rides in model_kwargs.
_FW_SAMPLING = {"temperature": 0.6, "max_tokens": 32000, "model_kwargs": {"top_p": 0.95}}
# Baseten: same cookbook sampling + reasoning ON. top_p is first-class on the
# OpenAI-compatible client so it is top-level; max_tokens is explicit because the
# server default (4096) starves the reasoning trace.
_BASETEN_DEFAULTS = {
    "temperature": 0.6,
    "max_tokens": 32000,
    "top_p": 0.95,
    "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
}


def test_resolve_init_kwargs_enables_thinking_for_baseten_nemotron() -> None:
    # Baseten serves Nemotron 3 Ultra reasoning-OFF by default; enable_thinking turns
    # it on, alongside the cookbook sampling + explicit max_tokens.
    assert langgraph_agent._resolve_init_kwargs(_BASETEN_NEMOTRON, {}) == _BASETEN_DEFAULTS


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
        "temperature": 0,  # caller wins over the 0.6 default
        "max_tokens": 32000,
        "top_p": 0.95,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}, "foo": 1},
    }
    # An explicit caller chat_template_kwargs overrides the injected default; the rest
    # of the Baseten defaults still apply.
    out2 = langgraph_agent._resolve_init_kwargs(
        _BASETEN_NEMOTRON, {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    )
    assert out2 == {
        "temperature": 0.6,
        "max_tokens": 32000,
        "top_p": 0.95,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def test_resolve_init_kwargs_does_not_mutate_caller_kwargs() -> None:
    caller = {"extra_body": {"foo": 1}}
    langgraph_agent._resolve_init_kwargs(_BASETEN_NEMOTRON, caller)
    assert caller == {"extra_body": {"foo": 1}}


def test_resolve_init_kwargs_sets_fireworks_ultra_sampling() -> None:
    # NVIDIA's Nemotron 3 Ultra cookbook recommends temp 0.6 / top_p 0.95 /
    # max_tokens 32000; the eval otherwise sends nothing and Fireworks runs at
    # its server default (~temp 1.0, unbounded output).
    assert langgraph_agent._resolve_init_kwargs(_FIREWORKS_NEMOTRON, {}) == _FW_SAMPLING


def test_resolve_init_kwargs_fireworks_caller_overrides_sampling() -> None:
    # A caller-provided sampling value wins over the spec default; the rest hold.
    out = langgraph_agent._resolve_init_kwargs(_FIREWORKS_NEMOTRON, {"temperature": 0.0})
    assert out == {"temperature": 0.0, "max_tokens": 32000, "model_kwargs": {"top_p": 0.95}}


def test_resolve_init_kwargs_sets_fireworks_deployment_ultra_sampling() -> None:
    # The Fireworks dedicated deployment gets the same cookbook sampling as the
    # serverless nvfp4 model, so it does not run at Fireworks server defaults.
    assert (
        langgraph_agent._resolve_init_kwargs(_FIREWORKS_NEMOTRON_DEPLOYMENT, {}) == _FW_SAMPLING
    )


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
    assert captured_init[0]["kwargs"] == _BASETEN_DEFAULTS
