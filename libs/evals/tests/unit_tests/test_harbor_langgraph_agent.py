"""Tests for running Deep Agents through Harbor's built-in LangGraph agent."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_langgraph_config_points_to_deepagent_factory() -> None:
    config_path = Path("deepagents_harbor/langgraph.json")

    config = json.loads(config_path.read_text())

    assert config["dependencies"] == ["."]
    assert config["graphs"] == {
        "deepagent": "./deepagents_harbor/langgraph_agent.py:make_graph",
    }


def test_make_graph_builds_headless_local_deepagent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from deepagents_harbor import langgraph_agent

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
    assert isinstance(captured_create[0]["system_prompt"], str)
    assert "Harbor benchmark sandbox" in captured_create[0]["system_prompt"]


def test_make_graph_defaults_to_app_workdir(monkeypatch: pytest.MonkeyPatch) -> None:
    from deepagents_harbor import langgraph_agent

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
