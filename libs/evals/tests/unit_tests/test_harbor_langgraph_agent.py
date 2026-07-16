"""Tests for running Deep Agents through Harbor's built-in LangGraph agent."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from deepagents_code.config import settings

from deepagents_harbor.langgraph_project import langgraph_agent

if TYPE_CHECKING:
    from collections.abc import Iterator

_MODEL_IDENTITY_FIELDS = (
    "model_name",
    "model_provider",
    "model_context_limit",
    "model_unsupported_modalities",
)


@pytest.fixture(autouse=True)
def _restore_model_identity_settings() -> Iterator[None]:
    """Snapshot/restore dcode's `settings` model-identity fields.

    `make_graph` writes these process-level singleton fields (so the system
    prompt's Model Identity section is populated). Without restoring them, a
    test that runs `make_graph` would leak the model identity into later tests.
    """
    saved = {field: getattr(settings, field) for field in _MODEL_IDENTITY_FIELDS}
    try:
        yield
    finally:
        for field, value in saved.items():
            setattr(settings, field, value)


def test_langgraph_config_points_to_deepagent_factory() -> None:
    project_path = Path("deepagents_harbor/langgraph_project")
    config_path = project_path / "langgraph.json"

    config = json.loads(config_path.read_text())

    assert config["graphs"] == {
        "deepagent": "./langgraph_agent.py:make_graph",
        "bare_deepagent": "./langgraph_agent.py:make_bare_graph",
        "tau3_deepagent": "./langgraph_agent.py:make_tau3_graph",
    }
    assert not (project_path / "langsmith.py").exists()


def test_langgraph_config_uses_harbor_env_for_fireworks_prereleases() -> None:
    config_path = Path("deepagents_harbor/langgraph_project/langgraph.json")

    dependencies = json.loads(config_path.read_text())["dependencies"]

    assert "langchain-fireworks>=1.4.2,<1.5.0" in dependencies
    assert not any(dependency.startswith("fireworks-ai") for dependency in dependencies)


def test_harbor_assistant_id_preserves_valid_ascii_id() -> None:
    assistant_id = f"-{'a' * 62}-"

    assert len(assistant_id) == 64
    assert langgraph_agent._harbor_assistant_id(assistant_id) == assistant_id


def test_harbor_assistant_id_hashes_valid_id_over_limit() -> None:
    session_id = f"-{'a' * 63}-"
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]

    assistant_id = langgraph_agent._harbor_assistant_id(session_id)

    assert len(session_id) == 65
    assert len(assistant_id) <= 64
    assert assistant_id.startswith(session_id[:16])
    assert assistant_id.endswith(f"-{digest}")


def test_harbor_assistant_id_normalizes_path_and_control_characters() -> None:
    session_id = "/install/windows::3.11\x00\ntrial/"
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]

    assert (
        langgraph_agent._harbor_assistant_id(session_id) == f"install-windows-3-11-trial-{digest}"
    )


def test_harbor_assistant_id_distinguishes_raw_ids_with_same_normalization() -> None:
    slash_id = "trial/session"
    colon_id = "trial:session"

    normalized_slash = langgraph_agent._harbor_assistant_id(slash_id)
    normalized_colon = langgraph_agent._harbor_assistant_id(colon_id)

    assert normalized_slash == (
        f"trial-session-{hashlib.sha256(slash_id.encode('utf-8')).hexdigest()[:12]}"
    )
    assert normalized_colon == (
        f"trial-session-{hashlib.sha256(colon_id.encode('utf-8')).hexdigest()[:12]}"
    )
    assert normalized_slash != normalized_colon


@pytest.mark.parametrize("session_id", [None, "", "/:.\x00\n"])
def test_harbor_assistant_id_falls_back_to_uuid(
    monkeypatch: pytest.MonkeyPatch, session_id: str | None
) -> None:
    fixed_uuid = "00000000-0000-4000-8000-000000000000"
    monkeypatch.setattr(langgraph_agent.uuid, "uuid4", lambda: fixed_uuid)

    assert langgraph_agent._harbor_assistant_id(session_id) == f"harbor-{fixed_uuid}"


def test_harbor_assistant_id_bounds_long_ids_without_truncation_collisions() -> None:
    shared_prefix = "a" * 300

    first = langgraph_agent._harbor_assistant_id(f"{shared_prefix}x")
    second = langgraph_agent._harbor_assistant_id(f"{shared_prefix}y")

    assert len(first) <= 64
    assert first == langgraph_agent._harbor_assistant_id(f"{shared_prefix}x")
    assert first != second
    assert all(
        character.isascii() and (character.isalnum() or character in "_-") for character in first
    )


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
    session_id = "install-windows-3.11__trial__env"
    monkeypatch.setenv("HARBOR_SESSION_ID", session_id)

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
    digest = hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]
    assert captured_create[0]["assistant_id"] == f"install-windows-3-11__trial__env-{digest}"
    assert captured_create[0]["cwd"] == tmp_path
    assert captured_create[0]["sandbox"] is None
    # `make_graph` must NOT pass `sandbox_type`: it runs locally (sandbox=None), and
    # a non-None sandbox_type routes get_system_prompt through
    # get_default_working_dir(), which raises for unregistered providers like
    # "harbor". Omitting it selects the local-mode prompt rooted at `cwd`.
    assert "sandbox_type" not in captured_create[0]
    assert captured_create[0]["interactive"] is False
    assert captured_create[0]["enable_ask_user"] is False
    assert captured_create[0]["auto_approve"] is True
    assert captured_create[0]["enable_memory"] is False
    assert captured_create[0]["enable_skills"] is False
    assert captured_create[0]["enable_shell"] is True
    # `make_graph` must NOT pass a system prompt: create_cli_agent then builds the
    # real dcode production prompt (via get_system_prompt), which is what the CLI
    # harness eval is supposed to evaluate. Passing an override would bypass it.
    assert "system_prompt" not in captured_create[0]


def test_make_graph_populates_model_identity_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`make_graph` must feed `configurable.model` into dcode `settings`.

    create_cli_agent -> get_system_prompt renders the prompt's Model Identity
    section from these settings, so without this wiring the eval agent's prompt
    would omit which model it is running as.
    """

    class _Model:
        # Shape mirrors a langchain model profile dict.
        profile = {  # noqa: RUF012
            "max_input_tokens": 200_000,
            "image_inputs": True,
            "audio_inputs": False,
            "video_inputs": False,
            "pdf_inputs": True,
        }

    monkeypatch.setattr(langgraph_agent, "init_chat_model", lambda *_a, **_k: _Model())
    monkeypatch.setattr(
        langgraph_agent,
        "create_cli_agent",
        lambda **_kwargs: (object(), object()),
    )
    monkeypatch.setenv("HARBOR_SESSION_ID", "trial-session")

    langgraph_agent.make_graph(
        {
            "configurable": {
                "model": "anthropic:claude-sonnet-4-5",
                "cwd": str(tmp_path),
            }
        }
    )

    assert settings.model_name == "claude-sonnet-4-5"
    assert settings.model_provider == "anthropic"
    assert settings.model_context_limit == 200_000
    assert settings.model_unsupported_modalities == frozenset({"audio", "video"})


def test_make_graph_resets_model_identity_for_model_without_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings.model_context_limit = 200_000
    settings.model_unsupported_modalities = frozenset({"audio", "video"})

    monkeypatch.setattr(langgraph_agent, "init_chat_model", lambda *_a, **_k: object())
    monkeypatch.setattr(
        langgraph_agent,
        "create_cli_agent",
        lambda **_kwargs: (object(), object()),
    )
    monkeypatch.setenv("HARBOR_SESSION_ID", "trial-session")

    langgraph_agent.make_graph(
        {
            "configurable": {
                "model": "ollama:local-model",
                "cwd": str(tmp_path),
            }
        }
    )

    assert settings.model_context_limit is None
    assert settings.model_unsupported_modalities == frozenset()


@pytest.mark.parametrize(
    ("model_spec", "expected_name", "expected_provider"),
    [
        ("claude-sonnet-4-5", "claude-sonnet-4-5", "anthropic"),
        ("totally-unknown-xyz", "totally-unknown-xyz", ""),
    ],
)
def test_make_graph_derives_identity_for_bare_model_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_spec: str,
    expected_name: str,
    expected_provider: str,
) -> None:
    """A bare (provider-less) model spec falls back to `detect_provider`.

    Exercises the `_apply_model_identity` else-branch: `ModelSpec.try_parse`
    returns None for a colon-less spec, so the name is taken verbatim and the
    provider is inferred by `detect_provider` (empty string when unknown).
    """
    monkeypatch.setattr(langgraph_agent, "init_chat_model", lambda *_a, **_k: object())
    monkeypatch.setattr(
        langgraph_agent,
        "create_cli_agent",
        lambda **_kwargs: (object(), object()),
    )
    monkeypatch.setenv("HARBOR_SESSION_ID", "trial-session")

    langgraph_agent.make_graph(
        {
            "configurable": {
                "model": model_spec,
                "cwd": str(tmp_path),
            }
        }
    )

    assert settings.model_name == expected_name
    assert settings.model_provider == expected_provider


@pytest.mark.parametrize(
    "model_spec",
    [
        "fireworks:accounts/fireworks/models/glm-5p2",
        "openrouter:z-ai/glm-5.2",
        "baseten:zai-org/GLM-5.2",
    ],
)
def test_make_graph_defaults_glm_reasoning_to_high(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_spec: str
) -> None:
    """`make_graph` defaults GLM-5.2 reasoning to `high` in nested model kwargs."""
    captured_kwargs: list[dict[str, object]] = []

    def fake_init_chat_model(_model: str, **kwargs: object) -> object:
        captured_kwargs.append(kwargs)
        return "chat-model"

    monkeypatch.setattr(langgraph_agent, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(langgraph_agent, "create_cli_agent", lambda **_k: (object(), object()))
    monkeypatch.setenv("HARBOR_SESSION_ID", "trial-session")

    langgraph_agent.make_graph(
        {
            "configurable": {
                "model": model_spec,
                "cwd": str(tmp_path),
            }
        }
    )

    assert captured_kwargs[0].get("model_kwargs") == {"reasoning_effort": "high"}


def test_make_graph_preserves_explicit_glm_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_kwargs: list[dict[str, object]] = []

    def fake_init_chat_model(_model: str, **kwargs: object) -> object:
        captured_kwargs.append(kwargs)
        return "chat-model"

    monkeypatch.setattr(langgraph_agent, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(langgraph_agent, "create_cli_agent", lambda **_k: (object(), object()))
    monkeypatch.setenv("HARBOR_SESSION_ID", "trial-session")

    langgraph_agent.make_graph(
        {
            "configurable": {
                "model": "fireworks:accounts/fireworks/models/glm-5p2",
                "cwd": str(tmp_path),
                "model_kwargs": {"model_kwargs": {"reasoning_effort": "max"}},
            }
        }
    )

    assert captured_kwargs[0].get("model_kwargs") == {"reasoning_effort": "max"}


def test_make_graph_preserves_top_level_glm_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_kwargs: list[dict[str, object]] = []

    def fake_init_chat_model(_model: str, **kwargs: object) -> object:
        captured_kwargs.append(kwargs)
        return "chat-model"

    monkeypatch.setattr(langgraph_agent, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(langgraph_agent, "create_cli_agent", lambda **_k: (object(), object()))
    monkeypatch.setenv("HARBOR_SESSION_ID", "trial-session")

    langgraph_agent.make_graph(
        {
            "configurable": {
                "model": "fireworks:accounts/fireworks/models/glm-5p2",
                "cwd": str(tmp_path),
                "model_kwargs": {"reasoning_effort": "max"},
            }
        }
    )

    assert captured_kwargs[0] == {"reasoning_effort": "max"}


@pytest.mark.parametrize(
    ("model_kwargs", "expected"),
    [
        ({"reasoning": {"effort": "max"}}, {"reasoning": {"effort": "max"}}),
        (
            {"model_kwargs": {"reasoning": {"effort": "max"}}},
            {"model_kwargs": {"reasoning": {"effort": "max"}}},
        ),
    ],
)
def test_make_graph_preserves_provider_native_glm_reasoning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    model_kwargs: dict[str, object],
    expected: dict[str, object],
) -> None:
    captured_kwargs: list[dict[str, object]] = []

    def fake_init_chat_model(_model: str, **kwargs: object) -> object:
        captured_kwargs.append(kwargs)
        return "chat-model"

    monkeypatch.setattr(langgraph_agent, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(langgraph_agent, "create_cli_agent", lambda **_k: (object(), object()))
    monkeypatch.setenv("HARBOR_SESSION_ID", "trial-session")

    langgraph_agent.make_graph(
        {
            "configurable": {
                "model": "openrouter:z-ai/glm-5.2",
                "cwd": str(tmp_path),
                "model_kwargs": model_kwargs,
            }
        }
    )

    assert captured_kwargs[0] == expected


@pytest.mark.parametrize(
    "model_spec",
    [
        "openrouter:vendor/glm-5.20",
        "openrouter:z-ai/glm-5.2-vision",
    ],
)
def test_make_graph_leaves_other_glm_model_kwargs_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, model_spec: str
) -> None:
    captured_kwargs: list[dict[str, object]] = []

    def fake_init_chat_model(_model: str, **kwargs: object) -> object:
        captured_kwargs.append(kwargs)
        return "chat-model"

    monkeypatch.setattr(langgraph_agent, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(langgraph_agent, "create_cli_agent", lambda **_k: (object(), object()))
    monkeypatch.setenv("HARBOR_SESSION_ID", "trial-session")

    langgraph_agent.make_graph(
        {
            "configurable": {
                "model": model_spec,
                "cwd": str(tmp_path),
            }
        }
    )

    assert captured_kwargs[0] == {}


def test_make_graph_leaves_non_glm_model_kwargs_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_kwargs: list[dict[str, object]] = []

    def fake_init_chat_model(_model: str, **kwargs: object) -> object:
        captured_kwargs.append(kwargs)
        return "chat-model"

    monkeypatch.setattr(langgraph_agent, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(langgraph_agent, "create_cli_agent", lambda **_k: (object(), object()))
    monkeypatch.setenv("HARBOR_SESSION_ID", "trial-session")

    langgraph_agent.make_graph(
        {
            "configurable": {
                "model": "anthropic:claude-x",
                "cwd": str(tmp_path),
                "model_kwargs": {"temperature": 0.0},
            }
        }
    )

    assert captured_kwargs[0] == {"temperature": 0.0}


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


def test_make_graph_openai_defaults_to_responses_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # OpenAI gates reasoning_effort + function tools to /v1/responses for
    # gpt-5.x; the agent builds the model directly, so it must set
    # use_responses_api=True itself.
    captured_init: list[dict[str, object]] = []

    def fake_init_chat_model(model: str, **kwargs: object) -> object:
        captured_init.append({"model": model, "kwargs": kwargs})
        return "chat-model"

    monkeypatch.setattr(langgraph_agent, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(langgraph_agent, "create_cli_agent", lambda **_kwargs: (object(), object()))

    langgraph_agent.make_graph(
        {"configurable": {"model": "openai:gpt-5.6-luna", "cwd": str(tmp_path)}}
    )

    assert captured_init == [
        {"model": "openai:gpt-5.6-luna", "kwargs": {"use_responses_api": True}}
    ]


def test_build_model_respects_explicit_use_responses_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_init: list[dict[str, object]] = []

    monkeypatch.setattr(
        langgraph_agent,
        "init_chat_model",
        lambda model, **kwargs: captured_init.append({"model": model, "kwargs": kwargs}),
    )

    langgraph_agent._build_model(
        {"model": "openai:gpt-5", "model_kwargs": {"use_responses_api": False}}
    )

    assert captured_init == [{"model": "openai:gpt-5", "kwargs": {"use_responses_api": False}}]


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
    assert "Harbor benchmark sandbox" in captured_create[0]["system_prompt"]


def test_mcp_connections_maps_streamable_http_server() -> None:
    connections = langgraph_agent._mcp_connections(
        {
            "mcp_servers": [
                {
                    "name": "tau3-runtime",
                    "transport": "streamable-http",
                    "url": "http://tau3-runtime:8000/mcp",
                    "command": None,
                    "args": [],
                }
            ]
        }
    )

    assert connections == {
        "tau3-runtime": {
            "transport": "streamable_http",
            "url": "http://tau3-runtime:8000/mcp",
        }
    }


def test_mcp_connections_rejects_stdio_servers() -> None:
    # A dataset-provided stdio server would run an arbitrary local command in the
    # agent sandbox; the tau3 graph must reject it rather than execute it.
    with pytest.raises(ValueError, match="stdio"):
        langgraph_agent._mcp_connections(
            {
                "mcp_servers": [
                    {
                        "name": "evil",
                        "transport": "stdio",
                        "command": "/bin/sh",
                        "args": ["-c", "exfiltrate-secrets"],
                    }
                ]
            }
        )


def test_mcp_connections_requires_forwarded_servers() -> None:
    with pytest.raises(ValueError, match="mcp_servers"):
        langgraph_agent._mcp_connections({})
