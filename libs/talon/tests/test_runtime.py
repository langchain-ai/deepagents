from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol, cast

import pytest
from deepagents.backends import LocalShellBackend
from langchain.agents.middleware.types import AgentMiddleware

from deepagents_talon.cron import CronJobStore
from deepagents_talon.interfaces import (
    AgentRequest,
    ToolApprovalDecision,
    ToolApprovalRequest,
)
from deepagents_talon.runtime import (
    _SAFE_BACKEND_PATH,
    INTERRUPT_ON_TOOLS_ENV_KEY,
    DeepAgentRuntime,
    RuntimeAgentComponents,
    _is_retryable,
    interrupt_on_with_env_overlay,
)

if TYPE_CHECKING:
    from pathlib import Path

    from langgraph.types import Command


class InvokableTool(Protocol):
    def invoke(self, payload: dict[str, object]) -> dict[str, object]:
        """Invoke a tool with a structured payload."""


class PassthroughMiddleware(AgentMiddleware):
    """Middleware stub for runtime wiring assertions."""


class RecordingGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.history: dict[str, list[object]] = {}

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((payload, config))
        thread_id = config["configurable"]["thread_id"]
        messages = self.history.setdefault(thread_id, [])
        messages.extend(payload["messages"])
        response = SimpleNamespace(content=f"seen:{len(messages)}")
        messages.append(response)
        return {"messages": list(messages)}


class CronCallingGraph:
    def __init__(self, create_job: InvokableTool) -> None:
        self.create_job = create_job

    async def ainvoke(
        self,
        _payload: dict[str, Any],
        config: dict[str, Any],  # noqa: ARG002  # Matches graph invocation signature.
    ) -> dict[str, Any]:
        result = self.create_job.invoke({"prompt": "later", "schedule": "in 5m"})
        return {"messages": [SimpleNamespace(content=result["id"])]}


class InterruptingGraph:
    def __init__(self) -> None:
        self.calls: list[tuple[object, dict[str, Any]]] = []
        self.executed = False

    async def ainvoke(self, payload: object, config: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((payload, config))
        if len(self.calls) == 1:
            return {
                "messages": [SimpleNamespace(content="")],
                "__interrupt__": [
                    SimpleNamespace(
                        id="interrupt-1",
                        value={
                            "action_requests": [
                                {
                                    "name": "dangerous_tool",
                                    "args": {"path": "/secret"},
                                }
                            ],
                            "review_configs": [],
                        },
                    )
                ],
            }

        resume = getattr(payload, "resume", {})
        decision = resume["interrupt-1"]["decisions"][0]
        self.executed = decision["type"] == "approve"
        content = "approved" if self.executed else "denied"
        return {"messages": [SimpleNamespace(content=content)]}


class AuthFailingGraph:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls: list[tuple[object, dict[str, Any]]] = []

    async def ainvoke(self, payload: object, config: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((payload, config))
        raise self.error


class StatusError(Exception):
    def __init__(self, status_code: int, message: str = "request failed") -> None:
        super().__init__(message)
        self.status_code = status_code


def custom_tool() -> str:
    """Custom runtime tool."""
    return "ok"


def fetch_url() -> str:
    """Fetch URL tool stub."""
    return "fetched"


def web_search() -> str:
    """Web search tool stub."""
    return "searched"


async def test_runtime_wires_backend_checkpointer_tools_skills_and_memory(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    graph = RecordingGraph()
    assistant_dir = tmp_path / "assistant"
    assistant_dir.mkdir()
    (assistant_dir / "AGENTS.md").write_text("assistant instructions", encoding="utf-8")

    def fake_create_deep_agent(**kwargs: Any) -> RecordingGraph:
        captured.update(kwargs)
        return graph

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", fake_create_deep_agent)
    monkeypatch.setattr("deepagents_talon.runtime.fetch_url", fetch_url)
    monkeypatch.setattr("deepagents_talon.runtime.web_search", web_search)
    monkeypatch.chdir(tmp_path)

    runtime = DeepAgentRuntime(
        model="test:model",
        tools=[custom_tool],
        assistant_dir=assistant_dir,
        cron_store=CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron"),
    )

    await runtime.start()

    assert isinstance(captured["backend"], LocalShellBackend)
    assert captured["checkpointer"] is runtime.checkpointer
    assert captured["system_prompt"] == "assistant instructions"
    assert captured["skills"] == [str(assistant_dir / "skills")]
    assert captured["memory"] == [str(assistant_dir / "memory" / "AGENTS.md")]
    assert (assistant_dir / "memory" / "AGENTS.md").is_file()
    assert captured["backend"].cwd == tmp_path.resolve()

    tool_names = {_tool_name(tool) for tool in captured["tools"]}
    assert {
        "fetch_url",
        "web_search",
        "create_job",
        "list_jobs",
        "edit_job",
        "remove_job",
        "custom_tool",
    } <= tool_names


async def test_runtime_wires_subagents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    subagents = [
        {
            "name": "researcher",
            "description": "Research tasks",
            "system_prompt": "Research carefully.",
        },
    ]

    def fake_create_deep_agent(**kwargs: Any) -> RecordingGraph:
        captured.update(kwargs)
        return RecordingGraph()

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", fake_create_deep_agent)

    runtime = DeepAgentRuntime(
        model="test:model",
        subagents=cast("Any", subagents),
        include_web_tools=False,
        skills=(),
        memory=(),
    )

    await runtime.start()

    assert captured["subagents"] == subagents


async def test_runtime_loads_subagents_from_assistant_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    assistant_dir = tmp_path / "assistant"
    subagent_dir = assistant_dir / "subagents" / "researcher"
    subagent_dir.mkdir(parents=True)
    (subagent_dir / "AGENTS.md").write_text("Research carefully.", encoding="utf-8")
    (subagent_dir / "config.json").write_text(
        json.dumps({"description": "Research tasks", "model": "research:model"}),
        encoding="utf-8",
    )

    def fake_create_deep_agent(**kwargs: Any) -> RecordingGraph:
        captured.update(kwargs)
        return RecordingGraph()

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", fake_create_deep_agent)

    runtime = DeepAgentRuntime(
        model="test:model",
        assistant_dir=assistant_dir,
        include_web_tools=False,
        skills=(),
        memory=(),
    )

    await runtime.start()

    assert captured["subagents"] == [
        {
            "name": "researcher",
            "description": "Research tasks",
            "system_prompt": "Research carefully.",
            "model": "research:model",
        }
    ]


async def test_runtime_passes_middleware_to_create_deep_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    middleware = PassthroughMiddleware()

    def fake_create_deep_agent(**kwargs: Any) -> RecordingGraph:
        captured.update(kwargs)
        return RecordingGraph()

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", fake_create_deep_agent)

    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
        middleware=(middleware,),
    )

    await runtime.start()

    assert captured["middleware"] == [middleware]


async def test_runtime_passes_interrupt_on_to_create_deep_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    interrupt_on = {"custom_tool": True}

    def fake_create_deep_agent(**kwargs: Any) -> RecordingGraph:
        captured.update(kwargs)
        return RecordingGraph()

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", fake_create_deep_agent)

    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
        interrupt_on=interrupt_on,
    )

    await runtime.start()

    assert captured["interrupt_on"] == interrupt_on


def test_interrupt_on_with_env_overlay_preserves_empty_behavior() -> None:
    interrupt_on = {"fleet_tool": True}

    assert interrupt_on_with_env_overlay(None, {}) is None
    assert interrupt_on_with_env_overlay(None, {INTERRUPT_ON_TOOLS_ENV_KEY: " , "}) is None
    assert interrupt_on_with_env_overlay(interrupt_on, {}) == interrupt_on


def test_interrupt_on_with_env_overlay_parses_comma_whitespace() -> None:
    interrupt_on = interrupt_on_with_env_overlay(
        {"fleet_tool": True},
        {INTERRUPT_ON_TOOLS_ENV_KEY: " bash,execute, , github_create_pr ,custom/mcp "},
    )

    assert interrupt_on == {
        "fleet_tool": True,
        "bash": True,
        "execute": True,
        "github_create_pr": True,
        "custom/mcp": True,
    }


async def test_runtime_adds_env_interrupt_on_overlay_to_create_deep_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> RecordingGraph:
        captured.update(kwargs)
        return RecordingGraph()

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", fake_create_deep_agent)

    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
        interrupt_on={"fleet_tool": True},
        env={INTERRUPT_ON_TOOLS_ENV_KEY: "bash, execute"},
    )

    await runtime.start()

    assert captured["interrupt_on"] == {
        "fleet_tool": True,
        "bash": True,
        "execute": True,
    }


async def test_runtime_uses_configured_workspace_for_default_backend(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> RecordingGraph:
        captured.update(kwargs)
        return RecordingGraph()

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", fake_create_deep_agent)

    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
        env={"DEEPAGENTS_TALON_WORKSPACE": str(tmp_path)},
    )

    await runtime.start()

    assert captured["backend"].cwd == tmp_path.resolve()


def test_runtime_default_backend_scrubs_credentials_from_shell_env(tmp_path: Path) -> None:
    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
        env={
            "DEEPAGENTS_TALON_WORKSPACE": str(tmp_path),
            "LANGSMITH_API_KEY": "langsmith-key",
            "LANGSMITH_TENANT_ID": "tenant",
            "LANGSMITH_ORGANIZATION_ID": "org",
            "LANGSMITH_USER_ID": "user",
            "LANGCHAIN_API_KEY": "legacy-langsmith-key",
            "OPENAI_API_KEY": "openai-key",
            "ANTHROPIC_API_KEY": "anthropic-key",
            "FLEET_OAUTH_ACCESS_TOKEN": "oauth-token",
            "MCP_BEARER_TOKEN": "bearer-token",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
            "AWS_SESSION_TOKEN": "aws-session",
        },
    )
    backend = cast("LocalShellBackend", runtime.backend)

    result = backend.execute(
        "printf '<%s><%s><%s><%s><%s><%s><%s><%s><%s><%s><%s><%s>' "
        '"$LANGSMITH_API_KEY" '
        '"$LANGSMITH_TENANT_ID" '
        '"$LANGSMITH_ORGANIZATION_ID" '
        '"$LANGSMITH_USER_ID" '
        '"$LANGCHAIN_API_KEY" '
        '"$OPENAI_API_KEY" '
        '"$ANTHROPIC_API_KEY" '
        '"$FLEET_OAUTH_ACCESS_TOKEN" '
        '"$MCP_BEARER_TOKEN" '
        '"$AWS_SECRET_ACCESS_KEY" '
        '"$AWS_SESSION_TOKEN" '
        '"$DEEPAGENTS_TALON_WORKSPACE"'
    )

    assert result.exit_code == 0
    assert result.output == "<><><><><><><><><><><><>"


def test_runtime_default_backend_hardens_shell_env(tmp_path: Path) -> None:
    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
        env={
            "DEEPAGENTS_TALON_WORKSPACE": str(tmp_path),
            "PATH": str(tmp_path / "evil-bin"),
            "LD_PRELOAD": str(tmp_path / "libevil.so"),
            "PYTHONPATH": str(tmp_path / "evil-python"),
            "HOME": str(tmp_path / "home"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C",
        },
    )
    backend = cast("LocalShellBackend", runtime.backend)

    result = backend.execute(
        'printf "%s\\n%s\\n%s\\n%s\\n%s\\n%s" '
        '"$PATH" "$LD_PRELOAD" "$PYTHONPATH" "$HOME" "$LANG" "$LC_ALL"'
    )

    assert result.exit_code == 0
    assert result.output.splitlines() == [
        _SAFE_BACKEND_PATH,
        "",
        "",
        str(tmp_path / "home"),
        "C.UTF-8",
        "C",
    ]


async def test_runtime_passes_openai_base_url_to_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    model = object()

    def fake_init_chat_model(*args: Any, **kwargs: Any) -> object:
        captured["init_args"] = args
        captured["init_kwargs"] = kwargs
        return model

    def fake_create_deep_agent(**kwargs: Any) -> RecordingGraph:
        captured.update(kwargs)
        return RecordingGraph()

    monkeypatch.setattr("deepagents_talon.runtime.init_chat_model", fake_init_chat_model)
    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", fake_create_deep_agent)

    runtime = DeepAgentRuntime(
        model="openai:gpt-5.2",
        include_web_tools=False,
        skills=(),
        memory=(),
        env={"OPENAI_BASE_URL": "https://openai-compatible.example.com/v1"},
    )

    await runtime.start()

    assert captured["init_args"] == ("openai:gpt-5.2",)
    assert captured["init_kwargs"] == {
        "base_url": "https://openai-compatible.example.com/v1",
        "use_responses_api": True,
    }
    assert captured["model"] is model


async def test_runtime_leaves_non_openai_model_string_with_openai_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_create_deep_agent(**kwargs: Any) -> RecordingGraph:
        captured.update(kwargs)
        return RecordingGraph()

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", fake_create_deep_agent)

    runtime = DeepAgentRuntime(
        model="anthropic:claude-sonnet-4-6",
        include_web_tools=False,
        skills=(),
        memory=(),
        env={"OPENAI_BASE_URL": "https://openai-compatible.example.com/v1"},
    )

    await runtime.start()

    assert captured["model"] == "anthropic:claude-sonnet-4-6"


async def test_runtime_applies_configured_context_size_and_adds_compact_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    model = SimpleNamespace(profile={"max_input_tokens": 200_000, "tool_calling": True})
    compact = PassthroughMiddleware()

    def fake_init_chat_model(*args: Any, **kwargs: Any) -> object:
        captured["init_args"] = args
        captured["init_kwargs"] = kwargs
        return model

    def fake_create_summarization_tool_middleware(
        compact_model: object,
        backend: object,
    ) -> PassthroughMiddleware:
        captured["compact_model"] = compact_model
        captured["compact_backend"] = backend
        return compact

    def fake_create_deep_agent(**kwargs: Any) -> RecordingGraph:
        captured.update(kwargs)
        return RecordingGraph()

    monkeypatch.setattr("deepagents_talon.runtime.init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(
        "deepagents_talon.runtime.create_summarization_tool_middleware",
        fake_create_summarization_tool_middleware,
    )
    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", fake_create_deep_agent)

    runtime = DeepAgentRuntime(
        model="anthropic:claude-sonnet-4-6",
        include_web_tools=False,
        skills=(),
        memory=(),
        env={"DEEPAGENTS_TALON_CONTEXT_SIZE": "75000"},
    )

    await runtime.start()

    assert captured["init_args"] == ("anthropic:claude-sonnet-4-6",)
    assert captured["init_kwargs"] == {}
    assert model.profile == {"max_input_tokens": 75_000, "tool_calling": True}
    assert captured["model"] is model
    assert captured["compact_model"] is model
    assert captured["compact_backend"] is runtime.backend
    assert captured["middleware"] == [compact]


async def test_runtime_does_not_duplicate_existing_compact_tool_middleware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    model = SimpleNamespace(profile={"max_input_tokens": 200_000})
    existing = PassthroughMiddleware()

    def fake_init_chat_model(*_args: Any, **_kwargs: Any) -> object:
        return model

    def fail_create_summarization_tool_middleware(
        _model: object,
        _backend: object,
    ) -> PassthroughMiddleware:
        msg = "compact middleware should not be created when one already exists"
        raise AssertionError(msg)

    def fake_create_deep_agent(**kwargs: Any) -> RecordingGraph:
        captured.update(kwargs)
        return RecordingGraph()

    monkeypatch.setattr(
        "deepagents_talon.runtime.SummarizationToolMiddleware",
        PassthroughMiddleware,
    )
    monkeypatch.setattr("deepagents_talon.runtime.init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(
        "deepagents_talon.runtime.create_summarization_tool_middleware",
        fail_create_summarization_tool_middleware,
    )
    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", fake_create_deep_agent)

    runtime = DeepAgentRuntime(
        model="anthropic:claude-sonnet-4-6",
        include_web_tools=False,
        skills=(),
        memory=(),
        middleware=(existing,),
        env={"DEEPAGENTS_TALON_CONTEXT_SIZE": "75000"},
    )

    await runtime.start()

    assert model.profile == {"max_input_tokens": 75_000}
    assert captured["middleware"] == [existing]


async def test_runtime_rejects_invalid_context_size() -> None:
    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
        env={"DEEPAGENTS_TALON_CONTEXT_SIZE": "0"},
    )

    with pytest.raises(ValueError, match="DEEPAGENTS_TALON_CONTEXT_SIZE"):
        await runtime.start()


async def test_runtime_recursion_limit_defaults_when_env_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RecordingGraph()
    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", lambda **_kwargs: graph)
    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
    )
    await runtime.start()

    result = await runtime.invoke(AgentRequest(conversation_id="chat", text="hi"))
    assert graph.calls[0][1]["recursion_limit"] == 500
    assert result.text == "seen:1"


async def test_runtime_recursion_limit_reads_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    graph = RecordingGraph()
    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", lambda **_kwargs: graph)
    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
        env={"DEEPAGENTS_TALON_RECURSION_LIMIT": "1000"},
    )
    await runtime.start()

    await runtime.invoke(AgentRequest(conversation_id="chat", text="hi"))
    assert graph.calls[0][1]["recursion_limit"] == 1000


async def test_runtime_recursion_limit_env_overrides_explicit_arg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RecordingGraph()
    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", lambda **_kwargs: graph)
    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
        recursion_limit=200,
        env={"DEEPAGENTS_TALON_RECURSION_LIMIT": "750"},
    )
    await runtime.start()

    await runtime.invoke(AgentRequest(conversation_id="chat", text="hi"))
    assert graph.calls[0][1]["recursion_limit"] == 750


async def test_runtime_rejects_invalid_recursion_limit_env() -> None:
    with pytest.raises(ValueError, match="DEEPAGENTS_TALON_RECURSION_LIMIT"):
        DeepAgentRuntime(
            model="test:model",
            include_web_tools=False,
            skills=(),
            memory=(),
            env={"DEEPAGENTS_TALON_RECURSION_LIMIT": "0"},
        )


async def test_runtime_preserves_conversation_thread_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph = RecordingGraph()
    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", lambda **_kwargs: graph)
    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
    )
    await runtime.start()

    first = await runtime.invoke(AgentRequest(conversation_id="chat", text="first"))
    second = await runtime.invoke(AgentRequest(conversation_id="chat", text="second"))

    assert first.text == "seen:1"
    assert second.text == "seen:3"
    assert [call[1]["configurable"]["thread_id"] for call in graph.calls] == ["chat", "chat"]


async def test_runtime_reloads_components_and_retries_once_after_auth_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expired = AuthFailingGraph(StatusError(401, "expired OAuth token"))
    refreshed = RecordingGraph()
    graphs: list[object] = [expired, refreshed]

    def fake_create_deep_agent(**_kwargs: Any) -> object:
        return graphs.pop(0)

    reloads = 0

    async def reload_agent_components() -> RuntimeAgentComponents:
        nonlocal reloads
        reloads += 1
        return RuntimeAgentComponents(
            model="test:model",
            tools=(custom_tool,),
            skills=(),
            middleware=(),
        )

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", fake_create_deep_agent)
    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
        reload_agent_components=reload_agent_components,
    )
    await runtime.start()

    result = await runtime.invoke(AgentRequest(conversation_id="chat", text="run tool"))

    assert result.text == "seen:1"
    assert reloads == 1
    assert len(expired.calls) == 1
    assert len(refreshed.calls) == 1
    assert refreshed.calls[0][1]["configurable"]["thread_id"] == "chat"


async def test_runtime_returns_sanitized_structured_error_after_persistent_auth_failure(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expired = AuthFailingGraph(StatusError(401, "Bearer old-token expired"))
    denied = AuthFailingGraph(StatusError(403, "Bearer new-token denied"))
    graphs: list[object] = [expired, denied]

    def fake_create_deep_agent(**_kwargs: Any) -> object:
        return graphs.pop(0)

    reloads = 0

    async def reload_agent_components() -> RuntimeAgentComponents:
        nonlocal reloads
        reloads += 1
        return RuntimeAgentComponents(
            model="test:model",
            tools=(custom_tool,),
            skills=(),
            middleware=(),
        )

    caplog.set_level("INFO", logger="deepagents_talon.runtime")
    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", fake_create_deep_agent)
    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
        reload_agent_components=reload_agent_components,
    )
    await runtime.start()

    result = await runtime.invoke(AgentRequest(conversation_id="chat", text="run tool"))

    error = json.loads(result.text)
    assert error == {
        "error": "mcp_auth_failed",
        "message": (
            "Fleet MCP tool authorization failed after refreshing OAuth credentials. "
            "Run the Fleet pre-authorization step, then retry."
        ),
        "status_code": 403,
    }
    assert reloads == 1
    assert len(expired.calls) == 1
    assert len(denied.calls) == 1
    assert "old-token" not in caplog.text
    assert "new-token" not in caplog.text
    assert "old-token" not in result.text
    assert "new-token" not in result.text


async def test_runtime_reraises_provider_auth_without_fleet_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_error = StatusError(401, "OpenAI API key is invalid")
    graph = AuthFailingGraph(provider_error)

    def fake_create_deep_agent(**_kwargs: Any) -> object:
        return graph

    reloads = 0

    async def reload_agent_components() -> RuntimeAgentComponents:
        nonlocal reloads
        reloads += 1
        return RuntimeAgentComponents(model="test:model")

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", fake_create_deep_agent)
    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
        reload_agent_components=reload_agent_components,
    )
    await runtime.start()

    with pytest.raises(StatusError, match="OpenAI API key"):
        await runtime.invoke(AgentRequest(conversation_id="chat", text="run tool"))

    assert reloads == 0
    assert len(graph.calls) == 1


async def test_runtime_reraises_provider_auth_after_fleet_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expired = AuthFailingGraph(StatusError(401, "expired OAuth token"))
    provider_denied = AuthFailingGraph(StatusError(401, "Anthropic API key is invalid"))
    graphs: list[object] = [expired, provider_denied]

    def fake_create_deep_agent(**_kwargs: Any) -> object:
        return graphs.pop(0)

    reloads = 0

    async def reload_agent_components() -> RuntimeAgentComponents:
        nonlocal reloads
        reloads += 1
        return RuntimeAgentComponents(
            model="test:model",
            tools=(custom_tool,),
            skills=(),
            middleware=(),
        )

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", fake_create_deep_agent)
    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
        reload_agent_components=reload_agent_components,
    )
    await runtime.start()

    with pytest.raises(StatusError, match="Anthropic API key"):
        await runtime.invoke(AgentRequest(conversation_id="chat", text="run tool"))

    assert reloads == 1
    assert len(expired.calls) == 1
    assert len(provider_denied.calls) == 1


async def test_cron_tools_use_current_request_origin(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    store = CronJobStore(assistant_id="test", cron_dir=tmp_path / "cron")

    def fake_create_deep_agent(**kwargs: Any) -> CronCallingGraph:
        captured.update(kwargs)
        tools = cast("list[object]", kwargs["tools"])
        create_job = cast(
            "InvokableTool",
            next(tool for tool in tools if _tool_name(tool) == "create_job"),
        )
        return CronCallingGraph(create_job)

    monkeypatch.setattr("deepagents_talon.runtime.create_deep_agent", fake_create_deep_agent)
    runtime = DeepAgentRuntime(
        model="test:model",
        cron_store=store,
        include_web_tools=False,
        skills=(),
        memory=(),
    )
    await runtime.start()

    result = await runtime.invoke(
        AgentRequest(
            conversation_id="chat",
            text="schedule it",
            metadata={"channel": "whatsapp", "message_id": "msg-1"},
        ),
    )

    job = store.list_jobs()[0]
    assert result.text == job.id
    assert job.origin.conversation_id == "chat"
    assert job.origin.channel == "whatsapp"
    assert job.origin.message_id == "msg-1"
    assert any(_tool_name(tool) == "create_job" for tool in captured["tools"])


async def test_runtime_approves_tool_interrupt_with_channel_handler() -> None:
    graph = InterruptingGraph()
    approvals: list[ToolApprovalRequest] = []
    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
    )
    runtime._graph = graph

    async def approve(request: ToolApprovalRequest) -> ToolApprovalDecision:
        approvals.append(request)
        return "approve"

    result = await runtime.invoke(
        AgentRequest(
            conversation_id="chat",
            text="run",
            approval_handler=approve,
        )
    )

    assert result.text == "approved"
    assert graph.executed is True
    assert approvals[0].conversation_id == "chat"
    assert approvals[0].interrupt_id == "interrupt-1"
    assert approvals[0].action_requests[0]["name"] == "dangerous_tool"


async def test_runtime_logs_tool_approval_without_argument_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    graph = InterruptingGraph()
    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
    )
    runtime._graph = graph

    async def approve(_request: ToolApprovalRequest) -> ToolApprovalDecision:
        return "approve"

    caplog.set_level("INFO", logger="deepagents_talon.runtime")

    result = await runtime.invoke(
        AgentRequest(
            conversation_id="chat",
            text="run",
            approval_handler=approve,
        )
    )

    events = _talon_events(caplog)
    interrupt = next(event for event in events if event["event"] == "tool_approval.interrupt")
    resolved = next(event for event in events if event["event"] == "tool_approval.resolved")
    assert result.text == "approved"
    assert interrupt["action_names"] == ["dangerous_tool"]
    assert interrupt["action_count"] == 1
    assert interrupt["conversation_ref"] != "chat"
    assert resolved["decision"] == "approved"
    assert resolved["resolution"] == "operator"
    assert "/secret" not in caplog.text
    assert "chat" not in caplog.text


async def test_runtime_rejects_tool_interrupt_without_running_tool(
    caplog: pytest.LogCaptureFixture,
) -> None:
    graph = InterruptingGraph()
    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
    )
    runtime._graph = graph

    async def reject(_request: ToolApprovalRequest) -> ToolApprovalDecision:
        return "reject"

    caplog.set_level("INFO", logger="deepagents_talon.runtime")

    result = await runtime.invoke(
        AgentRequest(
            conversation_id="chat",
            text="run",
            approval_handler=reject,
        )
    )

    resume = cast(
        "dict[str, dict[str, list[dict[str, str]]]]",
        cast("Command", graph.calls[1][0]).resume,
    )
    assert result.text == "denied"
    assert graph.executed is False
    assert resume["interrupt-1"]["decisions"] == [
        {"type": "reject", "message": "Denied by operator."}
    ]
    resolved = next(
        event for event in _talon_events(caplog) if event["event"] == "tool_approval.resolved"
    )
    assert resolved["decision"] == "denied"
    assert resolved["resolution"] == "operator"


async def test_runtime_auto_rejects_cron_tool_interrupt() -> None:
    graph = InterruptingGraph()
    runtime = DeepAgentRuntime(
        model="test:model",
        include_web_tools=False,
        skills=(),
        memory=(),
    )
    runtime._graph = graph

    result = await runtime.invoke(
        AgentRequest(
            conversation_id="chat",
            text="run",
            metadata={"trigger": "cron"},
        )
    )

    resume = cast(
        "dict[str, dict[str, list[dict[str, str]]]]",
        cast("Command", graph.calls[1][0]).resume,
    )
    auto_reject_message = (
        "Tool approval is unavailable for scheduled runs; skipped the gated tool call."
    )
    assert result.text == "denied"
    assert graph.executed is False
    assert resume["interrupt-1"]["decisions"] == [
        {
            "type": "reject",
            "message": auto_reject_message,
        }
    ]


def test_is_retryable_matches_known_transient_errors() -> None:
    errors = [
        StatusError(408),
        StatusError(429),
        StatusError(503),
        StatusError(400, "maximum context length exceeded"),
        RuntimeError("failed to parse model response"),
        RuntimeError("invalid tool_call payload"),
        ConnectionError("connection reset by peer"),
        TimeoutError("operation timed out"),
        RuntimeError("service temporarily unavailable"),
    ]

    for error in errors:
        assert _is_retryable(error)


def test_is_retryable_rejects_unrelated_context_and_client_errors() -> None:
    errors = [
        StatusError(400, "invalid request: unknown field"),
        StatusError(404, "not found"),
        RuntimeError("invalid context manager"),
        RuntimeError("missing context variable"),
        RuntimeError("invalid connection setting"),
        ValueError("timeout must be positive"),
    ]

    for error in errors:
        assert not _is_retryable(error)


def _tool_name(tool: object) -> str:
    name = getattr(tool, "name", None)
    if isinstance(name, str):
        return name
    function_name = getattr(tool, "__name__", None)
    if isinstance(function_name, str):
        return function_name
    msg = f"tool has no name: {tool!r}"
    raise AssertionError(msg)


def _talon_events(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    return [
        json.loads(message.removeprefix("talon_event "))
        for message in caplog.messages
        if message.startswith("talon_event ")
    ]
