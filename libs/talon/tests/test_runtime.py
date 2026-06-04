from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol, cast

from deepagents.backends import LocalShellBackend

from deepagents_talon.cron import CronJobStore
from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.runtime import DeepAgentRuntime

if TYPE_CHECKING:
    import pytest


class InvokableTool(Protocol):
    def invoke(self, payload: dict[str, object]) -> dict[str, object]:
        """Invoke a tool with a structured payload."""


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


def custom_tool() -> str:
    """Custom runtime tool."""
    return "ok"


def fetch_url() -> str:
    """Fetch URL tool stub."""
    return "fetched"


def web_search() -> str:
    """Web search tool stub."""
    return "searched"


def http_request() -> str:
    """HTTP request tool stub."""
    return "requested"


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
    monkeypatch.setattr(
        "deepagents_talon.runtime.build_web_tools",
        lambda: [fetch_url, web_search, http_request],
    )

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

    tool_names = {_tool_name(tool) for tool in captured["tools"]}
    assert {
        "fetch_url",
        "web_search",
        "http_request",
        "create_job",
        "list_jobs",
        "edit_job",
        "remove_job",
        "custom_tool",
    } <= tool_names


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


def _tool_name(tool: object) -> str:
    name = getattr(tool, "name", None)
    if isinstance(name, str):
        return name
    function_name = getattr(tool, "__name__", None)
    if isinstance(function_name, str):
        return function_name
    msg = f"tool has no name: {tool!r}"
    raise AssertionError(msg)
