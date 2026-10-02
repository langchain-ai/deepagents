"""Read-only context audits and channel command dispatch."""

from __future__ import annotations

import asyncio

import pytest
from deepagents.backends import FilesystemBackend
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool

from deepagents_code._fake_models import _ToolBindingFakeModel
from deepagents_talon.context_doctor import _usage
from deepagents_talon.host import TalonHost
from deepagents_talon.runtime import DeepAgentRuntime
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config, _wait_for_request


@pytest.fixture
async def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "deepagents_talon.runtime._resolve_model_from_env",
        lambda *_args, **_kwargs: _ToolBindingFakeModel(),
    )
    assistant = tmp_path / "assistant"
    assistant.mkdir()
    (assistant / "AGENTS.md").write_text("private system instructions")
    memory = assistant / "memory.md"
    memory.write_text("private memory content")
    skill = assistant / "skills" / "research" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("---\nname: research\ndescription: Find evidence.\n---\nprivate skill body")
    agent = DeepAgentRuntime(
        model="test",
        assistant_dir=assistant,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=False),
        memory=[str(memory)],
        include_web_tools=False,
        env={},
    )
    await agent.start()
    try:
        yield agent
    finally:
        await agent.stop()


async def test_fresh_report_reads_configured_sources_without_writing_state(runtime, tmp_path):
    report = await runtime.context_doctor("chat")

    assert "System prompt (configured)" in report
    assert "AGENTS.md memory (1 files)" in report
    assert "Skills index (1 loaded)" in report
    assert "research" in report
    assert "All (including MCP) tool schemas" in report
    assert "TOTAL injected before conversation" in report
    assert "private" not in report
    assert str(tmp_path) not in report
    assert not (await runtime._graph.aget_state({"configurable": {"thread_id": "chat"}})).values


async def test_report_uses_current_thread_checkpoint_and_cached_memory(runtime):
    config = {"configurable": {"thread_id": "chat"}}
    await runtime._graph.aupdate_state(
        config,
        {
            "messages": [
                HumanMessage(content="private question"),
                AIMessage(
                    content="private answer",
                    usage_metadata={"input_tokens": 1234, "output_tokens": 3, "total_tokens": 1237},
                ),
            ],
            "memory_contents": {"cached": "cached private memory"},
            "skills_metadata": [],
        },
    )
    before = await runtime._graph.aget_state(config)
    report = await runtime.context_doctor("chat")
    after = await runtime._graph.aget_state(config)

    assert "1,234" in report
    assert "Skills index (0 loaded)" in report
    assert "private" not in report
    assert before.config == after.config
    assert before.values == after.values
    assert "1,234" not in await runtime.context_doctor("other-chat")


async def test_report_reflects_reloaded_tools(runtime):
    @tool
    def added_tool() -> str:
        """A tool with a large schema description."""
        raise AssertionError

    added_tool.description = "Large tool schema. " * 1000
    before = await runtime.context_doctor("chat")

    async def reload_tools():
        return [added_tool]

    runtime.reload_tools = reload_tools
    await runtime.reload_mcp_configuration()
    after = await runtime.context_doctor("chat")
    assert before != after
    assert "Large tool schema" not in after


def test_usage_counts_effective_conversation_after_compaction():
    old = HumanMessage(content="old history " * 1000)
    recent = HumanMessage(content="recent")
    summary = HumanMessage(content="summary")
    tokens, provider = _usage(
        {
            "messages": [old, recent],
            "_summarization_event": {"cutoff_index": 1, "summary_message": summary},
        }
    )
    assert tokens == count_tokens_approximately([summary, recent])
    assert provider is None


class DiagnosticAgent(BlockingAgent):
    async def context_doctor(self, conversation_id: str) -> str:
        return f"audit:{conversation_id}"


async def test_command_does_not_interrupt_active_work_and_follows_new_thread(tmp_path):
    agent = DiagnosticAgent()
    channel = RecordingChannel()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await channel.receive("block")
        await _wait_for_request(agent, "block")
        active = host._tasks["test:chat"]
        await channel.receive("/context-doctor")
        assert channel.sent[-1] == ("chat", "audit:test:chat")
        assert not active.done()
        assert agent.recoveries == []
        assert len(agent.requests) == 1

        await channel.receive("/new")
        await channel.receive("/context-doctor")
        assert channel.sent[-1][1].startswith("audit:test:chat:talon-reset:")
    finally:
        await host.stop()


@pytest.mark.parametrize("failure", [False, True])
async def test_command_handles_unavailable_and_failed_diagnostics(tmp_path, failure):
    class FailingAgent(DiagnosticAgent):
        async def context_doctor(self, _conversation_id: str) -> str:
            msg = "private backend failure"
            raise RuntimeError(msg)

    agent = FailingAgent() if failure else BlockingAgent()
    channel = RecordingChannel()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await asyncio.wait_for(channel.receive("/context-doctor"), timeout=1)
        assert "private" not in channel.sent[-1][1]
        assert ("Could not" if failure else "unavailable") in channel.sent[-1][1]
        assert agent.requests == []
    finally:
        await host.stop()
