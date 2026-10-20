from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from deepagents_talon.config import TalonConfig, TalonConfigError
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import SendResult
from deepagents_talon.messaging import MESSAGE_HANDLER, progress_messages, send_message
from deepagents_talon.runtime import DeepAgentRuntime
from tests.conftest import RecordingChannel
from tests.test_host import _config
from tests.unit_tests.test_messaging import ToolCallingModel

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("value", [None, "1", "2", "3"])
def test_verbosity_configuration(tmp_path: Path, value: str | None) -> None:
    env = {} if value is None else {"DEEPAGENTS_TALON_VERBOSITY": value}
    config = TalonConfig.from_env(env, base_home=tmp_path)
    assert config.verbosity == (1 if value is None else int(value))
    runtime = DeepAgentRuntime(model="test:model", env=env)
    assert runtime.verbosity == config.verbosity
    names = {item.name for item in runtime._build_tools()}
    assert ("send_message" in names) is (config.verbosity > 1)
    prompt = runtime._resolve_system_prompt()
    assert prompt is not None
    assert {1: "only the final answer", 2: "not per subagent", 3: "brief, useful"}[
        config.verbosity
    ] in prompt


@pytest.mark.parametrize("level", [2, 3])
def test_progress_tool_guidance(level: int) -> None:
    runtime = DeepAgentRuntime(model="test:model", env={"DEEPAGENTS_TALON_VERBOSITY": str(level)})
    tools = {item.name: item for item in runtime._build_tools()}
    description = tools["send_message"].description
    assert "important updates" in description
    assert "changes in working direction" in description
    assert ("extremely sparingly" in description) is (level == 2)
    assert ("at most 1-2" in description) is (level == 2)
    prompt = runtime._resolve_system_prompt()
    assert prompt is not None
    assert ("extremely sparingly" in prompt) is (level == 2)
    assert ("no fixed message limit" in prompt) is (level == 3)


@pytest.mark.parametrize("value", ["", "0", "4", "-1", "2.0", "silent"])
def test_invalid_verbosity(tmp_path: Path, value: str) -> None:
    env = {"DEEPAGENTS_TALON_VERBOSITY": value}
    with pytest.raises(TalonConfigError, match="VERBOSITY must be 1, 2, or 3"):
        TalonConfig.from_env(env, base_home=tmp_path)
    with pytest.raises(TalonConfigError, match="VERBOSITY must be 1, 2, or 3"):
        DeepAgentRuntime(model="test:model", env=env)


@pytest.mark.parametrize(("level", "updates"), [(1, 0), (2, 2), (3, 6)])
async def test_multistep_turns_through_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    level: int,
    updates: int,
) -> None:
    caplog.set_level(logging.INFO, logger="deepagents_talon.runtime")
    effects: list[int] = []

    @tool
    def work(phase: int) -> str:
        """Complete one phase of a multi-step task."""
        effects.append(phase)
        return f"Phase {phase} complete"

    messages: list[AIMessage] = []
    for turn in range(2):
        for phase in range(6):
            explicit = level > 1 and phase % 2 == 1
            calls = [{"name": "work", "args": {"phase": phase}, "id": f"{turn}-{phase}"}]
            if explicit:
                calls.append(
                    {
                        "name": "send_message",
                        "args": {"text": f"Update {phase}"},
                        "id": f"update-{turn}-{phase}",
                    }
                )
            messages.append(AIMessage(content=f"Working on phase {phase}", tool_calls=calls))
        messages.append(AIMessage(content=f"Finished turn {turn}"))
    model = ToolCallingModel(messages=iter(messages))
    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_k: model)
    runtime = DeepAgentRuntime(
        model="test:model",
        tools=[work],
        assistant_dir=tmp_path,
        include_web_tools=False,
        skills=(),
        memory=(),
        env={
            "DEEPAGENTS_TALON_VERBOSITY": str(level),
            "DEEPAGENTS_TALON_AGENT_ACTIVITY_LOGGING": "true",
        },
    )
    snapshot = runtime.approval_store.ensure()
    runtime.approval_store.update({"work": False, "send_message": False}, snapshot.revision)
    channel = RecordingChannel()
    host = TalonHost(config=_config(tmp_path), agent=runtime, channels=[channel])
    await host.start()
    try:
        for turn in range(2):
            await channel.receive("Complete the multi-step task")
            await asyncio.gather(*host._tasks.values())
            assert channel.sent[-1] == ("chat", f"Finished turn {turn}")
            assert len(channel.sent) == (turn + 1) * (updates + 1)
        assert sorted(effects) == sorted(list(range(6)) * 2)
        activity = [
            json.loads(record.message.removeprefix("talon_event "))
            for record in caplog.records
            if record.message.startswith("talon_event ")
        ]
        assert (
            sum(
                event.get("event") == "agent.tool.completed" and event.get("tool_name") == "work"
                for event in activity
            )
            == 12
        )
        assert MESSAGE_HANDLER.get() is None
    finally:
        await host.stop()


async def test_sparse_budget_is_shared_by_parallel_calls_and_isolated_by_turn() -> None:
    barrier = asyncio.Barrier(2)

    async def run_turn() -> list[str]:
        sent: list[str] = []

        async def deliver(text: str) -> SendResult:
            await asyncio.sleep(0)
            sent.append(text)
            return SendResult(success=True)

        with progress_messages(deliver, 2):
            await barrier.wait()
            results = await asyncio.gather(
                *(send_message.ainvoke({"text": str(index)}) for index in range(5))
            )
            assert sum("Message sent." in result for result in results) == 2
            assert sum("limit reached" in result for result in results) == 3
        return sent

    assert await asyncio.gather(run_turn(), run_turn()) == [["0", "1"], ["0", "1"]]
    assert MESSAGE_HANDLER.get() is None


async def test_progress_scope_cleans_up_after_failure() -> None:
    msg = "interrupted"
    with pytest.raises(RuntimeError, match="interrupted"), progress_messages(None, 1):
        raise RuntimeError(msg)
    sent: list[str] = []

    async def deliver(text: str) -> SendResult:
        sent.append(text)
        return SendResult(success=True)

    token = MESSAGE_HANDLER.set(deliver)
    try:
        await send_message.ainvoke({"text": "new turn"})
        assert sent == ["new turn"]
    finally:
        MESSAGE_HANDLER.reset(token)
