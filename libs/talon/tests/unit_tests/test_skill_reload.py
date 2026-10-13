from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.runtime import DeepAgentRuntime

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from langchain_core.outputs import ChatResult


class PromptRecorder(FakeMessagesListChatModel):
    """Replies with fixed text and records each request's system prompt."""

    prompts: list[str] = []  # noqa: RUF012  # pydantic field, copied per instance

    def bind_tools(self, _tools: object, **_kwargs: object) -> PromptRecorder:
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.prompts.append(str(messages[0].content))
        return super()._generate(messages, *args, **kwargs)


def _write_skill(home: Path, name: str, description: str) -> None:
    skill = home / "skills" / name / "SKILL.md"
    skill.parent.mkdir(parents=True, exist_ok=True)
    skill.write_text(f"---\nname: {name}\ndescription: {description}\n---\nFollow the steps.")


def _runtime(home: Path, saver: InMemorySaver) -> DeepAgentRuntime:
    return DeepAgentRuntime(
        model="test:model",
        assistant_dir=home,
        checkpointer=saver,
        include_web_tools=False,
        memory=(),
    )


def _patch_model(monkeypatch: pytest.MonkeyPatch, model: PromptRecorder) -> None:
    monkeypatch.setattr(
        "deepagents_talon.runtime._resolve_model_from_env", lambda *_args, **_kwargs: model
    )


async def _turn(runtime: DeepAgentRuntime, text: str) -> None:
    await runtime.invoke(AgentRequest("chat", text))


async def test_new_skill_reaches_an_existing_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = PromptRecorder(responses=[AIMessage(content="one"), AIMessage(content="two")])
    _patch_model(monkeypatch, model)
    runtime = _runtime(tmp_path, InMemorySaver())
    await runtime.start()
    try:
        await _turn(runtime, "first")
        _write_skill(tmp_path, "release-notes", "Draft release notes")
        await _turn(runtime, "second")
    finally:
        await runtime.stop()

    assert "release-notes" not in model.prompts[0]
    assert "release-notes" in model.prompts[1]


async def test_restart_reloads_skills_for_a_checkpointed_conversation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    saver = InMemorySaver()
    model = PromptRecorder(responses=[AIMessage(content="one"), AIMessage(content="two")])
    _patch_model(monkeypatch, model)
    _write_skill(tmp_path, "release-notes", "Draft release notes")
    before = _runtime(tmp_path, saver)
    await before.start()
    try:
        await _turn(before, "first")
    finally:
        await before.stop()

    _write_skill(tmp_path, "release-notes", "Summarize merged pull requests")
    after = _runtime(tmp_path, saver)
    await after.start()
    try:
        await _turn(after, "second")
    finally:
        await after.stop()

    assert "Draft release notes" in model.prompts[0]
    assert "Summarize merged pull requests" in model.prompts[1]
