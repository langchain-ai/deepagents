from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage

from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.runtime import DeepAgentRuntime

if TYPE_CHECKING:
    from pathlib import Path

    from langchain_core.outputs import ChatResult

_BODY = "Count merged pull requests by label."


class RequestRecorder(FakeMessagesListChatModel):
    """Replies with fixed text and records each request's messages."""

    requests: list[list[BaseMessage]] = []  # noqa: RUF012  # pydantic field, copied per instance

    def bind_tools(self, _tools: object, **_kwargs: object) -> RequestRecorder:
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.requests.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


@pytest.mark.parametrize(("trigger", "pinned"), [("cron", True), (None, False)])
async def test_only_scheduled_turns_pin_skills(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, trigger: str | None, pinned: bool
) -> None:
    skill = tmp_path / "skills" / "weekly-report" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(f"---\nname: weekly-report\ndescription: Weekly report\n---\n{_BODY}")
    model = RequestRecorder(responses=[AIMessage(content="done")])
    monkeypatch.setattr(
        "deepagents_talon.runtime._resolve_model_from_env", lambda *_args, **_kwargs: model
    )
    runtime = DeepAgentRuntime(
        model="test:model", assistant_dir=tmp_path, include_web_tools=False, memory=()
    )
    metadata: dict[str, object] = {"pinned_skills": ["weekly-report"]}
    if trigger is not None:
        metadata["trigger"] = trigger
    await runtime.start()
    try:
        await runtime.invoke(AgentRequest("job:talon-cron", "run the report", metadata=metadata))
    finally:
        await runtime.stop()

    assert (_BODY in str(model.requests[0])) is pinned
