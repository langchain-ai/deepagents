from __future__ import annotations

import base64
import hashlib
from typing import TYPE_CHECKING, Any

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langgraph.checkpoint.memory import InMemorySaver

from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.media import build_model_content
from deepagents_talon.runtime import DeepAgentRuntime

if TYPE_CHECKING:
    from pathlib import Path

    import pytest
    from langchain_core.outputs import ChatResult

_PIXEL = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)


class RequestRecorder(FakeMessagesListChatModel):
    """Replies with fixed text and records each request's messages."""

    requests: list[list[BaseMessage]] = []  # noqa: RUF012  # pydantic field, copied per instance

    def bind_tools(self, _tools: object, **_kwargs: object) -> RequestRecorder:
        return self

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        self.requests.append(list(messages))
        return super()._generate(messages, *args, **kwargs)


def _inbound_image(tmp_path: Path) -> str | list[dict[str, object]]:
    image = tmp_path / "inbound.png"
    image.write_bytes(_PIXEL)
    return build_model_content(
        "what is this?",
        {"media_type": "image", "media_paths": [str(image)], "media_mime_types": ["image/png"]},
    )


async def test_inbound_image_is_stored_once_as_a_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = RequestRecorder(responses=[AIMessage(content="a pixel")])
    monkeypatch.setattr(
        "deepagents_talon.runtime._resolve_model_from_env", lambda *_args, **_kwargs: model
    )
    home = tmp_path / "home"
    runtime = DeepAgentRuntime(
        model="test:model",
        assistant_dir=home,
        checkpointer=InMemorySaver(),
        include_web_tools=False,
        skills=(),
        memory=(),
    )
    content = _inbound_image(tmp_path)
    await runtime.start()
    try:
        await runtime.invoke(
            AgentRequest("chat", "what is this?", metadata={"model_content": content})
        )
        state = await runtime._graph.aget_state({"configurable": {"thread_id": "chat"}})
    finally:
        await runtime.stop()

    encoded = base64.b64encode(_PIXEL).decode("ascii")
    assert encoded not in str(state.values["messages"])
    assert (
        home / "artifacts" / "blobs" / hashlib.sha256(_PIXEL).hexdigest()
    ).read_bytes() == _PIXEL
    assert encoded in str(model.requests[0])
