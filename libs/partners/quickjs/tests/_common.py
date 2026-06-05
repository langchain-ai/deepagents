"""Shared test helpers for QuickJS test suites."""

from __future__ import annotations

from collections.abc import (
    Iterator,  # noqa: TC003 — pydantic resolves field annotations at runtime
    Sequence,  # noqa: TC003 — pydantic resolves field annotations at runtime
)
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,  # noqa: TC002 — pydantic resolves field annotations at runtime
)
from pydantic import Field


class FakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel whose ``bind_tools`` returns self.

    Without this override, ``create_deep_agent``/``create_agent`` wraps the model in
    a binding that no longer reads from the scripted message iterator.
    """

    messages: Iterator[AIMessage | str] = Field(exclude=True)

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> FakeChatModel:
        del tools
        return self
