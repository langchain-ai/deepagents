"""Shared test helpers for QuickJS test suites."""

from __future__ import annotations

from collections.abc import (
    Iterator,  # noqa: TC003 — pydantic resolves field annotations at runtime
    Sequence,  # noqa: TC003 — pydantic resolves field annotations at runtime
)
from typing import TYPE_CHECKING, Any, Literal

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
)
from langgraph.types import Command
from pydantic import Field

if TYPE_CHECKING:
    from collections.abc import Callable


class FakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel whose `bind_tools` returns self.

    Without this override, `create_deep_agent`/`create_agent` wraps the model in
    a binding that no longer reads from the scripted message iterator.
    """

    messages: Iterator[AIMessage | str] = Field(exclude=True)

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> FakeChatModel:
        del tools
        return self


async def answer_pauses(
    agent: Any,
    answer: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    invoke_mode: Literal["invoke", "ainvoke"] = "ainvoke",
    durability: Literal["async", "exit"] = "async",
    max_pauses: int = 5,
) -> tuple[Any, list[list[dict[str, Any]]]]:
    """Run to completion, answering every pause; return the output and pauses."""
    config = {"configurable": {"thread_id": "t"}}

    async def step(payload: Any) -> Any:
        if invoke_mode == "ainvoke":
            return await agent.ainvoke(
                payload, config, version="v2", durability=durability
            )
        return agent.invoke(payload, config, version="v2", durability=durability)

    out = await step({"messages": [HumanMessage("go")]})
    pauses = []
    while out.interrupts and len(pauses) < max_pauses:
        pauses.append([i.value for i in out.interrupts])
        out = await step(
            Command(resume={i.id: answer(i.value) for i in out.interrupts})
        )
    return out, pauses
