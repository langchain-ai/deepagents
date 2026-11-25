"""Conformance: context usage, cost, and session titles after a turn."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from acp.schema import SessionInfoUpdate, UsageUpdate
from langchain_core.messages import AIMessage

from deepagents_acp.server import AgentServerACP, AgentSessionContext
from tests.conformance.harness import Harness, deep_agent, scripted_model

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class _MeteredServer(AgentServerACP):
    def context_window(self, context: AgentSessionContext) -> int | None:
        return 200_000

    def session_cost(self, values: Mapping[str, Any]) -> float | None:
        return 0.25

    async def session_title(self, context: AgentSessionContext, messages: Sequence[Any]) -> str:
        return "Fix the login bug"


def _answer() -> AIMessage:
    return AIMessage(
        content="Done",
        usage_metadata={"input_tokens": 1200, "output_tokens": 30, "total_tokens": 1230},
    )


async def test_turn_reports_usage_cost_and_a_title() -> None:
    harness = Harness.start(deep_agent(scripted_model(_answer())), server_class=_MeteredServer)
    session_id = await harness.new_session()

    await harness.prompt(session_id, "Fix the login bug please")
    await harness.settle()

    updates = harness.client.session_updates(session_id)
    [usage] = [update for update in updates if isinstance(update, UsageUpdate)]
    assert (usage.used, usage.size) == (1230, 200_000)
    assert usage.cost is not None
    assert (usage.cost.amount, usage.cost.currency) == (0.25, "USD")
    [info] = [update for update in updates if isinstance(update, SessionInfoUpdate)]
    assert info.title == "Fix the login bug"


async def test_no_usage_without_a_context_window_and_default_title() -> None:
    harness = Harness.start(deep_agent(scripted_model(_answer())))
    session_id = await harness.new_session()

    await harness.prompt(session_id, "Rename the config loader\nand more details")
    await harness.settle()

    updates = harness.client.session_updates(session_id)
    assert not any(isinstance(update, UsageUpdate) for update in updates)
    titles = [update.title for update in updates if isinstance(update, SessionInfoUpdate)]
    assert titles == ["Rename the config loader"]
