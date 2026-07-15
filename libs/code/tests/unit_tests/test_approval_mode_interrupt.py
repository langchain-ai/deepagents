"""End-to-end regression tests for live approval-mode interrupt suppression.

These reproduce the original bug: with `auto_approve=true` and a configured
`approval_mode_key`, gated tools still produced LangGraph `GraphInterrupt`
approval checkpoints on the LangGraph API server path, because the runtime
Store there is an `AsyncBatchedBaseStore` whose synchronous `get()` raises on
the running event loop. The synchronous `when` predicate caught that, failed
closed, and interrupted despite the live store saying auto-approve.

The tests run a real agent graph against an `AsyncBatchedBaseStore` — the store
shape used by the server — so a regression to a synchronous store read in the
interrupt path resurfaces as an unexpected interrupt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base.batch import AsyncBatchedBaseStore
from langgraph.store.memory import InMemoryStore

from deepagents_code._cli_context import CLIContextSchema

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from langchain.agents.middleware import InterruptOnConfig
    from langchain.agents.middleware.types import AgentMiddleware
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable
    from langchain_core.tools import BaseTool
    from langgraph.pregel import Pregel
    from langgraph.store.base import Op, Result


class _ToolCallingFakeModel(GenericFakeChatModel):
    """Fake chat model that tolerates `bind_tools` for agent graphs."""

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],  # noqa: ARG002
        *,
        tool_choice: str | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self


class _LoopBoundStore(AsyncBatchedBaseStore):
    """`AsyncBatchedBaseStore` backed by an `InMemoryStore`.

    Mirrors the server runtime store: created on the running event loop, so its
    synchronous `get()` raises `asyncio.InvalidStateError` when called there.
    """

    def __init__(self) -> None:
        super().__init__()
        self._base = InMemoryStore()

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await self._base.abatch(ops)

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        return self._base.batch(ops)


@tool
def touch_network(url: str) -> str:
    """A gated tool that would touch the network."""
    return f"fetched {url}"


def _make_agent(
    model: _ToolCallingFakeModel, store: AsyncBatchedBaseStore
) -> Pregel[Any, Any, Any, Any]:
    from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT, SubAgent

    from deepagents_code.agent import (
        ApprovalModePrefetchMiddleware,
        _should_interrupt_tool_call,
    )

    interrupt_on: dict[str, bool | InterruptOnConfig] = {
        "touch_network": {
            "allowed_decisions": ["approve", "reject"],
            "when": _should_interrupt_tool_call,
        }
    }
    prefetch_main = cast(
        "list[AgentMiddleware[Any, Any]]", [ApprovalModePrefetchMiddleware()]
    )
    general_purpose: SubAgent = {
        "name": GENERAL_PURPOSE_SUBAGENT["name"],
        "description": GENERAL_PURPOSE_SUBAGENT["description"],
        "system_prompt": GENERAL_PURPOSE_SUBAGENT["system_prompt"],
        "middleware": cast(
            "list[AgentMiddleware[Any, Any]]", [ApprovalModePrefetchMiddleware()]
        ),
    }
    return create_deep_agent(
        model=model,
        tools=[touch_network],
        middleware=prefetch_main,
        interrupt_on=interrupt_on,
        context_schema=CLIContextSchema,
        subagents=[general_purpose],
        checkpointer=InMemorySaver(),
        store=store,
    )


async def _seed_mode(
    store: AsyncBatchedBaseStore, thread_id: str, *, auto: bool
) -> str:
    from deepagents_code.approval_mode import (
        APPROVAL_MODE_NAMESPACE,
        approval_mode_key,
        approval_mode_payload,
    )

    key = approval_mode_key(thread_id)
    await store.aput(
        APPROVAL_MODE_NAMESPACE, key, dict(approval_mode_payload(auto_approve=auto))
    )
    return key


@pytest.mark.parametrize(
    ("auto", "expect_interrupt"),
    [(True, False), (False, True)],
)
async def test_top_level_gated_tool_honors_live_store(
    auto: bool, expect_interrupt: bool
) -> None:
    """A top-level gated tool honors the live store on the async-batched path."""
    store = _LoopBoundStore()
    thread_id = "thread-top"
    key = await _seed_mode(store, thread_id, auto=auto)

    model = _ToolCallingFakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "touch_network",
                            "args": {"url": "https://example.com"},
                            "id": "c1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="done"),
            ]
        )
    )
    agent = _make_agent(model, store)
    ctx = CLIContextSchema(
        auto_approve=auto, approval_mode_key=key, thread_id=thread_id
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "go"}]},
        {"configurable": {"thread_id": thread_id}},
        context=ctx,
    )
    assert bool(result.get("__interrupt__")) is expect_interrupt


@pytest.mark.parametrize(
    ("auto", "expect_interrupt"),
    [(True, False), (False, True)],
)
async def test_subagent_gated_tool_honors_live_store(
    auto: bool, expect_interrupt: bool
) -> None:
    """A gated tool inside a general-purpose subagent honors the live store."""
    store = _LoopBoundStore()
    thread_id = "thread-sub"
    key = await _seed_mode(store, thread_id, auto=auto)

    model = _ToolCallingFakeModel(
        messages=iter(
            [
                # Main agent delegates to the general-purpose subagent.
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "task",
                            "args": {
                                "description": "fetch a page",
                                "subagent_type": "general-purpose",
                            },
                            "id": "t1",
                            "type": "tool_call",
                        }
                    ],
                ),
                # Subagent turn: call the gated tool.
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "touch_network",
                            "args": {"url": "https://example.com"},
                            "id": "c1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="subagent done"),
                AIMessage(content="all done"),
            ]
        )
    )
    agent = _make_agent(model, store)
    ctx = CLIContextSchema(
        auto_approve=auto, approval_mode_key=key, thread_id=thread_id
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "go"}]},
        {"configurable": {"thread_id": thread_id}},
        context=ctx,
    )
    assert bool(result.get("__interrupt__")) is expect_interrupt
