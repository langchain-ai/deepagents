"""Accounting-only transport for JavaScript subagents."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import itertools
import json
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool
from langchain_quickjs import CodeInterpreterMiddleware
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointMetadata,
    empty_checkpoint,
)
from langgraph.config import get_config
from langgraph.types import Command

from deepagents_code.cost_tracking import _parent_checkpoint_scope

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.messages import ToolMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.prebuilt.tool_node import ToolCallRequest, ToolRuntime

_ACCOUNTING_KEY = "__deepagents_js_cost_owner"
_RESPONSE_FORMAT_KEY = "__deepagents_subagent_response_format"


class _ReceiptMetadata(CheckpointMetadata):
    js_cost_owner: str


def _digest(value: str) -> str:
    """Return a hashed identity without storing prompts in checkpoint names."""
    return hashlib.sha256(value.encode()).hexdigest()


class CostAwareCodeInterpreterMiddleware(CodeInterpreterMiddleware):
    """Transport cost receipts without caching JavaScript or forwarding state."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Return owned receipts after settling all dispatched children."""
        if request.tool not in self.tools:
            return await handler(request)
        configurable = request.runtime.config.get("configurable", {})
        scope = configurable.get("checkpoint_ns", "")
        owner = configurable.get(_ACCOUNTING_KEY)
        accounting = (
            owner or f"{scope}|js_cost:{_digest(request.runtime.tool_call_id or '')}"
        )
        active: set[asyncio.Task] = set()
        tools = [
            _cost_task(tool, request.runtime, accounting, active)
            if isinstance(tool, StructuredTool) and tool.name == "task"
            else tool
            for tool in request.runtime.tools
        ]
        runtime = dataclasses.replace(request.runtime, tools=tools)
        try:
            result = await handler(request.override(runtime=runtime))
        finally:
            for invocation in list(active):
                if not invocation.cancelling():
                    invocation.cancel()
            await asyncio.gather(*active, return_exceptions=True)
        if owner:
            return result
        total = await _receipt_total(runtime.config, accounting)
        if total <= 0:
            return result
        transfers = {
            accounting: {
                "owner_scope": _parent_checkpoint_scope(scope),
                "cost_usd": total,
            }
        }
        if isinstance(result, Command):
            update = dict(result.update) if isinstance(result.update, Mapping) else {}
            update["_session_cost_transfers"] = transfers
            return dataclasses.replace(result, update=update)
        return Command(
            update={"messages": [result], "_session_cost_transfers": transfers}
        )


def _cost_task(
    tool: StructuredTool,
    outer_runtime: ToolRuntime,
    owner: str,
    active: set[asyncio.Task],
) -> StructuredTool:
    """Return a task proxy with request-isolated child graph checkpoints."""
    occurrences: dict[str, int] = {}

    async def invoke(
        description: str, subagent_type: str, runtime: ToolRuntime
    ) -> object:
        configurable = dict(runtime.config.get("configurable", {}))
        response_format = configurable.get(_RESPONSE_FORMAT_KEY)
        fingerprint = _digest(
            json.dumps(
                [description, subagent_type, getattr(response_format, "schema", None)],
                sort_keys=True,
            )
        )
        occurrence = occurrences.get(fingerprint, 0)
        occurrences[fingerprint] = occurrence + 1
        scope = outer_runtime.config.get("configurable", {}).get("checkpoint_ns", "")
        configurable["checkpoint_ns"] = (
            f"{scope}|js_dispatch:{fingerprint}_{occurrence}"
        )
        configurable[_ACCOUNTING_KEY] = owner
        configurable["__pregel_durability"] = "sync"
        configurable["__deepagents_js_cost_loop"] = asyncio.get_running_loop()
        scratchpad = configurable.get("__pregel_scratchpad")
        if scratchpad is not None:
            configurable["__pregel_scratchpad"] = dataclasses.replace(
                scratchpad, subgraph_counter=itertools.count().__next__
            )
        config: RunnableConfig = {**runtime.config, "configurable": configurable}
        invocation = asyncio.current_task()
        if invocation is not None:
            active.add(invocation)
        try:
            return await tool.arun(
                {
                    "description": description,
                    "subagent_type": subagent_type,
                    "runtime": dataclasses.replace(runtime, config=config),
                },
                config=config,
                tool_call_id=runtime.tool_call_id,
            )
        finally:
            if invocation is not None:
                active.discard(invocation)

    return tool.model_copy(update={"coroutine": invoke})


def record_cost_receipt(amount: float) -> None:
    """Persist one locally priced node delta before its graph update returns."""
    configurable = get_config().get("configurable", {})
    owner = configurable.get(_ACCOUNTING_KEY)
    saver = configurable.get("__pregel_checkpointer")
    if not owner or not isinstance(saver, BaseCheckpointSaver) or amount <= 0:
        return
    namespace = configurable.get("checkpoint_ns", "")
    config: RunnableConfig = {
        "configurable": {
            "thread_id": configurable["thread_id"],
            "checkpoint_ns": f"{owner}|receipt:{_digest(namespace)}",
        }
    }
    loop = configurable["__deepagents_js_cost_loop"]
    asyncio.run_coroutine_threadsafe(
        _put_receipt(saver, config, owner, amount), loop
    ).result()


async def _put_receipt(
    saver: BaseCheckpointSaver, config: RunnableConfig, owner: str, amount: float
) -> None:
    """Persist a node's first priced delta without replacing it on replay."""
    if await saver.aget_tuple(config) is not None:
        return
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"cost_usd": amount}
    checkpoint["channel_versions"] = {"cost_usd": checkpoint["id"]}
    metadata: _ReceiptMetadata = {"js_cost_owner": owner}
    await saver.aput(config, checkpoint, metadata, checkpoint["channel_versions"])


async def _receipt_total(config: RunnableConfig, owner: str) -> float:
    """Return the sum of the latest local receipts owned by this eval."""
    configurable = config.get("configurable", {})
    saver = configurable.get("__pregel_checkpointer")
    if not isinstance(saver, BaseCheckpointSaver):
        return 0.0
    seen: set[str] = set()
    total = 0.0
    async for receipt in saver.alist(
        {"configurable": {"thread_id": configurable["thread_id"]}},
        filter={"js_cost_owner": owner},
    ):
        namespace = receipt.config["configurable"]["checkpoint_ns"]
        if namespace in seen:
            continue
        seen.add(namespace)
        amount = receipt.checkpoint["channel_values"].get("cost_usd")
        if (
            isinstance(amount, int | float)
            and not isinstance(amount, bool)
            and math.isfinite(amount)
            and amount > 0
        ):
            total += amount
    return total
