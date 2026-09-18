"""Durable cost-only transport for deterministic JavaScript task replays."""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

from langchain_core.runnables.config import ensure_config
from langchain_core.tools import StructuredTool
from langchain_quickjs import CodeInterpreterMiddleware
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphBubbleUp
from langgraph.func import task
from langgraph.types import Command

from deepagents_code.cost_tracking import _CostTransfer, _parent_checkpoint_scope

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.messages import ToolMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.prebuilt.tool_node import ToolCallRequest, ToolRuntime


logger = logging.getLogger(__name__)


class CostAwareCodeInterpreterMiddleware(CodeInterpreterMiddleware):
    """Carry completed task costs without merging child messages or state."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Return the eval result with only its owned accounting transfers."""
        if request.tool not in self.tools:
            return await handler(request)
        transfers: dict[str, _CostTransfer] = {}
        pending: dict[str, RunnableConfig] = {}
        tools = [
            _cost_task(tool, transfers, pending, request.runtime)
            if isinstance(tool, StructuredTool) and tool.name == "task"
            else tool
            for tool in request.runtime.tools
        ]
        runtime = dataclasses.replace(request.runtime, tools=tools)
        result = await handler(request.override(runtime=runtime))
        owner = _parent_checkpoint_scope(
            runtime.config.get("configurable", {}).get("checkpoint_ns")
        )
        for config in list(pending.values()):
            transfers.update(await _failed_cost(config, owner))
        if not transfers:
            return result
        if isinstance(result, Command):
            update = dict(result.update) if isinstance(result.update, Mapping) else {}
            update["_session_cost_transfers"] = transfers
            return dataclasses.replace(result, update=update)
        return Command(
            update={"messages": [result], "_session_cost_transfers": transfers}
        )


def _cost_transfers(result: object, scope: str, owner: str) -> dict[str, _CostTransfer]:
    """Return valid totals addressed to this dispatch's immediate owner."""
    if not isinstance(result, Command) or not isinstance(result.update, Mapping):
        return {}
    pending = result.update.get("_session_cost_transfers")
    if not isinstance(pending, Mapping):
        return {}
    return {
        source: {"owner_scope": owner, "cost_usd": float(amount)}
        for source, transfer in pending.items()
        if source == scope
        and isinstance(transfer, Mapping)
        and transfer.get("owner_scope") == _parent_checkpoint_scope(scope)
        and _valid_cost(amount := transfer.get("cost_usd"))
    }


def _valid_cost(amount: object) -> bool:
    """Return whether a value is a finite, positive cost."""
    return (
        isinstance(amount, int | float)
        and not isinstance(amount, bool)
        and math.isfinite(amount)
        and amount > 0
    )


async def _failed_cost(config: RunnableConfig, owner: str) -> dict[str, _CostTransfer]:
    """Return only spend already persisted by a failed child graph."""
    configurable = config.get("configurable", {})
    saver = configurable.get("__pregel_checkpointer")
    if not isinstance(saver, BaseCheckpointSaver):
        return {}
    scope = configurable.get("checkpoint_ns", "")
    checkpoint = await saver.aget_tuple(
        {
            "configurable": {
                "thread_id": configurable["thread_id"],
                "checkpoint_ns": scope,
            }
        }
    )
    if checkpoint is None:
        return {}
    values = checkpoint.checkpoint["channel_values"]
    amount = values.get("_session_cost_usd", 0.0)
    total = float(amount) if _valid_cost(amount) else 0.0
    pending = values.get("_session_cost_transfers", {})
    if isinstance(pending, Mapping):
        total += sum(
            float(transfer["cost_usd"])
            for transfer in pending.values()
            if isinstance(transfer, Mapping)
            and transfer.get("owner_scope") == scope
            and _valid_cost(transfer.get("cost_usd"))
        )
    return {scope: {"owner_scope": owner, "cost_usd": total}} if total > 0 else {}


def _cost_task(
    tool: StructuredTool,
    transfers: dict[str, _CostTransfer],
    pending: dict[str, RunnableConfig],
    outer_runtime: ToolRuntime,
) -> StructuredTool:
    """Return an eval-local proxy, leaving direct SDK task calls unchanged."""
    owner = _parent_checkpoint_scope(
        outer_runtime.config.get("configurable", {}).get("checkpoint_ns")
    )

    async def invoke(
        description: str, subagent_type: str, runtime: ToolRuntime
    ) -> object:
        async def js_subagent_cost() -> tuple[
            object, dict[str, _CostTransfer], str | None
        ]:
            config = ensure_config()
            configurable = config.get("configurable", {})
            scope = configurable.get("checkpoint_ns", "")
            pending[scope] = config
            response_format_key = "__deepagents_subagent_response_format"
            if response_format_key in runtime.config.get("configurable", {}):
                configurable[response_format_key] = runtime.config["configurable"][
                    response_format_key
                ]
            try:
                result = await tool.arun(
                    {
                        "description": description,
                        "subagent_type": subagent_type,
                        "runtime": dataclasses.replace(runtime, config=config),
                    },
                    config=config,
                    tool_call_id=runtime.tool_call_id,
                )
            except GraphBubbleUp:
                raise
            except Exception as exc:
                logger.debug("JS subagent failed", exc_info=True)
                costs = await _failed_cost(config, owner)
                pending.pop(scope, None)
                return None, costs, str(exc)
            costs = _cost_transfers(result, scope, owner)
            pending.pop(scope, None)
            if isinstance(result, Command) and isinstance(result.update, Mapping):
                result = Command(update={"messages": result.update.get("messages", [])})
            return result, costs, None

        dispatch_task = task(js_subagent_cost)
        result, costs, error = await dispatch_task()
        transfers.update(costs)
        if error is not None:
            raise RuntimeError(error)
        return result

    return tool.model_copy(update={"coroutine": invoke})
