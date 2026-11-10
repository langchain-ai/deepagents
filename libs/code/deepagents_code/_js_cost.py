"""Cost receipt transport and interrupted report recovery for JavaScript subagents."""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import itertools
import json
import logging
import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, NotRequired, TypedDict, override

from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langchain_quickjs import CodeInterpreterMiddleware
from langchain_quickjs._format import (  # noqa: PLC2701  # Match the interpreter's report unwrapping.
    coerce_tool_output_for_ptc,
)
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    CheckpointMetadata,
    empty_checkpoint,
)
from langgraph.config import get_config
from langgraph.errors import GraphBubbleUp, NodeCancelledError
from langgraph.types import Command

from deepagents_code.cost_tracking import (
    _empty_cost_breakdown,
    _merge_cost_breakdowns,
    _parent_checkpoint_scope,
)
from deepagents_code.thread_ownership import OWNER_KEY

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest
    from langchain_core.messages import ToolMessage
    from langchain_core.runnables import RunnableConfig
    from langchain_quickjs.middleware import REPLState
    from langgraph.prebuilt.tool_node import ToolCallRequest, ToolRuntime
    from langgraph.runtime import Runtime

    from deepagents_code.cost_tracking import CostBreakdown

logger = logging.getLogger(__name__)

_ACCOUNTING_KEY = "__deepagents_js_cost_owner"
_RESPONSE_FORMAT_KEY = "__deepagents_subagent_response_format"


class _DispatchResult(TypedDict):
    index: int
    subagent_type: str
    description: str
    result: NotRequired[object]


class _ReceiptMetadata(CheckpointMetadata):
    js_cost_owner: str


def _digest(value: str) -> str:
    """Return a hashed identity without storing prompts in checkpoint names."""
    return hashlib.sha256(value.encode()).hexdigest()


class CostAwareCodeInterpreterMiddleware(CodeInterpreterMiddleware):
    """Transport cost receipts and preserve reports from interrupted JavaScript."""

    @override
    async def abefore_agent(
        self, state: REPLState, runtime: Runtime[None]
    ) -> dict[str, object] | None:
        """Return interrupted reports alongside the REPL's normal preparation."""
        update = await super().abefore_agent(state, runtime)
        messages = await _interrupted_reports(get_config(), state)
        if messages:
            return {**(update or {}), "messages": messages}
        return update

    @override
    def _prepare_for_call(self, request: ModelRequest) -> str:
        """Return a compact reference while retaining all PTC bindings."""
        from langchain_core.tools import BaseTool
        from langchain_quickjs.middleware import render_ptc_prompt

        tools = [tool for tool in request.tools if isinstance(tool, BaseTool)]
        prompt = super()._prepare_for_call(request.override(tools=list(tools)))
        slot = self._slot_id(request.state)
        cached = self._ptc_prompt_by_slot.get(slot)
        if cached is None:
            return prompt
        exposed, reference = cached
        visible = [
            tool for tool in exposed if not (tool.extras or {}).get("defer_loading")
        ]
        if len(visible) == len(exposed):
            return prompt
        replacement = render_ptc_prompt(visible, tool_name=self._tool_name)
        return prompt.removesuffix(reference) + replacement

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """Preserve synchronous execution; receipt transport is async-only.

        Args:
            request: Tool call and its runtime context.
            handler: Synchronous tool executor.

        Returns:
            The tool result without modification.
        """
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """Return owned receipts after settling all dispatched children."""  # noqa: DOC501  # Control-flow signals are propagated unchanged.
        if request.tool not in self.tools:
            return await handler(request)
        configurable = request.runtime.config.get("configurable", {})
        scope = configurable.get("checkpoint_ns", "")
        owner = configurable.get(_ACCOUNTING_KEY)
        accounting = (
            owner or f"{scope}|js_cost:{_digest(request.runtime.tool_call_id or '')}"
        )
        active: set[asyncio.Task] = set()
        dispatched: dict[str, _DispatchResult] = {}
        completed: dict[str, _DispatchResult] = {}
        interrupted = False
        tools = [
            _cost_task(tool, request.runtime, accounting, active, dispatched, completed)
            if isinstance(tool, StructuredTool) and tool.name == "task"
            else tool
            for tool in request.runtime.tools
        ]
        runtime = dataclasses.replace(request.runtime, tools=tools)
        try:
            result = await handler(request.override(runtime=runtime))
        except asyncio.CancelledError, GraphBubbleUp, NodeCancelledError:
            interrupted = True
            raise
        finally:
            for invocation in list(active):
                if not invocation.cancelling():
                    invocation.cancel()
            await asyncio.gather(*active, return_exceptions=True)
            if interrupted and dispatched:
                await _save_interrupted_reports(runtime, dispatched, completed)
        await _save_interrupted_reports(
            runtime, dispatched, completed, interrupted=False
        )
        if owner:
            return result
        receipt = await _receipt_total(runtime.config, accounting)
        if receipt is None:
            return result
        total, breakdown = receipt
        transfers = {
            accounting: {
                "owner_scope": _parent_checkpoint_scope(scope),
                "cost_usd": total,
                "breakdown": breakdown,
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
    dispatched: dict[str, _DispatchResult] | None = None,
    completed: dict[str, _DispatchResult] | None = None,
) -> StructuredTool:
    """Return a task proxy with request-isolated child graph checkpoints."""
    occurrences: dict[str, int] = {}
    dispatched = dispatched if dispatched is not None else {}
    completed = completed if completed is not None else {}

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
        key = f"{fingerprint}_{occurrence}"
        dispatched[key] = {
            "index": len(dispatched),
            "subagent_type": subagent_type,
            "description": description[:200],
        }
        scope = outer_runtime.config.get("configurable", {}).get("checkpoint_ns", "")
        configurable["checkpoint_ns"] = (
            f"{scope}|js_dispatch:{fingerprint}_{occurrence}"
        )
        configurable[_ACCOUNTING_KEY] = owner
        # The SDK's task closure owns the actual child ainvoke, including
        # dynamic response schemas. Public durability="sync" belongs there;
        # until that boundary exposes it, inherit LangGraph 1.2's config key.
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
            result = await tool.arun(
                {
                    "description": description,
                    "subagent_type": subagent_type,
                    "runtime": dataclasses.replace(runtime, config=config),
                },
                config=config,
                tool_call_id=runtime.tool_call_id,
            )
        except asyncio.CancelledError, GraphBubbleUp, NodeCancelledError:
            raise
        except Exception as exc:
            logger.debug("Subagent %r failed", subagent_type, exc_info=exc)
            return {
                "error": (
                    f"Subagent {subagent_type!r} failed: "
                    f"{type(exc).__name__}: {str(exc)[:300]}. You may retry this task."
                ),
                "failed": True,
            }
        else:
            completed[key] = {
                **dispatched[key],
                "result": coerce_tool_output_for_ptc(result),
            }
            return result
        finally:
            if invocation is not None:
                active.discard(invocation)

    return tool.model_copy(update={"coroutine": invoke})


async def _save_interrupted_reports(
    runtime: ToolRuntime,
    dispatched: dict[str, _DispatchResult],
    completed: dict[str, _DispatchResult],
    *,
    interrupted: bool = True,
) -> None:
    """Persist partial results separately from cost receipts and graph state."""
    configurable = dict(runtime.config.get("configurable", {}))
    saver = configurable.get("__pregel_checkpointer")
    if not isinstance(saver, BaseCheckpointSaver):
        return
    scope = _parent_checkpoint_scope(configurable.get("checkpoint_ns", ""))
    configurable["checkpoint_ns"] = (
        f"{scope}|js_interrupted:{_digest(runtime.tool_call_id or '')}"
    )
    configurable.pop("checkpoint_id", None)
    config: RunnableConfig = {"configurable": configurable}
    if not interrupted and await saver.aget_tuple(config) is None:
        return
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {
        "tool_call_id": runtime.tool_call_id,
        "interrupted": interrupted,
        "completed": list(completed.values()),
        "unfinished": [
            value for key, value in dispatched.items() if key not in completed
        ],
    }
    checkpoint["channel_versions"] = dict.fromkeys(
        checkpoint["channel_values"], checkpoint["id"]
    )
    await saver.aput(
        config,
        checkpoint,
        {"js_interrupted_scope": scope},
        checkpoint["channel_versions"],
    )


async def _interrupted_reports(
    config: RunnableConfig, state: REPLState
) -> list[HumanMessage]:
    """Return saved partial reports once per tool call in conversation history."""
    configurable = config.get("configurable", {})
    saver = configurable.get("__pregel_checkpointer")
    if not isinstance(saver, BaseCheckpointSaver):
        return []
    seen = {message.id for message in state["messages"]}
    messages = []
    async for receipt in saver.alist(
        {"configurable": {"thread_id": configurable["thread_id"]}},
        filter={
            "js_interrupted_scope": _parent_checkpoint_scope(
                configurable.get("checkpoint_ns", "")
            )
        },
    ):
        values = receipt.checkpoint["channel_values"]
        message_id = f"js-interrupted:{values['tool_call_id']}"
        if message_id in seen:
            continue
        seen.add(message_id)
        if not values["interrupted"]:
            continue
        messages.append(
            HumanMessage(
                id=message_id,
                content=(
                    "Recovered subagent reports from interrupted js_eval. "
                    "Use completed reports; only unfinished briefs need further work.\n"
                    + json.dumps(values, default=str)
                ),
            )
        )
    return messages


def record_cost_receipt(
    amount: float, *, breakdown: CostBreakdown | None = None
) -> None:
    """Persist one local node delta before its graph update returns.

    Args:
        amount: Dollars priced here, excluding claimed descendant transfers.
        breakdown: Local requests, including free and unpriceable usage.
            Omitted detail is preserved as incomplete legacy accounting.
    """
    configurable = get_config().get("configurable", {})
    owner = configurable.get(_ACCOUNTING_KEY)
    saver = configurable.get("__pregel_checkpointer")
    if not owner or not isinstance(saver, BaseCheckpointSaver):
        return
    if amount <= 0 and (breakdown is None or breakdown["request_count"] == 0):
        return
    namespace = configurable.get("checkpoint_ns", "")
    config: RunnableConfig = {
        "configurable": {
            "thread_id": configurable["thread_id"],
            "checkpoint_ns": f"{owner}|receipt:{_digest(namespace)}",
        }
    }
    if OWNER_KEY in configurable:
        config["configurable"][OWNER_KEY] = configurable[OWNER_KEY]
    loop = configurable["__deepagents_js_cost_loop"]
    asyncio.run_coroutine_threadsafe(
        _put_receipt(saver, config, owner, amount, breakdown), loop
    ).result()


async def _put_receipt(
    saver: BaseCheckpointSaver,
    config: RunnableConfig,
    owner: str,
    amount: float,
    breakdown: CostBreakdown | None,
) -> None:
    """Persist a node's first usage delta without replacing it on replay."""
    if await saver.aget_tuple(config) is not None:
        return
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"cost_usd": amount}
    if breakdown is not None:
        checkpoint["channel_values"]["breakdown"] = breakdown
    checkpoint["channel_versions"] = dict.fromkeys(
        checkpoint["channel_values"], checkpoint["id"]
    )
    metadata: _ReceiptMetadata = {"js_cost_owner": owner}
    await saver.aput(config, checkpoint, metadata, checkpoint["channel_versions"])


async def _receipt_total(
    config: RunnableConfig, owner: str
) -> tuple[float, CostBreakdown] | None:
    """Return the sum of the latest local receipts owned by this eval."""
    configurable = config.get("configurable", {})
    saver = configurable.get("__pregel_checkpointer")
    if not isinstance(saver, BaseCheckpointSaver):
        return None
    seen: set[str] = set()
    total = 0.0
    breakdown = _empty_cost_breakdown()
    async for receipt in saver.alist(
        {"configurable": {"thread_id": configurable["thread_id"]}},
        filter={"js_cost_owner": owner},
    ):
        namespace = receipt.config["configurable"]["checkpoint_ns"]
        if namespace in seen:
            continue
        seen.add(namespace)
        values = receipt.checkpoint["channel_values"]
        amount = values.get("cost_usd")
        if (
            isinstance(amount, int | float)
            and not isinstance(amount, bool)
            and math.isfinite(amount)
            and amount >= 0
        ):
            total += amount
            detail = values.get("breakdown")
            if not isinstance(detail, Mapping):
                # Keep legacy dollars without inventing requests or attribution.
                detail = _empty_cost_breakdown(historical_complete=False)
                detail["total_cost_usd"] = amount
            breakdown = _merge_cost_breakdowns(breakdown, detail)
        else:
            breakdown["historical_complete"] = False
    return (total, breakdown) if seen else None
