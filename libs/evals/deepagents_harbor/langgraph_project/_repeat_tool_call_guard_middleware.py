"""Middleware that breaks pathological repeated-identical tool-call loops.

Why: on hard Terminal-Bench tasks (theorem proving, reverse engineering, source
builds, git-history recovery) the agent re-issues the *same* tool call with
byte-identical arguments dozens of times, with empty reasoning between calls,
burning 3-16M tokens until LangGraph raises `GraphRecursionError`. Confirmed cases
(from LangSmith trace analysis): `strings|grep` for a password pattern called 210x,
`ls -la` 211x, an identical import script 12x, `git log --all ...` 43x. The model is
NOT exercising judgment between repeats, so a system-prompt nudge does not help — a
deterministic guard is required.

This middleware keeps a small ring buffer of recent tool-call signatures. When an
incoming call is identical to >= `threshold` of the last `window` calls, it does NOT
execute it — it returns an error `ToolMessage` telling the agent the repetition is
unproductive and to change approach. This converts a multi-million-token re-execution
loop into a handful of cheap, informative blocks; paired with a lowered
`recursion_limit`, it bounds the worst case and lets the run close with a real answer
instead of a recursion crash.

Signature = (tool_name, canonicalized args). For shell/`execute` calls the command is
compared with trailing whitespace stripped, so `cmd` and `cmd\n` count as identical.
State is per-middleware-instance; Harbor builds one graph (hence one instance) per
trial, each a single thread, so a plain instance-level buffer is correct here.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage


def _canonical_args(args: Any) -> str:
    """Stable string form of tool args; string values are right-stripped."""
    if isinstance(args, dict):
        norm = {
            key: (value.rstrip() if isinstance(value, str) else value)
            for key, value in args.items()
        }
        return json.dumps(norm, sort_keys=True, default=str)
    if isinstance(args, str):
        return args.rstrip()
    return json.dumps(args, default=str, sort_keys=True)


class RepeatToolCallGuardMiddleware(AgentMiddleware):
    """Block tool calls repeated with identical args, and nudge a change of approach."""

    def __init__(self, *, window: int = 8, threshold: int = 4) -> None:
        super().__init__()
        self._window = window
        self._threshold = threshold
        self._recent: deque[tuple[str, str]] = deque(maxlen=window)
        self._block_counts: dict[tuple[str, str], int] = {}

    def _maybe_block(self, request: Any) -> ToolMessage | None:
        tool_call = getattr(request, "tool_call", None) or {}
        name = str(tool_call.get("name", ""))
        signature = (name, _canonical_args(tool_call.get("args")))
        prior_matches = sum(1 for seen in self._recent if seen == signature)
        self._recent.append(signature)
        if prior_matches < self._threshold:
            return None
        self._block_counts[signature] = self._block_counts.get(signature, 0) + 1
        content = (
            f"[loop-guard] You have already called `{name}` with identical arguments "
            f"{prior_matches + 1} times recently without progress; this call was BLOCKED "
            "and NOT executed. Repeating it will not help. Do something different: change "
            "the command or its arguments, inspect a different file or location, or try "
            "another approach. If you are genuinely stuck, stop repeating — write your best "
            "solution to the required output path now, then summarize what you tried and "
            "why you are blocked."
        )
        return ToolMessage(
            content=content,
            tool_call_id=str(tool_call.get("id", "")),
            name=name,
            status="error",
        )

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        blocked = self._maybe_block(request)
        return blocked if blocked is not None else handler(request)

    async def awrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Awaitable[Any]],
    ) -> Any:
        blocked = self._maybe_block(request)
        if blocked is not None:
            return blocked
        return await handler(request)
