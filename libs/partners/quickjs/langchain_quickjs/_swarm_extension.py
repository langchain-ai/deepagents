"""Swarm as an interpreter extension.

A first-party ``InterpreterExtension`` that makes swarm dispatch importable
from guest JS as ``import { swarm } from "swarm"`` — without going through
PTC. The extension registers the swarm dispatch closure as a host function,
ships a tiny JS module that forwards to it, and contributes a system-prompt
fragment so the agent knows to reach for it.

This is the deterministic-capability counterpart to ``create_swarm_task_tool``
(which targets the PTC path): same dispatch logic via ``build_swarm_dispatch``,
delivered as an always-available extension instead of a PTC tool.

Usage:

    from langchain_quickjs import CodeInterpreterMiddleware, swarm

    CodeInterpreterMiddleware(
        extensions=[swarm(subagents=[screener, classifier], default_model=model)],
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from quickjs_rs import ModuleScope

from langchain_quickjs._swarm_task import build_swarm_dispatch

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

    from langchain_quickjs._extensions import ExtensionContext
    from langchain_quickjs._swarm_task import SwarmDispatch, SwarmSubAgent

# Host symbol the JS module forwards to. Extension-owned, distinctive
# prefix so it doesn't collide with another extension's globals.
_DISPATCH_SYMBOL = "__swarm_dispatch"

# The guest-facing module. A thin async wrapper around the host symbol —
# the symbol name is fixed source, never interpolated from guest input.
_SWARM_MODULE_SOURCE = """\
export async function swarm(opts) {
  return await __swarm_dispatch(opts);
}
"""

_SYSTEM_PROMPT = """\
## swarm

`swarm` dispatches a task to a subagent from inside the code interpreter.
Import and call it from an `eval`:

```js
import { swarm } from "swarm";
const result = await swarm({
  description: "Triage this trace",
  subagentType: "screener",   // omit for mode "invoke"
  mode: "agent",              // "agent" (default) or "invoke"
});
```

Use it to fan work out to subagents in code (loops, `Promise.all`) rather
than one task at a time.
"""


@dataclass(frozen=True)
class SwarmExtension:
    """Interpreter extension exposing swarm dispatch to guest JS.

    Construct via the :func:`swarm` factory. ``on_setup`` registers the
    dispatch host function and the ``swarm`` module on every fresh context.
    """

    dispatch: SwarmDispatch
    system_prompt: str | None = _SYSTEM_PROMPT

    def on_setup(self, ctx: ExtensionContext) -> None:
        dispatch = self.dispatch

        @ctx.function(name=_DISPATCH_SYMBOL)
        async def _swarm_dispatch(payload: Any = None) -> str:
            # Guest payload is untrusted: require a dict and pull only the
            # known keys. camelCase from JS maps to snake_case kwargs.
            if not isinstance(payload, dict):
                msg = "swarm() expects an options object"
                raise TypeError(msg)
            return await dispatch(
                payload.get("description"),
                payload.get("subagentType"),
                payload.get("responseSchema"),
                payload.get("mode"),
            )

        ctx.module(
            ModuleScope({"swarm": ModuleScope({"index.js": _SWARM_MODULE_SOURCE})})
        )


def swarm(
    *,
    subagents: list[SwarmSubAgent] | None = None,
    default_model: str | BaseChatModel,
) -> SwarmExtension:
    """Build a swarm interpreter extension.

    The returned extension makes ``import { swarm } from "swarm"`` available
    in the code interpreter, dispatching to subagents via the same logic as
    ``create_swarm_task_tool`` — but as an always-available extension rather
    than a PTC tool.

    Args:
        subagents: Subagent specifications for dispatch targets.
        default_model: Default model for subagents without their own, and
            for ``invoke`` mode direct calls.

    Returns:
        A ``SwarmExtension`` to pass to ``CodeInterpreterMiddleware(extensions=...)``.
    """
    dispatch = build_swarm_dispatch(subagents=subagents, default_model=default_model)
    return SwarmExtension(dispatch=dispatch)


__all__ = ["SwarmExtension", "swarm"]
