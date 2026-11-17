"""Helpers for reaching a middleware's slots in the shared registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from langchain_core.tools import BaseTool

    from langchain_quickjs import CodeInterpreterMiddleware
    from langchain_quickjs._repl import _Slot


def slot_of(mw: CodeInterpreterMiddleware, slot_id: str) -> Any:
    """Return `mw`'s slot for `slot_id`, or `None`."""
    return mw._registry._slots.get((slot_id, mw._slot_config))


def slots_of(mw: CodeInterpreterMiddleware) -> list[Any]:
    """Return every slot `mw` has in the registry, least recent first."""
    return [
        slot
        for (_, config), slot in mw._registry._slots.items()
        if config == mw._slot_config
    ]


def warm_slot(mw: CodeInterpreterMiddleware, slot_id: str) -> Any:
    """Return `mw`'s slot for `slot_id`, creating it, without holding a lease."""
    slot: _Slot = mw._registry.acquire(slot_id, mw._slot_config)
    mw._registry.release(slot)
    return slot


def drop_slots(mw: CodeInterpreterMiddleware) -> None:
    """Close and remove every slot `mw` has in the registry."""
    for key in [key for key in mw._registry._slots if key[1] == mw._slot_config]:
        mw._registry._slots.pop(key).close()


def install_tools(slot: _Slot, tools: Sequence[BaseTool]) -> None:
    """Bind `tools` on the slot's turn Context, as an eval with them would."""
    slot.eval_sync("undefined", tools=tools)
