"""Identify conversation turns independently of model selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from langchain_core.messages import BaseMessage


@dataclass(frozen=True)
class RoutingTurn:
    """A user turn's stable key and the conversation available when it arrived.

    Args:
        key: Identifies the turn across model calls and checkpoint restores.
        messages: Context to assess, ending with the latest user request.
    """

    key: str
    messages: Sequence[BaseMessage]


type TurnReader = Callable[[Sequence[BaseMessage]], RoutingTurn | None]


def read_human_turn(messages: Sequence[BaseMessage]) -> RoutingTurn | None:
    """Read the latest human turn, using its position if no message ID was assigned.

    Args:
        messages: The agent's conversation history.

    Returns:
        The latest human turn and its preceding context, or no turn yet.
    """
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, HumanMessage):
            return RoutingTurn(message.id or f"human:{index}", messages[: index + 1])
    return None
