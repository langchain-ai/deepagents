"""Feature detection for langgraph's `DeltaChannel`.

`DeltaChannel` landed in `langgraph>=1.2.0`. On older installs the import
fails, so deep agents fall back to `add_messages` for the `messages` channel
and warn the user to upgrade — without it, checkpoint storage grows O(N^2)
with conversation length instead of O(N).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from langgraph.graph.message import add_messages

from deepagents._messages_reducer import _messages_delta_reducer

if TYPE_CHECKING:
    from collections.abc import Callable

    from langgraph.channels.base import BaseChannel

try:
    from langgraph.channels.delta import DeltaChannel

    _DELTA_CHANNEL_SUPPORTED = True
except ImportError:
    _DELTA_CHANNEL_SUPPORTED = False

_MESSAGES_SNAPSHOT_FREQUENCY = 50

_UNSUPPORTED_MESSAGE = (
    "The installed `langgraph` version does not support `DeltaChannel` "
    "(added in `langgraph>=1.2.0`). Deep agents fall back to `add_messages` "
    "for the `messages` channel, which grows checkpoint storage O(N^2) with "
    "conversation length. Upgrade langgraph (e.g. `pip install -U langgraph`) "
    "to enable O(N) checkpoint growth."
)


def delta_channel_supported() -> bool:
    """`True` if the installed `langgraph` exposes `DeltaChannel`."""
    return _DELTA_CHANNEL_SUPPORTED


def messages_channel_reducer() -> BaseChannel | Callable:
    """Reducer for the `messages` channel.

    Returns a `DeltaChannel` when supported, otherwise `add_messages`.
    """
    if _DELTA_CHANNEL_SUPPORTED:
        return DeltaChannel(_messages_delta_reducer, snapshot_frequency=_MESSAGES_SNAPSHOT_FREQUENCY)  # ty: ignore[invalid-argument-type]
    return add_messages


def warn_if_delta_channel_unsupported(*, stacklevel: int = 2) -> None:
    """Warn (once per call site) when langgraph predates `DeltaChannel`."""
    if _DELTA_CHANNEL_SUPPORTED:
        return
    warnings.warn(_UNSUPPORTED_MESSAGE, UserWarning, stacklevel=stacklevel + 1)
