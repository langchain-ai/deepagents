"""Mid-turn steering inbox shared by the Textual client and agent server.

Queued chat input normally waits for the running turn to end. *Steering* hands a
queued message to the agent **during** the turn: the client writes it into a
per-thread LangGraph Store namespace, and `SteeringMiddleware` drains that inbox
at the agent's next model call, so the message arrives as a `HumanMessage` in the
live run instead of the run having to be cancelled and restarted.

The Store is the only live client-to-running-run channel the platform offers
today (`POST /threads/{id}/state` is rejected with `409` while a run is in
flight), and it is the same channel `AsyncApprovalHITLMiddleware` already uses to
read the live approval mode mid-run.

Item layout, one item per steered message:

- namespace: `("deepagents_code", "steer", <sha256 of thread id>)`
- key: the zero-padded `seq`, so lexical order matches delivery order
- value: `SteerPayload`

`seq` is a wall-clock millisecond stamp, which keeps increasing across client
restarts on the same thread. Consumption is tracked by a checkpointed watermark
in agent state rather than by deleting items, so a run that dies before its
checkpoint is written redelivers the message instead of dropping it.

This module deliberately imports no LangChain/LangGraph module at import time:
the Textual client writes the inbox and must stay off the slow-import path. The
middleware that consumes the inbox lives in `steering_middleware`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from inspect import isawaitable
from typing import TYPE_CHECKING, NotRequired, TypedDict

from deepagents_code._env_vars import STEERING, is_env_truthy

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

STEER_NAMESPACE_ROOT: tuple[str, str] = ("deepagents_code", "steer")
"""Store namespace prefix for per-thread steering inboxes."""

STEER_ITEM_TTL_MINUTES = 120
"""Expiry for inbox items, so an abandoned session cannot strand them forever."""

MAX_PENDING_STEERS = 20
"""Largest number of inbox items read (and delivered) in one model boundary."""

_SEQ_KEY_WIDTH = 20
"""Zero-padding width for keys, so lexical key order equals numeric seq order."""


class SteerPayload(TypedDict):
    """Stored steering item written by the client and read by the server."""

    seq: int
    text: str
    literal_user_text: str
    turn_id: NotRequired[str | None]


@dataclass(frozen=True)
class SteerItem:
    """A validated inbox item ready to be injected into the running turn."""

    seq: int
    key: str
    text: str
    literal_user_text: str
    turn_id: str | None


class _SeqCounter:
    """Monotonic millisecond sequence source shared by one client process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last = 0

    def next(self) -> int:
        """Return a strictly increasing millisecond-resolution sequence number.

        Returns:
            The current wall clock in milliseconds, bumped past the previously
            issued value so two steers submitted in the same millisecond still
            order deterministically.
        """
        with self._lock:
            seq = max(time.time_ns() // 1_000_000, self._last + 1)
            self._last = seq
            return seq


_SEQ = _SeqCounter()


def steering_enabled() -> bool:
    """Whether queued messages may be steered into a running turn.

    Returns:
        `True` unless `DEEPAGENTS_CODE_STEERING` is set to a falsy value.
    """
    return is_env_truthy(STEERING, default=True)


def steer_namespace(thread_id: str) -> tuple[str, ...]:
    """Return the Store namespace holding a thread's steering inbox.

    The thread id is hashed (matching `approval_mode_key`) so raw thread ids are
    not exposed in Store namespaces.

    Args:
        thread_id: LangGraph thread id for the active session.

    Returns:
        Namespace tuple scoped to exactly this thread.
    """
    digest = sha256(thread_id.encode("utf-8")).hexdigest()
    return (*STEER_NAMESPACE_ROOT, digest)


def steer_key(seq: int) -> str:
    """Return the Store key for a sequence number.

    Args:
        seq: Sequence number from `next_steer_seq`.

    Returns:
        Zero-padded key so lexical ordering matches delivery order.
    """
    return f"{seq:0{_SEQ_KEY_WIDTH}d}"


def next_steer_seq() -> int:
    """Return the next steering sequence number for this client process.

    Returns:
        Strictly increasing millisecond stamp.
    """
    return _SEQ.next()


def steer_payload(
    text: str,
    *,
    seq: int,
    literal_user_text: str | None = None,
    turn_id: str | None = None,
) -> SteerPayload:
    """Build the stored payload for one steered message.

    Args:
        text: Message text delivered to the agent.
        seq: Sequence number from `next_steer_seq`.
        literal_user_text: Text as typed, when it differs from `text`. Used for
            the trusted prompt metadata the Auto-approval classifier reads.
        turn_id: Id of the turn being steered, when one is active.

    Returns:
        JSON-serializable Store value.
    """
    return SteerPayload(
        seq=seq,
        text=text,
        literal_user_text=literal_user_text if literal_user_text is not None else text,
        turn_id=turn_id,
    )


def parse_steer_item(key: object, value: object) -> SteerItem | None:
    """Validate one raw Store item, or reject it.

    Every field is type-checked before use: a malformed item is dropped rather
    than partially trusted, since the inbox is consumed inside a live graph run
    where raising would fail the user's turn.

    Args:
        key: Raw Store key.
        value: Raw Store value.

    Returns:
        The validated item, or `None` when it cannot be trusted.
    """
    if not isinstance(key, str) or not key:
        return None
    if not isinstance(value, Mapping):
        return None
    seq = value.get("seq")
    text = value.get("text")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq <= 0:
        return None
    if not isinstance(text, str) or not text.strip():
        return None
    literal = value.get("literal_user_text")
    turn_id = value.get("turn_id")
    return SteerItem(
        seq=seq,
        key=key,
        text=text,
        literal_user_text=literal if isinstance(literal, str) and literal else text,
        turn_id=turn_id if isinstance(turn_id, str) and turn_id else None,
    )


def coerce_consumed_seq(value: object) -> int:
    """Narrow a checkpointed consumption watermark to a usable integer.

    Args:
        value: Raw watermark read back from agent state.

    Returns:
        The watermark, or `0` when it is missing or not a positive integer.
    """
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return 0


async def awrite_steer(
    agent: object,
    thread_id: str,
    *,
    text: str,
    literal_user_text: str | None = None,
    turn_id: str | None = None,
) -> int | None:
    """Write one steering message into a thread's inbox.

    Args:
        agent: Agent object. Remote agents expose `aput_store_item`.
        thread_id: LangGraph thread id for the active session.
        text: Message text to deliver to the running turn.
        literal_user_text: Text as typed, when it differs from `text`.
        turn_id: Id of the turn being steered, when one is active.

    Returns:
        The written sequence number, or `None` when the agent has no Store
        writer (for example a local in-process graph).
    """
    put = getattr(agent, "aput_store_item", None)
    if put is None:
        return None
    seq = next_steer_seq()
    await put(
        steer_namespace(thread_id),
        steer_key(seq),
        steer_payload(
            text,
            seq=seq,
            literal_user_text=literal_user_text,
            turn_id=turn_id,
        ),
        ttl=STEER_ITEM_TTL_MINUTES,
    )
    return seq


async def aread_pending_steers(
    store: object,
    thread_id: str,
    *,
    after_seq: int,
) -> list[SteerItem]:
    """Read a thread's undelivered inbox items in delivery order.

    Reads only the namespace belonging to `thread_id`, so one thread's run can
    never observe another thread's steering messages. Store failures are logged
    and reported as an empty inbox: a steer that cannot be read is delivered on
    a later boundary (or, at worst, after the turn) rather than failing the run.

    Args:
        store: `runtime.store` from the graph server.
        thread_id: Thread whose inbox is being drained.
        after_seq: Watermark; only items with a greater `seq` are returned.

    Returns:
        Validated items ordered by `seq`, capped at `MAX_PENDING_STEERS`.
    """
    if store is None:
        return []
    search = getattr(store, "asearch", None) or getattr(store, "search", None)
    if not callable(search):
        logger.debug("Steering store exposes neither asearch() nor search()")
        return []

    namespace = steer_namespace(thread_id)
    try:
        result = search(namespace, limit=MAX_PENDING_STEERS)
        raw_items = await result if isawaitable(result) else result
    except Exception:
        logger.warning("Could not read the steering inbox", exc_info=True)
        return []

    items: list[SteerItem] = []
    for raw in raw_items or []:
        if tuple(getattr(raw, "namespace", ()) or ()) != namespace:
            # Defensive: a store that widened the prefix must not leak another
            # thread's inbox into this run.
            continue
        item = parse_steer_item(getattr(raw, "key", None), getattr(raw, "value", None))
        if item is None:
            logger.warning("Discarding malformed steering item")
            continue
        if item.seq <= after_seq:
            continue
        items.append(item)
    items.sort(key=lambda item: item.seq)
    return items[:MAX_PENDING_STEERS]


async def adelete_steers(
    store: object,
    thread_id: str,
    keys: Sequence[str],
) -> None:
    """Remove delivered items from a thread's inbox, best effort.

    Deletion only keeps the namespace tidy. Redelivery is prevented by the
    checkpointed watermark, so a failed delete is safe to ignore.

    Args:
        store: `runtime.store` from the graph server.
        thread_id: Thread whose inbox was drained.
        keys: Keys of the items already delivered.
    """
    delete = getattr(store, "adelete", None) or getattr(store, "delete", None)
    if not callable(delete) or not keys:
        return
    namespace = steer_namespace(thread_id)
    for key in keys:
        try:
            result = delete(namespace, key)
            if isawaitable(result):
                await result
        except Exception:
            logger.debug("Could not delete steering item %s", key, exc_info=True)
