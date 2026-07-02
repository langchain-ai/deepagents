"""Patch-chain delta encoding for the QuickJS REPL heap snapshot.

The QuickJS snapshot is a full serialization of the REPL heap, rewritten in its
entirety on every turn. Persisting it through a plain ``LastValue`` channel
copies the whole payload (~1.4 MB in practice) into every checkpoint, so
checkpoint storage grows linearly with thread length.

Empirically the heap is ~98-100% byte-stable between consecutive turns, so a
binary diff (``bsdiff4``) between successive snapshots is tiny (~200 B-1 KB vs
1.4 MB, a ~1000x reduction). We therefore store a *patch chain* on a
``DeltaChannel``: each turn writes one record describing the delta from the
previous snapshot, and the channel's bulk reducer (:func:`replay_snapshot_chain`)
replays the chain back into the full snapshot bytes on reconstruction.

A write is one of these records, a plain ``(kind, blob)`` 2-tuple of primitives
(the serializer round-trips it as a list, so the reducer accepts either). The
kind is a bare string so the record serializes through msgpack with no
custom/unregistered type:

    ("snap",  full_snapshot_bytes)  -- anchor; ignores the running base
    ("patch", bsdiff4_patch_bytes)  -- delta applied to the running base
    ("clear", b"")                  -- reset the running base to empty
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import bsdiff4

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

SNAP = "snap"
PATCH = "patch"
CLEAR = "clear"

SnapshotRecord = tuple[str, bytes]


def coerce_record(write: object) -> tuple[str, bytes] | None:
    """Normalize a single channel write into a ``(kind, blob)`` record.

    Accepts the canonical record forms — a ``(kind, blob)`` tuple, or the list
    the serializer round-trips it as — and ``None``, which clears the chain.
    Anything else returns ``None`` and is skipped by the reducer.
    """
    if write is None:
        return (CLEAR, b"")
    if isinstance(write, (tuple, list)) and len(write) == 2:  # noqa: PLR2004
        kind, blob = write
        if isinstance(kind, str) and isinstance(blob, (bytes, bytearray)):
            return (kind, bytes(blob))
    return None


def replay_snapshot_chain(
    state: bytes | None,
    writes: Sequence[object],
) -> bytes:
    """Bulk ``DeltaChannel`` reducer that replays a snapshot patch chain.

    ``state`` is the fully materialized snapshot bytes reconstructed so far
    (``b""`` for an empty channel); ``writes`` is the ordered sequence of
    records to fold in. Returns the new materialized full snapshot bytes.

    Folding is left-to-right and deterministic:
      * ``("snap", blob)``  -> base becomes ``blob`` (anchor; prior base ignored)
      * ``("patch", blob)`` -> base becomes ``bsdiff4.patch(base, blob)``
      * ``("clear", _)``    -> base becomes ``b""``

    This is associative as ``DeltaChannel`` requires — re-batching the writes
    yields the same value, since folding ``[xs, ys]`` onto ``state`` equals
    folding ``[ys]`` onto the result of folding ``[xs]``. It is pure (no I/O,
    randomness, or clock reads), so it is safe to re-run on every reconstruction
    or time-travel replay.
    """
    base = state if isinstance(state, (bytes, bytearray)) else b""
    base = bytes(base)
    for write in writes:
        record = coerce_record(write)
        if record is None:
            continue
        kind, blob = record
        if kind == SNAP:
            base = blob
        elif kind == PATCH:
            base = bsdiff4.patch(base, blob)
        elif kind == CLEAR:
            base = b""
    return base


def encode_snapshot(payload: bytes, prior: bytes) -> SnapshotRecord:
    """Encode a fresh snapshot ``payload`` as a patch-chain record.

    ``prior`` is the previous turn's fully materialized snapshot bytes (``b""``
    when none exists — e.g. first turn, a fork from before snapshots, or a fresh
    process). The delta is computed statelessly against ``prior``, so no
    in-process cache is needed and the result is correct across forks, restores,
    and time travel.

    Returns:
      * ``("snap", payload)`` when there is no usable prior, or when the bsdiff
        patch would not be smaller than a fresh anchor;
      * ``("patch", diff)`` otherwise.
    """
    if not prior:
        return (SNAP, payload)
    try:
        patch = bsdiff4.diff(prior, payload)
    except Exception:  # noqa: BLE001  # never let diffing break the turn
        logger.warning(
            "Failed to diff QuickJS snapshot; storing full anchor",
            exc_info=True,
        )
        return (SNAP, payload)
    # A patch only pays off when it is smaller than re-anchoring; otherwise
    # store the full snapshot so the chain stays compact and self-healing.
    if len(patch) >= len(payload):
        return (SNAP, payload)
    return (PATCH, patch)
