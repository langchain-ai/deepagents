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
    ("pages", changed_pages_bytes)  -- delta applied to the running base
    ("clear", b"")                  -- reset the running base to empty
"""

from __future__ import annotations

import hmac
import logging
import struct
import zlib
from hashlib import sha256
from typing import TYPE_CHECKING

import bsdiff4

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

SNAP = "snap"
PATCH = "patch"
PAGES = "pages"
CLEAR = "clear"

SnapshotRecord = tuple[str, bytes]

_PAGE_SIZE = 256
"""Diff granularity: a one-field edit changes about a dozen pages of this size."""

_RUN = struct.Struct("<QQ")
"""Header of one run of changed bytes in a `pages` record: offset, length."""

_PAGES_VERSION = 1
"""First byte of a `pages` record, so a reader refuses a layout it does not know."""

# Domain-separation prefix folded into every signed message so a snapshot HMAC
# can never be confused with an HMAC computed over some other blob using the
# same key. Bump the version suffix if the signed-message layout ever changes.
_HMAC_DOMAIN = b"langchain-quickjs/snapshot-hmac/v1"


def normalize_signing_key(key: str | bytes) -> bytes:
    """Coerce a user-supplied signing key into raw ``bytes``.

    ``str`` keys are UTF-8 encoded; ``bytes`` are used verbatim. Empty keys are
    rejected because an empty HMAC key provides no integrity guarantee.
    """
    material = key.encode("utf-8") if isinstance(key, str) else bytes(key)
    if not material:
        msg = "`snapshot_signing_key` must be a non-empty str or bytes."
        raise ValueError(msg)
    return material


def sign_snapshot(key: bytes, payload: bytes, thread_id: str) -> bytes:
    """Return the HMAC-SHA256 tag over a fully materialized snapshot.

    The tag is computed over the *completed materialized* snapshot bytes (the
    full heap serialization) bound to ``thread_id``, so a valid snapshot for one
    thread cannot be replayed into another by a state-store adversary. This is
    signed before the payload is delta-encoded (``encode_snapshot``) and flushed
    onto the ``bsdiff`` patch chain; verification recomputes the tag over the
    materialized bytes the chain replays back to.
    """
    return hmac.new(key, _signed_message(payload, thread_id), sha256).digest()


def verify_snapshot(
    key: bytes, payload: bytes, thread_id: str, tag: bytes | None
) -> bool:
    """Constant-time check that ``tag`` authenticates ``payload`` for ``thread_id``.

    Returns ``False`` for a missing/short tag or any mismatch. The comparison uses
    :func:`hmac.compare_digest` to avoid leaking timing information about how much
    of the tag matched.
    """
    if not tag:
        return False
    expected = sign_snapshot(key, payload, thread_id)
    return hmac.compare_digest(expected, bytes(tag))


def _signed_message(payload: bytes, thread_id: str) -> bytes:
    """Build the length-prefixed message HMAC is computed over.

    Framing (domain, then a length-prefixed ``thread_id``, then the payload)
    makes no two distinct ``(thread_id, payload)`` pairs can serialize to the
    same byte string, so an attacker cannot shift bytes across the boundary
    to forge a collision.
    """
    tid = thread_id.encode("utf-8")
    return b"".join((_HMAC_DOMAIN, len(tid).to_bytes(8, "big"), tid, bytes(payload)))


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
      * ``("pages", blob)`` -> base with the changed pages in `blob` written over it
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
        elif kind == PAGES:
            base = _apply_pages(base, blob)
        elif kind == CLEAR:
            base = b""
    return base


def _diff_pages(prior: bytes, payload: bytes) -> bytes:
    """Encode the pages of `payload` that differ from `prior`.

    QuickJS allocates in place and never moves live objects, so a turn's
    changes land in a few pages. Comparing aligned pages finds them in linear
    time, where `bsdiff4.diff` suffix-sorts the whole snapshot.
    """
    parts = [bytes([_PAGES_VERSION]), len(payload).to_bytes(8, "little")]
    start = None
    for offset in range(0, len(payload), _PAGE_SIZE):
        end = offset + _PAGE_SIZE
        if prior[offset:end] != payload[offset:end]:
            if start is None:
                start = offset
        elif start is not None:
            parts += (_RUN.pack(start, offset - start), payload[start:offset])
            start = None
    if start is not None:
        parts += (_RUN.pack(start, len(payload) - start), payload[start:])
    return zlib.compress(b"".join(parts), 1)


def _apply_pages(base: bytes, blob: bytes) -> bytes:
    """Rebuild a snapshot from `base` and a record made by `_diff_pages`."""
    data = zlib.decompress(blob)
    if data[:1] != bytes([_PAGES_VERSION]):
        msg = f"Unsupported `pages` snapshot record version {data[:1].hex() or 'none'}"
        raise ValueError(msg)
    size = int.from_bytes(data[1:9], "little")
    out = bytearray(base[:size].ljust(size, b"\0"))
    pos = 9
    while pos < len(data):
        offset, length = _RUN.unpack_from(data, pos)
        pos += _RUN.size
        out[offset : offset + length] = data[pos : pos + length]
        pos += length
    return bytes(out)


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
