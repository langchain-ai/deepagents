"""Unit tests for the QuickJS snapshot patch-chain delta encoding.

Covers the pure encoding helpers in ``langchain_quickjs._snapshot``
(``coerce_record``, ``replay_snapshot_chain``, the snap/patch/clear records)
and the ``CodeInterpreterMiddleware`` policy around them: ``_snapshot_update``
encoding decisions, ``before_agent``/``after_agent`` snapshot roundtrips and
failure handling, and the ``DeltaChannel`` checkpoint-storage behavior through
a real compiled graph.

Also covers HMAC signing of persisted snapshots (advisory AT-5b): the pure
helpers (``normalize_signing_key``, ``sign_snapshot``, ``verify_snapshot``) and
the middleware policy that signs the completed materialized snapshot in
``after_agent`` before it is delta-encoded onto the ``bsdiff`` patch chain, then
verifies it in ``before_agent`` before restore. A snapshot whose signature is
missing, wrong, tampered, or bound to a different thread must be rejected
instead of executed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import bsdiff4
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from langchain_quickjs import CodeInterpreterMiddleware
from langchain_quickjs._snapshot import (
    coerce_record,
    normalize_signing_key,
    replay_snapshot_chain,
    sign_snapshot,
    verify_snapshot,
)

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatResult


def _assert_slot_update(update: dict[str, Any] | None) -> str:
    assert isinstance(update, dict)
    slot_id = update.get("_quickjs_slot_id")
    assert isinstance(slot_id, str)
    return slot_id


def test_after_agent_snapshot_roundtrip_with_before_agent() -> None:
    """Snapshots from ``after_agent`` restore into fresh slots in ``before_agent``.

    ``after_agent`` emits a patch-chain record; the ``DeltaChannel`` reducer
    materializes the chain into full snapshot bytes before ``before_agent``
    reads it. The test runs the reducer explicitly to model that contract.
    """
    mw = CodeInterpreterMiddleware()
    try:
        slot_id = "snapshot-roundtrip"
        state = {"_quickjs_slot_id": slot_id}
        repl = mw._registry.get(slot_id)
        repl.eval_sync("const answer = 42")
        update = mw.after_agent(state=state, runtime=MagicMock())
        assert isinstance(update, dict)
        assert slot_id not in mw._registry._slots

        materialized = replay_snapshot_chain(b"", [update["_quickjs_snapshot_payload"]])
        before_update = mw.before_agent(
            state={"_quickjs_snapshot_payload": materialized},
            runtime=MagicMock(),
        )
        restore_slot_id = _assert_slot_update(before_update)
        restored = mw._registry.get(restore_slot_id)
        assert restored.eval_sync("answer").result == "42"
    finally:
        mw._registry.close()


async def test_aafter_agent_snapshot_roundtrip_with_abefore_agent() -> None:
    """Async snapshot roundtrip restores state in a fresh slot."""
    mw = CodeInterpreterMiddleware()
    try:
        slot_id = "async-snapshot-roundtrip"
        state = {"_quickjs_slot_id": slot_id}
        repl = mw._registry.get(slot_id)
        await repl.eval_async("const answer = 42")
        update = await mw.aafter_agent(state=state, runtime=MagicMock())
        assert isinstance(update, dict)
        assert slot_id not in mw._registry._slots

        materialized = replay_snapshot_chain(b"", [update["_quickjs_snapshot_payload"]])
        before_update = await mw.abefore_agent(
            state={"_quickjs_snapshot_payload": materialized},
            runtime=MagicMock(),
        )
        restore_slot_id = _assert_slot_update(before_update)
        restored = mw._registry.get(restore_slot_id)
        assert restored.eval_sync("answer").result == "42"
    finally:
        mw._registry.close()


def test_before_agent_clears_payload_on_restore_failure() -> None:
    mw = CodeInterpreterMiddleware()
    try:
        update = mw.before_agent(
            state={"_quickjs_snapshot_payload": b"not-a-snapshot"},
            runtime=MagicMock(),
        )

        _assert_slot_update(update)
        assert update["_quickjs_snapshot_payload"] is None
        assert update["_quickjs_snapshot_hmac"] is None
    finally:
        mw._registry.close()


def test_before_agent_ignores_empty_delta_channel_seed() -> None:
    """The `DeltaChannel` seeds a never-written channel to `b""` (its value
    type is `bytes`). `before_agent` must treat that empty seed like a missing
    payload — not attempt to restore it (which would fail "shorter than
    header") and not spuriously clear it."""
    mw = CodeInterpreterMiddleware()
    try:
        update = mw.before_agent(
            state={"_quickjs_snapshot_payload": b""},
            runtime=MagicMock(),
        )
        slot_id = _assert_slot_update(update)
        assert mw._registry.get_if_exists(slot_id) is None
    finally:
        mw._registry.close()


async def test_abefore_agent_ignores_empty_delta_channel_seed() -> None:
    """Async variant: empty `b""` seed is a no-op restore."""
    mw = CodeInterpreterMiddleware()
    try:
        update = await mw.abefore_agent(
            state={"_quickjs_snapshot_payload": b""},
            runtime=MagicMock(),
        )
        slot_id = _assert_slot_update(update)
        assert mw._registry.get_if_exists(slot_id) is None
    finally:
        mw._registry.close()


def test_after_agent_clears_payload_on_snapshot_failure() -> None:
    mw = CodeInterpreterMiddleware()
    try:
        slot_id = "snapshot-failure"
        state = {"_quickjs_slot_id": slot_id}
        repl = mw._registry.get(slot_id)
        with patch.object(repl, "create_snapshot", side_effect=RuntimeError("boom")):
            update = mw.after_agent(state=state, runtime=MagicMock())
        assert update == {
            "_quickjs_snapshot_payload": None,
            "_quickjs_snapshot_hmac": None,
        }
        assert slot_id not in mw._registry._slots
    finally:
        mw._registry.close()


def test_after_agent_drops_payload_above_snapshot_size_cap() -> None:
    mw = CodeInterpreterMiddleware(max_snapshot_bytes=4)
    try:
        slot_id = "snapshot-size-cap"
        state = {"_quickjs_slot_id": slot_id}
        repl = mw._registry.get(slot_id)
        with patch.object(repl, "create_snapshot", return_value=b"12345"):
            update = mw.after_agent(state=state, runtime=MagicMock())
        assert update == {
            "_quickjs_snapshot_payload": None,
            "_quickjs_snapshot_hmac": None,
        }
        assert slot_id not in mw._registry._slots
    finally:
        mw._registry.close()


async def test_aafter_agent_drops_payload_above_snapshot_size_cap() -> None:
    mw = CodeInterpreterMiddleware(max_snapshot_bytes=4)
    try:
        slot_id = "async-snapshot-size-cap"
        state = {"_quickjs_slot_id": slot_id}
        repl = mw._registry.get(slot_id)
        with patch.object(
            repl,
            "acreate_snapshot",
            new=AsyncMock(return_value=b"12345"),
        ):
            update = await mw.aafter_agent(state=state, runtime=MagicMock())
        assert update == {
            "_quickjs_snapshot_payload": None,
            "_quickjs_snapshot_hmac": None,
        }
        assert slot_id not in mw._registry._slots
    finally:
        mw._registry.close()


def _make_snapshots() -> list[bytes]:
    """Three realistic, mostly-stable QuickJS-like snapshots for chain tests."""
    base = bytes(range(256)) * 64  # 16 KiB of stable bytes
    s0 = base
    s1 = bytearray(base)
    s1[100:105] = b"AAAAA"  # tiny mutation
    s2 = bytearray(s1)
    s2[8000:8003] = b"ZZZ"
    return [bytes(s0), bytes(s1), bytes(s2)]


def _build_chain(snapshots: list[bytes]) -> list[tuple[str, bytes]]:
    """Construct the records a sequence of ``after_agent`` calls would emit."""
    records: list[tuple[str, bytes]] = []
    prior = b""
    for snap in snapshots:
        if not prior:
            records.append(("snap", snap))
        else:
            records.append(("patch", bsdiff4.diff(prior, snap)))
        prior = snap
    return records


def test_replay_chain_reconstructs_latest_snapshot() -> None:
    """Folding the full record chain yields the final snapshot bytes."""
    snaps = _make_snapshots()
    chain = _build_chain(snaps)
    assert replay_snapshot_chain(b"", chain) == snaps[-1]


def test_replay_chain_is_associative() -> None:
    """Any batching of the writes materializes to the same value.

    ``DeltaChannel`` may replay writes in arbitrary groupings; the reducer must
    be associative for reconstruction to be deterministic.
    """
    snaps = _make_snapshots()
    chain = _build_chain(snaps)
    whole = replay_snapshot_chain(b"", chain)
    for split in range(len(chain) + 1):
        left = replay_snapshot_chain(b"", chain[:split])
        combined = replay_snapshot_chain(left, chain[split:])
        assert combined == whole


def test_replay_chain_patch_subset_uses_materialized_base() -> None:
    """Replaying patches on top of an already-materialized anchor base works.

    This models reconstruction after a `DeltaChannel` ``snapshot_frequency``
    boundary, where the base is the full prior snapshot (not a fresh seed) and
    only the trailing patch records are replayed.
    """
    snaps = _make_snapshots()
    chain = _build_chain(snaps)
    # Materialize through the first anchor only, then replay the remaining
    # patches on top of that full-bytes base.
    anchor_base = replay_snapshot_chain(b"", chain[:1])
    assert anchor_base == snaps[0]
    result = replay_snapshot_chain(anchor_base, chain[1:])
    assert result == snaps[-1]


def test_replay_chain_clear_resets_base() -> None:
    """A ``clear`` record (None write) drops the running base to empty."""
    snaps = _make_snapshots()
    chain = _build_chain(snaps)
    assert replay_snapshot_chain(b"", [*chain, ("clear", b"")]) == b""
    # A fresh anchor after a clear re-establishes state.
    rebuilt = replay_snapshot_chain(b"", [*chain, ("clear", b""), ("snap", snaps[0])])
    assert rebuilt == snaps[0]


def test_replay_chain_anchor_resets_chain() -> None:
    """A ``snap`` record overrides whatever base preceded it."""
    snaps = _make_snapshots()
    other = b"completely-different-bytes" * 10
    result = replay_snapshot_chain(snaps[2], [("snap", other)])
    assert result == other


def test_coerce_record_accepts_tuple_list_and_none() -> None:
    """The reducer normalizes every record form into a ``(kind, blob)``."""
    assert coerce_record(("patch", b"x")) == ("patch", b"x")
    # The serializer round-trips tuples as lists; both must work.
    assert coerce_record(["snap", b"y"]) == ("snap", b"y")
    assert coerce_record(("snap", bytearray(b"z"))) == ("snap", b"z")
    # None clears the chain.
    assert coerce_record(None) == ("clear", b"")
    # Anything that is not a canonical record is ignored (skipped by reducer).
    assert coerce_record(b"bare-bytes") is None
    assert coerce_record(("only-one",)) is None
    assert coerce_record(("patch", "not-bytes")) is None
    assert coerce_record(42) is None


def test_replay_chain_skips_unrecognized_records() -> None:
    """Unrecognized writes are skipped, not fatal, during replay."""
    snaps = _make_snapshots()
    chain = _build_chain(snaps)
    noisy = [chain[0], 42, ("bogus",), chain[1], None, ("snap", snaps[0])]
    # Folds: snap s0, skip, skip, patch->s1, clear->b"", snap->s0.
    assert replay_snapshot_chain(b"", noisy) == snaps[0]


def test_snapshot_update_first_write_is_anchor() -> None:
    """With no prior, ``_snapshot_update`` emits a full ``snap`` anchor."""
    mw = CodeInterpreterMiddleware()
    try:
        update = mw._snapshot_update(payload=b"hello-world", prior=b"", slot_id="t")
        assert update == {"_quickjs_snapshot_payload": ("snap", b"hello-world")}
    finally:
        mw._registry.close()


def test_snapshot_update_subsequent_write_is_patch() -> None:
    """With a prior snapshot, ``_snapshot_update`` emits a small patch record."""
    mw = CodeInterpreterMiddleware()
    try:
        snaps = _make_snapshots()
        update = mw._snapshot_update(payload=snaps[1], prior=snaps[0], slot_id="t")
        kind, blob = update["_quickjs_snapshot_payload"]
        assert kind == "patch"
        # The patch is dramatically smaller than the full snapshot.
        assert len(blob) < len(snaps[1])
        # And it reconstructs the new snapshot exactly.
        assert bsdiff4.patch(snaps[0], blob) == snaps[1]
    finally:
        mw._registry.close()


def test_snapshot_update_falls_back_to_anchor_when_patch_not_smaller() -> None:
    """If a patch is not smaller than re-anchoring, store the full snapshot."""
    mw = CodeInterpreterMiddleware()
    try:
        # Two unrelated short blobs: the patch carries the whole new payload,
        # so it is not smaller than just re-anchoring.
        prior = b"abcd"
        payload = b"wxyz1234"
        update = mw._snapshot_update(payload=payload, prior=prior, slot_id="t")
        assert update == {"_quickjs_snapshot_payload": ("snap", payload)}
    finally:
        mw._registry.close()


def test_after_agent_emits_patch_against_prior_state() -> None:
    """End-to-end: a second turn with prior state emits a ``patch`` record."""
    mw = CodeInterpreterMiddleware()
    try:
        # Turn 1: establish a snapshot anchor.
        slot_id = "patch-prior"
        state = {"_quickjs_slot_id": slot_id}
        repl = mw._registry.get(slot_id)
        repl.eval_sync("globalThis.x = 1")
        first = mw.after_agent(state=state, runtime=MagicMock())
        prior_full = replay_snapshot_chain(b"", [first["_quickjs_snapshot_payload"]])

        # Turn 2: restore, mutate, snapshot again against the materialized prior.
        before_update = mw.before_agent(
            state={"_quickjs_snapshot_payload": prior_full}, runtime=MagicMock()
        )
        slot_id = _assert_slot_update(before_update)
        repl2 = mw._registry.get(slot_id)
        repl2.eval_sync("globalThis.y = 2")
        second = mw.after_agent(
            state={
                "_quickjs_slot_id": slot_id,
                "_quickjs_snapshot_payload": prior_full,
            },
            runtime=MagicMock(),
        )
        kind, _blob = second["_quickjs_snapshot_payload"]
        assert kind == "patch"

        # The full chain reconstructs a snapshot that restores both globals.
        chain = [
            first["_quickjs_snapshot_payload"],
            second["_quickjs_snapshot_payload"],
        ]
        final = replay_snapshot_chain(b"", chain)
        restore_update = mw.before_agent(
            state={"_quickjs_snapshot_payload": final}, runtime=MagicMock()
        )
        restored = mw._registry.get(_assert_slot_update(restore_update))
        assert restored.eval_sync("x + y").result == "3"
    finally:
        mw._registry.close()


class _GrowingHeapModel(GenericFakeChatModel):
    """Each turn: emit one `eval` that grows the JS heap, then answer.

    The heap stays mostly byte-stable across turns, which is exactly the
    regime where the snapshot patch chain pays off.
    """

    counter: Any = Field(default_factory=lambda: iter(range(1, 10_000)), exclude=True)

    def bind_tools(self, _tools: Any, **_: Any) -> _GrowingHeapModel:
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        **_: Any,
    ) -> ChatResult:
        from langchain_core.outputs import ChatGeneration, ChatResult

        last = messages[-1] if messages else None
        if last is not None and getattr(last, "type", None) == "tool":
            ai = AIMessage(content="done")
        else:
            n = next(self.counter)
            code = (
                f"globalThis.blob_{n} = 'y'.repeat(64); Object.keys(globalThis).length"
            )
            ai = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "eval",
                        "args": {"code": code},
                        "id": f"call_{n}",
                        "type": "tool_call",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=ai)])


def _snapshot_blob_bytes(saver: Any) -> int:
    """Bytes of the snapshot channel stored in the checkpoint *blob* store."""
    total = 0
    for (_, _, channel, _), (_, blob) in saver.blobs.items():
        if channel == "_quickjs_snapshot_payload" and isinstance(
            blob, (bytes, bytearray)
        ):
            total += len(blob)
    return total


def _snapshot_writes_bytes(saver: Any) -> int:
    """Bytes of the snapshot channel stored in the per-step *writes* log."""
    total = 0
    for writes in saver.writes.values():
        for w in writes.values():
            # Each write w is (task_id, channel, (type, blob), path).
            channel = w[1]
            serialized_blob = w[2][1]
            if channel == "_quickjs_snapshot_payload" and isinstance(
                serialized_blob, (bytes, bytearray)
            ):
                total += len(serialized_blob)
    return total


def test_delta_channel_bounds_checkpoint_blob_growth() -> None:
    """Through a real compiled graph the snapshot channel persists only deltas.

    The `DeltaChannel` keeps per-turn deltas in the writes log and never copies
    the full ~MB snapshot into the checkpoint blob store, so blob-store growth
    for the channel is zero. The total persisted bytes across all turns stays a
    small multiple of one snapshot rather than ``turns * snapshot_size``.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    turns = 4
    saver = InMemorySaver()
    agent = create_agent(
        model=_GrowingHeapModel(messages=iter(())),
        tools=[],
        middleware=[CodeInterpreterMiddleware()],
        checkpointer=saver,
    )
    config = {"configurable": {"thread_id": "delta-demo"}}
    for i in range(turns):
        agent.invoke({"messages": [HumanMessage(content=f"turn {i}")]}, config)

    blob_bytes = _snapshot_blob_bytes(saver)
    writes_bytes = _snapshot_writes_bytes(saver)

    # The materialized snapshot is full bytes (reducer coalesced the chain).
    state = agent.get_state(config)
    payload = state.values.get("_quickjs_snapshot_payload")
    assert isinstance(payload, bytes)
    one_snapshot = len(payload)
    assert one_snapshot > 1000  # a real, non-trivial heap snapshot

    # DeltaChannel never writes the channel into the blob store.
    assert blob_bytes == 0
    # Total persisted snapshot bytes stays bounded: one anchor plus small
    # patches, well under what a LastValue channel would store
    # (~turns * one_snapshot). Allow generous headroom for the anchor.
    assert writes_bytes < 3 * one_snapshot
    assert writes_bytes < turns * one_snapshot


def test_delta_channel_resume_from_history_reconstructs_state() -> None:
    """Forking from a mid-history checkpoint reconstructs the heap correctly.

    This exercises the stateless diff-against-prior design: ``after_agent`` on
    the resumed branch diffs against the *materialized* prior snapshot read
    from the forked state, not any in-process cache, so the patch chain stays
    valid across forks and time travel.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    agent = create_agent(
        model=_GrowingHeapModel(messages=iter(())),
        tools=[],
        middleware=[CodeInterpreterMiddleware()],
        checkpointer=saver,
    )
    config = {"configurable": {"thread_id": "fork-demo"}}
    for i in range(4):
        agent.invoke({"messages": [HumanMessage(content=f"turn {i}")]}, config)

    history = list(agent.get_state_history(config))
    assert len(history) > 4
    # Resume from a checkpoint partway back and continue two more turns.
    mid = history[len(history) // 2]
    agent.invoke({"messages": [HumanMessage(content="resumed-1")]}, mid.config)
    final = agent.invoke({"messages": [HumanMessage(content="resumed-2")]}, config)

    # State still materializes to full snapshot bytes after the fork.
    payload = agent.get_state(config).values.get("_quickjs_snapshot_payload")
    assert isinstance(payload, bytes)
    assert len(payload) > 1000
    assert any(getattr(m, "content", None) == "done" for m in final["messages"])


def test_mode_turn_keeps_reset_behavior() -> None:
    mw = CodeInterpreterMiddleware(mode="turn")
    try:
        slot_id = "mode-turn"
        state = {"_quickjs_slot_id": slot_id}
        repl = mw._registry.get(slot_id)
        repl.eval_sync("globalThis.answer = 42")
        update = mw.after_agent(state=state, runtime=MagicMock())
        assert update is None
        assert slot_id not in mw._registry._slots

        before_update = mw.before_agent(
            state={"_quickjs_snapshot_payload": b"ignored"},
            runtime=MagicMock(),
        )
        before_slot_id = _assert_slot_update(before_update)
        assert mw._registry.get_if_exists(before_slot_id) is None
    finally:
        mw._registry.close()


def test_mode_call_ignores_snapshot_payload() -> None:
    mw = CodeInterpreterMiddleware(mode="call")
    try:
        before_update = mw.before_agent(
            state={"_quickjs_snapshot_payload": b"ignored"},
            runtime=MagicMock(),
        )
        before_slot_id = _assert_slot_update(before_update)
        assert mw._registry.get_if_exists(before_slot_id) is None
    finally:
        mw._registry.close()


# Signing keys used by the HMAC tests below. `_SIGNING_KEY` is the "real"
# deployment secret; `_WRONG_SIGNING_KEY` stands in for a key an adversary
# would use to forge a signature (or a key-rotation mismatch).
_SIGNING_KEY = "correct horse battery staple"
_WRONG_SIGNING_KEY = b"a-different-secret"


def _signing_thread_id(mw: CodeInterpreterMiddleware) -> str:
    """The thread id the middleware signs/verifies snapshots under.

    Bound into every HMAC tag, so signing and verifying test helpers must agree
    on it. Mirrors the middleware's own `_resolve_thread_id` fallback.
    """
    return "signed-snapshot-slot"


def test_normalize_signing_key_encodes_str_and_passes_bytes() -> None:
    assert normalize_signing_key("abc") == b"abc"
    assert normalize_signing_key(b"abc") == b"abc"
    assert normalize_signing_key(bytearray(b"abc")) == b"abc"


def test_normalize_signing_key_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        normalize_signing_key("")
    with pytest.raises(ValueError, match="non-empty"):
        normalize_signing_key(b"")


def test_sign_verify_roundtrip() -> None:
    key = normalize_signing_key(_SIGNING_KEY)
    tag = sign_snapshot(key, b"payload", "thread-1")
    assert verify_snapshot(key, b"payload", "thread-1", tag) is True


def test_verify_rejects_missing_tag() -> None:
    key = normalize_signing_key(_SIGNING_KEY)
    assert verify_snapshot(key, b"payload", "thread-1", None) is False
    assert verify_snapshot(key, b"payload", "thread-1", b"") is False


def test_verify_rejects_tampered_payload() -> None:
    key = normalize_signing_key(_SIGNING_KEY)
    tag = sign_snapshot(key, b"payload", "thread-1")
    assert verify_snapshot(key, b"payload-EVIL", "thread-1", tag) is False


def test_verify_rejects_wrong_key() -> None:
    good = normalize_signing_key(_SIGNING_KEY)
    bad = normalize_signing_key(_WRONG_SIGNING_KEY)
    tag = sign_snapshot(good, b"payload", "thread-1")
    assert verify_snapshot(bad, b"payload", "thread-1", tag) is False


def test_verify_rejects_cross_thread_replay() -> None:
    """A tag signed for one thread must not authenticate another thread.

    This is the state-store adversary who copies a legitimately-signed snapshot
    from thread A into thread B's slot in the checkpointer.
    """
    key = normalize_signing_key(_SIGNING_KEY)
    tag = sign_snapshot(key, b"payload", "thread-A")
    assert verify_snapshot(key, b"payload", "thread-B", tag) is False


def test_thread_id_framing_is_unambiguous() -> None:
    """Length-prefixed framing prevents boundary-shift collisions.

    Without length-prefixing, ``("ab", "cX")`` and ``("abc", "X")`` could
    serialize to the same bytes. The tags must differ.
    """
    key = normalize_signing_key(_SIGNING_KEY)
    tag1 = sign_snapshot(key, b"cX-payload", "ab")
    tag2 = sign_snapshot(key, b"X-payload", "abc")
    assert tag1 != tag2


def test_empty_key_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CodeInterpreterMiddleware(snapshot_signing_key="")


def test_after_agent_emits_hmac_when_key_set() -> None:
    mw = CodeInterpreterMiddleware(snapshot_signing_key=_SIGNING_KEY)
    try:
        repl = mw._registry.get(_signing_thread_id(mw))
        repl.eval_sync("const answer = 42")
        update = mw.after_agent(
            state={"_quickjs_slot_id": _signing_thread_id(mw)}, runtime=MagicMock()
        )
        assert isinstance(update, dict)
        tag = update["_quickjs_snapshot_hmac"]
        assert isinstance(tag, bytes)
        assert len(tag) == 32  # SHA-256 digest

        # The tag authenticates the *materialized* snapshot the chain replays to.
        materialized = replay_snapshot_chain(b"", [update["_quickjs_snapshot_payload"]])
        assert verify_snapshot(
            mw._snapshot_signing_key, materialized, _signing_thread_id(mw), tag
        )
    finally:
        mw._registry.close()


def test_after_agent_no_hmac_when_key_unset() -> None:
    mw = CodeInterpreterMiddleware()  # thread mode, no key -> no signing
    try:
        repl = mw._registry.get(_signing_thread_id(mw))
        repl.eval_sync("const answer = 42")
        update = mw.after_agent(
            state={"_quickjs_slot_id": _signing_thread_id(mw)}, runtime=MagicMock()
        )
        assert isinstance(update, dict)
        assert "_quickjs_snapshot_hmac" not in update
    finally:
        mw._registry.close()


def test_signed_snapshot_roundtrip_restores() -> None:
    """A snapshot signed in after_agent restores cleanly in before_agent."""
    mw = CodeInterpreterMiddleware(snapshot_signing_key=_SIGNING_KEY)
    try:
        repl = mw._registry.get(_signing_thread_id(mw))
        repl.eval_sync("const answer = 42")
        update = mw.after_agent(
            state={"_quickjs_slot_id": _signing_thread_id(mw)}, runtime=MagicMock()
        )
        materialized = replay_snapshot_chain(b"", [update["_quickjs_snapshot_payload"]])

        before_update = mw.before_agent(
            state={
                "_quickjs_slot_id": _signing_thread_id(mw),
                "_quickjs_snapshot_payload": materialized,
                "_quickjs_snapshot_hmac": update["_quickjs_snapshot_hmac"],
            },
            runtime=MagicMock(),
        )
        assert before_update is None  # accepted, restored in place
        restored = mw._registry.get(_signing_thread_id(mw))
        assert restored.eval_sync("answer").result == "42"
    finally:
        mw._registry.close()


def test_before_agent_rejects_missing_hmac() -> None:
    """Key configured but snapshot carries no tag -> rejected, not restored."""
    mw = CodeInterpreterMiddleware(snapshot_signing_key=_SIGNING_KEY)
    try:
        repl = mw._registry.get(_signing_thread_id(mw))
        repl.eval_sync("const answer = 42")
        update = mw.after_agent(
            state={"_quickjs_slot_id": _signing_thread_id(mw)}, runtime=MagicMock()
        )
        materialized = replay_snapshot_chain(b"", [update["_quickjs_snapshot_payload"]])

        repl = mw._registry.get(_signing_thread_id(mw))
        with patch.object(repl, "restore_snapshot") as restore:
            before_update = mw.before_agent(
                state={
                    "_quickjs_slot_id": _signing_thread_id(mw),
                    "_quickjs_snapshot_payload": materialized,
                },  # no hmac
                runtime=MagicMock(),
            )
            restore.assert_not_called()
        assert before_update == {
            "_quickjs_snapshot_payload": None,
            "_quickjs_snapshot_hmac": None,
        }
    finally:
        mw._registry.close()


def test_before_agent_rejects_tampered_payload() -> None:
    """Attacker mutates the stored snapshot bytes; the tag no longer matches."""
    mw = CodeInterpreterMiddleware(snapshot_signing_key=_SIGNING_KEY)
    try:
        repl = mw._registry.get(_signing_thread_id(mw))
        repl.eval_sync("const answer = 42")
        update = mw.after_agent(
            state={"_quickjs_slot_id": _signing_thread_id(mw)}, runtime=MagicMock()
        )
        materialized = bytearray(
            replay_snapshot_chain(b"", [update["_quickjs_snapshot_payload"]])
        )
        materialized[-1] ^= 0xFF  # flip a byte

        before_update = mw.before_agent(
            state={
                "_quickjs_snapshot_payload": bytes(materialized),
                "_quickjs_snapshot_hmac": update["_quickjs_snapshot_hmac"],
            },
            runtime=MagicMock(),
        )
        _assert_slot_update(before_update)
        assert before_update["_quickjs_snapshot_payload"] is None
        assert before_update["_quickjs_snapshot_hmac"] is None
    finally:
        mw._registry.close()


def test_before_agent_rejects_wrong_key_signature() -> None:
    """Snapshot signed under a different key (forged by an adversary)."""
    signer = CodeInterpreterMiddleware(snapshot_signing_key=_WRONG_SIGNING_KEY)
    verifier = CodeInterpreterMiddleware(snapshot_signing_key=_SIGNING_KEY)
    try:
        repl = signer._registry.get(_signing_thread_id(signer))
        repl.eval_sync("const answer = 42")
        update = signer.after_agent(
            state={"_quickjs_slot_id": _signing_thread_id(signer)}, runtime=MagicMock()
        )
        materialized = replay_snapshot_chain(b"", [update["_quickjs_snapshot_payload"]])

        before_update = verifier.before_agent(
            state={
                "_quickjs_slot_id": _signing_thread_id(verifier),
                "_quickjs_snapshot_payload": materialized,
                "_quickjs_snapshot_hmac": update["_quickjs_snapshot_hmac"],
            },
            runtime=MagicMock(),
        )
        assert before_update == {
            "_quickjs_snapshot_payload": None,
            "_quickjs_snapshot_hmac": None,
        }
    finally:
        signer._registry.close()
        verifier._registry.close()


async def test_abefore_agent_rejects_tampered_payload() -> None:
    """Async restore path enforces the same rejection."""
    mw = CodeInterpreterMiddleware(snapshot_signing_key=_SIGNING_KEY)
    try:
        repl = mw._registry.get(_signing_thread_id(mw))
        await repl.eval_async("const answer = 42")
        update = await mw.aafter_agent(
            state={"_quickjs_slot_id": _signing_thread_id(mw)}, runtime=MagicMock()
        )
        materialized = bytearray(
            replay_snapshot_chain(b"", [update["_quickjs_snapshot_payload"]])
        )
        materialized[0] ^= 0xFF

        before_update = await mw.abefore_agent(
            state={
                "_quickjs_snapshot_payload": bytes(materialized),
                "_quickjs_snapshot_hmac": update["_quickjs_snapshot_hmac"],
            },
            runtime=MagicMock(),
        )
        _assert_slot_update(before_update)
        assert before_update["_quickjs_snapshot_payload"] is None
        assert before_update["_quickjs_snapshot_hmac"] is None
    finally:
        mw._registry.close()


async def test_asigned_snapshot_roundtrip_restores() -> None:
    mw = CodeInterpreterMiddleware(snapshot_signing_key=_SIGNING_KEY)
    try:
        repl = mw._registry.get(_signing_thread_id(mw))
        await repl.eval_async("const answer = 42")
        update = await mw.aafter_agent(
            state={"_quickjs_slot_id": _signing_thread_id(mw)}, runtime=MagicMock()
        )
        materialized = replay_snapshot_chain(b"", [update["_quickjs_snapshot_payload"]])

        before_update = await mw.abefore_agent(
            state={
                "_quickjs_slot_id": _signing_thread_id(mw),
                "_quickjs_snapshot_payload": materialized,
                "_quickjs_snapshot_hmac": update["_quickjs_snapshot_hmac"],
            },
            runtime=MagicMock(),
        )
        assert before_update is None
        restored = mw._registry.get(_signing_thread_id(mw))
        assert restored.eval_sync("answer").result == "42"
    finally:
        mw._registry.close()


def test_no_key_restores_without_verification() -> None:
    """Legacy behavior: with no key, an unsigned snapshot still restores.

    This preserves backward compatibility for trusted-store deployments, where
    the caller opts out of integrity verification by not configuring a key.
    """
    mw = CodeInterpreterMiddleware()  # no key
    try:
        repl = mw._registry.get(_signing_thread_id(mw))
        repl.eval_sync("const answer = 42")
        update = mw.after_agent(
            state={"_quickjs_slot_id": _signing_thread_id(mw)}, runtime=MagicMock()
        )
        materialized = replay_snapshot_chain(b"", [update["_quickjs_snapshot_payload"]])
        before_update = mw.before_agent(
            state={
                "_quickjs_slot_id": _signing_thread_id(mw),
                "_quickjs_snapshot_payload": materialized,
            },  # no hmac
            runtime=MagicMock(),
        )
        assert before_update is None
        restored = mw._registry.get(_signing_thread_id(mw))
        assert restored.eval_sync("answer").result == "42"
    finally:
        mw._registry.close()


def test_snapshot_size_cap_clears_hmac() -> None:
    """A dropped oversized snapshot must also clear any stale signature."""
    mw = CodeInterpreterMiddleware(
        snapshot_signing_key=_SIGNING_KEY, max_snapshot_bytes=4
    )
    try:
        repl = mw._registry.get(_signing_thread_id(mw))
        with patch.object(repl, "create_snapshot", return_value=b"12345"):
            update = mw.after_agent(
                state={"_quickjs_slot_id": _signing_thread_id(mw)}, runtime=MagicMock()
            )
        assert update == {
            "_quickjs_snapshot_payload": None,
            "_quickjs_snapshot_hmac": None,
        }
    finally:
        mw._registry.close()
