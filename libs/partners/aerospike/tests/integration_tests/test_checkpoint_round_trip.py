"""Integration tests for AerospikeSaver checkpoint round-trip and factory compatibility.

Requires aerospike-server:dev running at AEROSPIKE_HOST:AEROSPIKE_PORT.
All tests skip gracefully if the server is unreachable (see tests/conftest.py session fixture).
"""

from __future__ import annotations

import contextlib
import os
import uuid

import pytest
from aerospike_sdk import DataSet
from langgraph.checkpoint.aerospike import AerospikeSaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from deepagents_aerospike import create_aerospike_checkpointer


@pytest.fixture
def checkpointer(session, aerospike_namespace):
    """Yield a fresh AerospikeSaver with unique sets and clean them up after each test.

    Using unique set names per test prevents parallel workers from truncating each
    other's state. Sets are truncated both before and after the test.
    """
    suffix = str(uuid.uuid4()).replace("-", "")[:8]
    set_cp = f"lg_cp_{suffix}"
    set_writes = f"lg_cp_w_{suffix}"
    set_meta = f"lg_cp_meta_{suffix}"
    sets = (set_cp, set_writes, set_meta)

    saver = AerospikeSaver(
        session=session,
        namespace=aerospike_namespace,
        set_cp=set_cp,
        set_writes=set_writes,
        set_meta=set_meta,
    )

    def _truncate() -> None:
        for set_name in sets:
            dataset = DataSet.of(aerospike_namespace, set_name)
            with contextlib.suppress(Exception):
                session.truncate(dataset, before_nanos=0)

    _truncate()
    try:
        yield saver
    finally:
        _truncate()


def test_checkpoint_write_read_round_trip(checkpointer):
    """A checkpoint written via put() is retrievable via get_tuple()."""
    config = {"configurable": {"thread_id": "test-round-trip-1"}}
    checkpoint = {
        "v": 1,
        "id": "cp-test-1",
        "ts": "2026-01-01T00:00:00+00:00",
        "channel_values": {"__root__": {"role": "user", "content": "hello"}},
        "channel_versions": {"__root__": 1},
        "versions_seen": {},
        "pending_sends": [],
    }
    metadata = {
        "source": "test",
        "step": 1,
        "writes": {},
        "parents": {},
    }
    new_versions = {"__root__": 1}

    checkpointer.put(config, checkpoint, metadata, new_versions)
    result = checkpointer.get_tuple(config)

    assert result is not None
    assert result.checkpoint["channel_values"] == checkpoint["channel_values"]


def test_get_tuple_returns_none_for_missing_thread(checkpointer):
    """get_tuple() returns None for a thread_id that has never been written."""
    config = {"configurable": {"thread_id": "nonexistent-thread-999"}}
    result = checkpointer.get_tuple(config)
    assert result is None


@pytest.mark.integration
def test_factory_checkpointer_is_basecheckpointsaver(session):
    """create_aerospike_checkpointer returns a BaseCheckpointSaver-compatible instance."""
    host = os.getenv("AEROSPIKE_HOST", "localhost")
    port = int(os.getenv("AEROSPIKE_PORT", "3000"))
    namespace = os.getenv("AEROSPIKE_NAMESPACE", "test")

    checkpointer = create_aerospike_checkpointer(
        host=host,
        port=port,
        namespace=namespace,
    )

    assert isinstance(checkpointer, BaseCheckpointSaver)


def test_idempotent_index_creation(session, aerospike_namespace):
    """Creating two AerospikeSaver instances on the same namespace does not fail."""
    saver1 = AerospikeSaver(session=session, namespace=aerospike_namespace)
    saver2 = AerospikeSaver(session=session, namespace=aerospike_namespace)

    assert saver1 is not None
    assert saver2 is not None


def test_put_writes_round_trip(checkpointer):
    """Pending writes persisted via put_writes() appear in get_tuple()."""
    config = {"configurable": {"thread_id": "test-writes-1"}}
    checkpoint = {
        "v": 1,
        "id": "cp-writes-1",
        "ts": "2026-01-01T00:00:00+00:00",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    metadata = {"source": "test", "step": 1, "writes": {}, "parents": {}}

    config = checkpointer.put(config, checkpoint, metadata, {})
    checkpointer.put_writes(
        config,
        [("channel1", "value1"), ("channel2", "value2")],
        task_id="task-1",
        task_path="root",
    )

    result = checkpointer.get_tuple(config)
    assert result is not None
    assert len(result.pending_writes) == 2
    channels = {w[1] for w in result.pending_writes}
    assert channels == {"channel1", "channel2"}


def test_list_checkpoints_with_limit(checkpointer):
    """list() returns checkpoints in reverse chronological order, limited by ``limit``."""
    config = {"configurable": {"thread_id": "test-list-1"}}
    metadata = {"source": "test", "step": 1, "writes": {}, "parents": {}}
    ids = []
    for i in range(3):
        cp_id = f"cp-list-{i}"
        ids.append(cp_id)
        checkpoint = {
            "v": 1,
            "id": cp_id,
            "ts": f"2026-01-0{i+1}T00:00:00+00:00",
            "channel_values": {},
            "channel_versions": {},
            "versions_seen": {},
            "pending_sends": [],
        }
        config = checkpointer.put(config, checkpoint, metadata, {})

    listed = list(checkpointer.list({"configurable": {"thread_id": "test-list-1"}}, limit=2))
    assert len(listed) == 2
    # Most recent first.
    assert listed[0].checkpoint["id"] == ids[-1]
    assert listed[1].checkpoint["id"] == ids[-2]


def test_delete_thread_removes_history(checkpointer):
    """delete_thread() removes the checkpoint, writes, and latest pointer."""
    thread_id = "test-delete-1"
    config = {"configurable": {"thread_id": thread_id}}
    checkpoint = {
        "v": 1,
        "id": "cp-delete-1",
        "ts": "2026-01-01T00:00:00+00:00",
        "channel_values": {},
        "channel_versions": {},
        "versions_seen": {},
        "pending_sends": [],
    }
    metadata = {"source": "test", "step": 1, "writes": {}, "parents": {}}

    config = checkpointer.put(config, checkpoint, metadata, {})
    checkpointer.put_writes(config, [("channel1", "value1")], task_id="task-1")

    assert checkpointer.get_tuple(config) is not None

    checkpointer.delete_thread(thread_id)

    assert checkpointer.get_tuple(config) is None
    assert list(checkpointer.list(config)) == []


def test_factory_checkpointer_can_close():
    """``create_aerospike_checkpointer`` returns a saver whose ``close()`` is safe."""
    host = os.getenv("AEROSPIKE_HOST", "localhost")
    port = int(os.getenv("AEROSPIKE_PORT", "3000"))
    namespace = os.getenv("AEROSPIKE_NAMESPACE", "test")

    checkpointer = create_aerospike_checkpointer(
        host=host,
        port=port,
        namespace=namespace,
    )

    checkpointer.close()

    assert isinstance(checkpointer, BaseCheckpointSaver)
