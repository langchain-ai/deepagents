"""Tests for the built-in Deep Agents thread inspector script."""

import importlib.util
import json
import sqlite3
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "deepagents_code"
    / "built_in_skills"
    / "deepagents-thread-inspector"
    / "scripts"
    / "inspect_sessions.py"
)


@pytest.fixture
def inspector(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the standalone script without requiring it to be importable as a module."""
    monkeypatch.delenv("LANGGRAPH_STRICT_MSGPACK", raising=False)
    spec = importlib.util.spec_from_file_location("inspect_sessions", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _create_database(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE checkpoints (
            thread_id TEXT NOT NULL,
            checkpoint_id TEXT NOT NULL,
            metadata TEXT
        );
        CREATE TABLE writes (
            thread_id TEXT NOT NULL,
            checkpoint_ns TEXT NOT NULL,
            checkpoint_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            idx INTEGER NOT NULL,
            channel TEXT NOT NULL,
            type TEXT,
            value BLOB
        );
        """
    )
    return conn


def test_connects_read_only_and_resolves_thread_prefix(
    inspector: ModuleType, tmp_path: Path
) -> None:
    db = tmp_path / "sessions.db"
    writable = _create_database(db)
    writable.executemany(
        "INSERT INTO checkpoints VALUES (?, ?, ?)",
        [
            ("thread-123", "001", "{}"),
            ("thread-456", "002", "{}"),
        ],
    )
    writable.commit()
    writable.close()

    conn = inspector._connect_read_only(db)
    try:
        assert inspector._resolve_thread_id(conn, "thread-1") == "thread-123"
        with pytest.raises(SystemExit, match="ambiguous"):
            inspector._resolve_thread_id(conn, "thread-")
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute(
                "INSERT INTO checkpoints VALUES (?, ?, ?)",
                ("new-thread", "003", "{}"),
            )
    finally:
        conn.close()


def test_reconstructs_messages_and_latest_turn(
    inspector: ModuleType, tmp_path: Path
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    db = tmp_path / "sessions.db"
    thread_id = "thread-123"
    writable = _create_database(db)
    serde = JsonPlusSerializer()
    writes = [
        [HumanMessage(content="first", id="user-1")],
        [AIMessage(content="answer", id="assistant-1")],
        [HumanMessage(content="second", id="user-2")],
        [AIMessage(content="done", id="assistant-2")],
    ]
    for index, messages in enumerate(writes, start=1):
        checkpoint_id = f"{index:03d}"
        metadata = json.dumps(
            {
                "updated_at": f"2026-01-01T00:00:0{index}Z",
                "turn_number": 2,
                "turn_id": "turn-2",
            }
        )
        writable.execute(
            "INSERT INTO checkpoints VALUES (?, ?, ?)",
            (thread_id, checkpoint_id, metadata),
        )
        type_name, value = serde.dumps_typed(messages)
        writable.execute(
            "INSERT INTO writes VALUES (?, '', ?, 'task', 0, 'messages', ?, ?)",
            (thread_id, checkpoint_id, type_name, value),
        )
    writable.commit()
    writable.close()

    conn = inspector._connect_read_only(db)
    try:
        messages = inspector._reconstruct_messages(conn, thread_id)
        turns, preamble = inspector._turns(messages, 100)
    finally:
        conn.close()

    assert preamble == []
    assert len(turns) == 2
    assert [message["content"] for message in turns[-1]["messages"]] == [
        "second",
        "done",
    ]


def test_message_record_hides_reasoning_and_marks_truncation(
    inspector: ModuleType,
) -> None:
    record = inspector._message_record(
        0,
        {
            "role": "assistant",
            "content": [
                {"type": "reasoning", "text": "hidden"},
                {"type": "text", "text": "visible"},
            ],
            "tool_calls": [
                {"name": "example", "id": "call-1", "args": {"value": "long"}}
            ],
        },
        4,
    )

    assert record["content"] == "visi…"
    assert record["content_truncated"] is True
    assert record["tool_calls"][0]["args_truncated"] is True
