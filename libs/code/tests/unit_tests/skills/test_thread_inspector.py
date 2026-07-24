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
            checkpoint_ns TEXT NOT NULL,
            checkpoint_id TEXT NOT NULL,
            metadata TEXT,
            type TEXT,
            checkpoint BLOB
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


def test_runtime_probe_requires_all_langgraph_modules(
    inspector: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = inspector.importlib.import_module

    def import_module(name: str) -> ModuleType:
        if name == "langgraph.graph.message":
            msg = "No module named 'langgraph.graph.message'"
            raise ModuleNotFoundError(msg)
        return real_import(name)

    monkeypatch.setattr(inspector.importlib, "import_module", import_module)

    assert inspector._has_runtime() is False


def test_connects_read_only_and_resolves_thread_prefix(
    inspector: ModuleType, tmp_path: Path
) -> None:
    db = tmp_path / "sessions.db"
    writable = _create_database(db)
    writable.executemany(
        "INSERT INTO checkpoints "
        "(thread_id, checkpoint_ns, checkpoint_id, metadata) VALUES (?, '', ?, ?)",
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
                "INSERT INTO checkpoints "
                "(thread_id, checkpoint_ns, checkpoint_id, metadata) "
                "VALUES (?, '', ?, ?)",
                ("new-thread", "003", "{}"),
            )
    finally:
        conn.close()


def test_thread_queries_exclude_subagent_namespaces(
    inspector: ModuleType, tmp_path: Path
) -> None:
    db = tmp_path / "sessions.db"
    writable = _create_database(db)
    root_metadata = json.dumps(
        {
            "updated_at": "2026-01-01T00:00:01Z",
            "agent_name": "root-agent",
            "git_branch": "root-branch",
            "cwd": "/root",
            "turn_number": 1,
            "turn_id": "root-turn",
        }
    )
    subagent_metadata = json.dumps(
        {
            "updated_at": "2026-01-01T00:00:02Z",
            "agent_name": "subagent",
            "git_branch": "subagent-branch",
            "cwd": "/subagent",
            "turn_number": 99,
            "turn_id": "subagent-turn",
        }
    )
    writable.executemany(
        "INSERT INTO checkpoints "
        "(thread_id, checkpoint_ns, checkpoint_id, metadata) VALUES (?, ?, ?, ?)",
        [
            ("thread-123", "", "001", root_metadata),
            ("thread-123", "subagent:abc", "999", subagent_metadata),
            ("subagent-only", "subagent:def", "999", subagent_metadata),
        ],
    )
    writable.executemany(
        "INSERT INTO writes VALUES (?, ?, ?, 'task', 0, 'messages', NULL, NULL)",
        [
            ("thread-123", "", "001"),
            ("thread-123", "subagent:abc", "999"),
        ],
    )
    writable.commit()
    writable.close()

    conn = inspector._connect_read_only(db)
    try:
        summary, metadata = inspector._thread_summary(conn, "thread-123")
        threads = inspector._list_threads(conn, 10)
        with pytest.raises(SystemExit, match="Thread not found"):
            inspector._resolve_thread_id(conn, "subagent-only")
    finally:
        conn.close()

    assert metadata["turn_number"] == 1
    assert metadata["turn_id"] == "root-turn"
    assert summary["agent_name"] == "root-agent"
    assert summary["created_at"] == "2026-01-01T00:00:01Z"
    assert summary["updated_at"] == "2026-01-01T00:00:01Z"
    assert summary["latest_checkpoint_id"] == "001"
    assert summary["checkpoint_count"] == 1
    assert summary["writes_count"] == 1
    assert threads == [
        {
            "thread_id": "thread-123",
            "updated_at": "2026-01-01T00:00:01Z",
            "created_at": "2026-01-01T00:00:01Z",
            "agent_name": "root-agent",
            "git_branch": "root-branch",
            "cwd": "/root",
            "checkpoint_count": 1,
        }
    ]


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
            "INSERT INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, metadata) "
            "VALUES (?, '', ?, ?)",
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


def test_reconstructs_messages_from_latest_inline_checkpoint(
    inspector: ModuleType, tmp_path: Path
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    db = tmp_path / "sessions.db"
    thread_id = "historical-thread"
    writable = _create_database(db)
    serde = JsonPlusSerializer()
    checkpoint = {
        "channel_values": {
            "messages": [
                HumanMessage(content="historical question"),
                AIMessage(content="historical answer"),
            ]
        }
    }
    type_name, value = serde.dumps_typed(checkpoint)
    writable.execute(
        "INSERT INTO checkpoints "
        "(thread_id, checkpoint_ns, checkpoint_id, metadata, type, checkpoint) "
        "VALUES (?, '', '001', '{}', ?, ?)",
        (thread_id, type_name, value),
    )
    writable.commit()
    writable.close()

    conn = inspector._connect_read_only(db)
    try:
        messages = inspector._reconstruct_messages(conn, thread_id)
    finally:
        conn.close()

    assert [message.content for message in messages] == [
        "historical question",
        "historical answer",
    ]


def test_reconstructs_pending_writes_from_latest_inline_checkpoint(
    inspector: ModuleType, tmp_path: Path
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    db = tmp_path / "sessions.db"
    thread_id = "active-thread"
    writable = _create_database(db)
    serde = JsonPlusSerializer()
    checkpoint = {
        "channel_values": {
            "messages": [
                HumanMessage(content="historical question", id="user-1"),
                AIMessage(content="historical answer", id="assistant-1"),
            ]
        }
    }
    type_name, value = serde.dumps_typed(checkpoint)
    writable.execute(
        "INSERT INTO checkpoints "
        "(thread_id, checkpoint_ns, checkpoint_id, metadata, type, checkpoint) "
        "VALUES (?, '', '002', '{}', ?, ?)",
        (thread_id, type_name, value),
    )
    pending = [
        [HumanMessage(content="current question", id="user-2")],
        [AIMessage(content="current answer", id="assistant-2")],
    ]
    for index, messages in enumerate(pending):
        write_type, write_value = serde.dumps_typed(messages)
        writable.execute(
            "INSERT INTO writes VALUES (?, '', '002', 'task', ?, 'messages', ?, ?)",
            (thread_id, index, write_type, write_value),
        )
    writable.commit()
    writable.close()

    conn = inspector._connect_read_only(db)
    try:
        messages = inspector._reconstruct_messages(conn, thread_id)
    finally:
        conn.close()

    assert [message.content for message in messages] == [
        "historical question",
        "historical answer",
        "current question",
        "current answer",
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


def test_message_role_normalizes_dict_and_object(inspector: ModuleType) -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    assert inspector._message_role({"type": "human"}) == "user"
    assert inspector._message_role({"role": "ai"}) == "assistant"
    assert inspector._message_role({"role": "user"}) == "user"
    assert inspector._message_role(HumanMessage(content="x")) == "user"
    assert inspector._message_role(AIMessage(content="x")) == "assistant"


def test_message_record_surfaces_malformed_tool_calls(inspector: ModuleType) -> None:
    warnings: list[str] = []
    record = inspector._message_record(
        3,
        {"role": "assistant", "content": "x", "tool_calls": "oops"},
        100,
        warnings,
    )

    assert "tool_calls" not in record
    assert record["tool_calls_malformed"] == "oops"
    assert any("tool_calls" in warning for warning in warnings)


def test_message_record_reports_tool_call_id_and_status(inspector: ModuleType) -> None:
    record = inspector._message_record(
        0,
        {"role": "tool", "content": "result", "tool_call_id": "call-9", "status": "ok"},
        100,
    )

    assert record["role"] == "tool"
    assert record["tool_call_id"] == "call-9"
    assert record["status"] == "ok"


def test_content_text_handles_varied_shapes(inspector: ModuleType) -> None:
    assert inspector._content_text(None) == ""
    assert inspector._content_text("plain") == "plain"
    assert inspector._content_text({"a": 1}) == '{"a": 1}'
    combined = inspector._content_text(
        [
            {"type": "thinking", "text": "hidden"},
            {"phase": "analysis", "text": "hidden"},
            {"type": "text", "text": "shown"},
            {"content": "nested"},
        ]
    )
    assert combined == "shown\nnested"


def test_turns_skips_system_user_messages(inspector: ModuleType) -> None:
    turns, preamble = inspector._turns(
        [
            {"role": "user", "content": "[SYSTEM] injected"},
            {"role": "assistant", "content": "hi"},
        ],
        100,
    )

    assert turns == []
    assert [record["content"] for record in preamble] == ["[SYSTEM] injected", "hi"]


def test_turns_captures_preamble_before_first_user_message(
    inspector: ModuleType,
) -> None:
    turns, preamble = inspector._turns(
        [
            {"role": "assistant", "content": "warmup"},
            {"role": "user", "content": "real question"},
            {"role": "assistant", "content": "answer"},
        ],
        100,
    )

    assert [record["content"] for record in preamble] == ["warmup"]
    assert len(turns) == 1
    assert turns[0]["start_message_index"] == 1
    assert [record["content"] for record in turns[0]["messages"]] == [
        "real question",
        "answer",
    ]


def test_resolve_thread_id_escapes_like_metacharacters(
    inspector: ModuleType, tmp_path: Path
) -> None:
    db = tmp_path / "sessions.db"
    writable = _create_database(db)
    writable.executemany(
        "INSERT INTO checkpoints "
        "(thread_id, checkpoint_ns, checkpoint_id, metadata) VALUES (?, '', ?, '{}')",
        [("a_bc", "001"), ("axbc", "002"), ("a%d", "003"), ("aXd", "004")],
    )
    writable.commit()
    writable.close()

    conn = inspector._connect_read_only(db)
    try:
        # "_" and "%" must be treated literally, not as SQL LIKE wildcards.
        assert inspector._resolve_thread_id(conn, "a_b") == "a_bc"
        assert inspector._resolve_thread_id(conn, "a%") == "a%d"
    finally:
        conn.close()


def test_connect_read_only_rejects_missing_file(
    inspector: ModuleType, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit, match="not found"):
        inspector._connect_read_only(tmp_path / "missing.db")


def test_connect_read_only_rejects_unsupported_schema(
    inspector: ModuleType, tmp_path: Path
) -> None:
    db = tmp_path / "sessions.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE checkpoints (thread_id TEXT)")
    conn.commit()
    conn.close()

    with pytest.raises(SystemExit, match="missing tables: writes"):
        inspector._connect_read_only(db)


def test_default_db_path_prefers_env_override(
    inspector: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPAGENTS_SESSIONS_DB", "/tmp/custom/sessions.db")
    assert inspector._default_db_path() == Path("/tmp/custom/sessions.db")
    monkeypatch.delenv("DEEPAGENTS_SESSIONS_DB")
    assert inspector._default_db_path() == Path.home() / ".deepagents" / ".state" / (
        "sessions.db"
    )


def test_decode_metadata_warns_on_corrupt_json(inspector: ModuleType) -> None:
    warnings: list[str] = []
    assert inspector._decode_metadata("not json", warnings) == {}
    assert any("JSON" in warning for warning in warnings)


def test_reconstruct_applies_and_skips_malformed_overwrite(
    inspector: ModuleType, tmp_path: Path
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.types import Overwrite

    db = tmp_path / "sessions.db"
    thread_id = "thread-ow"
    writable = _create_database(db)
    serde = JsonPlusSerializer()
    for checkpoint_id in ("001", "002", "003"):
        writable.execute(
            "INSERT INTO checkpoints "
            "(thread_id, checkpoint_ns, checkpoint_id, metadata) "
            "VALUES (?, '', ?, '{}')",
            (thread_id, checkpoint_id),
        )
    first_type, first_value = serde.dumps_typed(
        [HumanMessage(content="a", id="u1"), AIMessage(content="b", id="a1")]
    )
    ow_type, ow_value = serde.dumps_typed(
        Overwrite(value=[HumanMessage(content="c", id="u2")])
    )
    bad_type, bad_value = serde.dumps_typed(Overwrite(value="not-a-list"))
    writable.executemany(
        "INSERT INTO writes VALUES (?, '', ?, 'task', 0, 'messages', ?, ?)",
        [
            (thread_id, "001", first_type, first_value),
            (thread_id, "002", ow_type, ow_value),
            (thread_id, "003", bad_type, bad_value),
        ],
    )
    writable.commit()
    writable.close()

    conn = inspector._connect_read_only(db)
    warnings: list[str] = []
    try:
        messages = inspector._reconstruct_messages(conn, thread_id, warnings)
    finally:
        conn.close()

    # 001 seeds [a, b]; 002 overwrites to [c]; 003 is malformed and preserves [c].
    assert [message.content for message in messages] == ["c"]
    assert any("malformed channel overwrite" in warning for warning in warnings)


def test_load_inline_messages_falls_back_when_malformed(
    inspector: ModuleType, tmp_path: Path
) -> None:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    db = tmp_path / "sessions.db"
    writable = _create_database(db)
    serde = JsonPlusSerializer()
    type_name, value = serde.dumps_typed(
        {"channel_values": {"messages": {"not": "a list"}}}
    )
    writable.execute(
        "INSERT INTO checkpoints "
        "(thread_id, checkpoint_ns, checkpoint_id, metadata, type, checkpoint) "
        "VALUES ('t', '', '001', '{}', ?, ?)",
        (type_name, value),
    )
    writable.commit()
    writable.close()

    conn = inspector._connect_read_only(db)
    warnings: list[str] = []
    try:
        result = inspector._load_inline_messages(conn, "t", serde, warnings)
    finally:
        conn.close()

    assert result is None
    assert any("malformed" in warning.lower() for warning in warnings)


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (["prog", "thread", "--max-content", "0"], "must be positive"),
        (["prog", "--list", "0"], "--list must be positive"),
        (["prog"], "Provide a thread ID or use --list"),
        (["prog", "thread", "--list", "5"], "not both"),
    ],
)
def test_main_rejects_invalid_arguments(
    inspector: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    match: str,
) -> None:
    monkeypatch.setattr(inspector.sys, "argv", argv)
    with pytest.raises(SystemExit, match=match):
        inspector.main()


def test_main_lists_threads(
    inspector: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = tmp_path / "sessions.db"
    writable = _create_database(db)
    writable.execute(
        "INSERT INTO checkpoints "
        "(thread_id, checkpoint_ns, checkpoint_id, metadata) VALUES (?, '', ?, ?)",
        ("thread-abc", "001", json.dumps({"updated_at": "2026-01-01T00:00:01Z"})),
    )
    writable.commit()
    writable.close()

    monkeypatch.setattr(
        inspector.sys,
        "argv",
        ["prog", "--db", str(db), "--list", "5", "--include-metadata"],
    )
    inspector.main()
    output = json.loads(capsys.readouterr().out)

    assert [thread["thread_id"] for thread in output["threads"]] == ["thread-abc"]
    # --include-metadata is not meaningful with --list; surfaced as a warning.
    assert any("ignored when listing" in warning for warning in output["warnings"])


def test_main_latest_turn_end_to_end(
    inspector: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    db = tmp_path / "sessions.db"
    writable = _create_database(db)
    serde = JsonPlusSerializer()
    checkpoint = {
        "channel_values": {
            "messages": [
                HumanMessage(content="hi", id="u1"),
                AIMessage(content="hello", id="a1"),
            ]
        }
    }
    type_name, value = serde.dumps_typed(checkpoint)
    writable.execute(
        "INSERT INTO checkpoints "
        "(thread_id, checkpoint_ns, checkpoint_id, metadata, type, checkpoint) "
        "VALUES (?, '', '001', ?, ?, ?)",
        (
            "thread-xyz",
            json.dumps(
                {
                    "updated_at": "2026-01-01T00:00:01Z",
                    "turn_number": 1,
                    "turn_id": "turn-1",
                }
            ),
            type_name,
            value,
        ),
    )
    writable.commit()
    writable.close()

    monkeypatch.setattr(inspector.sys, "argv", ["prog", "thread-xyz", "--db", str(db)])
    inspector.main()
    output = json.loads(capsys.readouterr().out)

    assert output["thread"]["thread_id"] == "thread-xyz"
    assert output["turn_count"] == 1
    assert output["latest_turn"]["turn_id"] == "turn-1"
    assert output["latest_turn"]["stored_turn_number"] == 1
    assert [record["content"] for record in output["latest_turn"]["messages"]] == [
        "hi",
        "hello",
    ]
    assert "warnings" not in output
