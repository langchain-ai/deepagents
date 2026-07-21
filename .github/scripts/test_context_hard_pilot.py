"""Tests for the Context-Hard candidate validation helpers."""

from __future__ import annotations

import importlib.machinery
import json
import sqlite3
import subprocess
import sys
from decimal import Decimal
from types import ModuleType
from typing import TYPE_CHECKING

import context_hard_pilot
import pytest
from context_hard_pilot import (
    HARD_TYPES,
    LETTA_REPOSITORY,
    MIN_REASONING_QUERIES,
    MIN_REQUIRED_FILES,
    HardModeRegisterTool,
    HardModeValidationError,
    ReadOnlySQLTool,
    checkout_letta_generator,
    run_upstream_checkout,
    validate_candidate,
    validate_readonly_sql,
    verify_answer,
)

if TYPE_CHECKING:
    from pathlib import Path


_EXPENSIVE_RECURSIVE_CTE = """
WITH RECURSIVE counter(value) AS (
    SELECT 1
    UNION ALL
    SELECT value + 1 FROM counter WHERE value < 1000000
)
SELECT sum(value) FROM counter
"""

_EXPENSIVE_NESTED_SUBQUERY = """
SELECT value
FROM results
WHERE id = 1 AND (
    SELECT count(*)
    FROM results AS one
    CROSS JOIN results AS two
    CROSS JOIN results AS three
    CROSS JOIN results AS four
    CROSS JOIN results AS five
    CROSS JOIN results AS six
    CROSS JOIN results AS seven
) > 0
"""

_NESTED_SUBQUERY_VERIFICATION_QUERY = """
SELECT label
FROM results
WHERE id = (
    SELECT id FROM results WHERE label = 'Alice'
)
"""


def _candidate(**updates: object) -> dict[str, object]:
    """Build a valid candidate, allowing each test to override one field."""
    candidate: dict[str, object] = {
        "question_type": "multi_hop_chain",
        "question": "Which account has the highest verified balance?",
        "answer": "Alice",
        "verification_query": "SELECT label FROM results WHERE id = 1",
        "required_files": [f"record-{index}.txt" for index in range(6)],
        "sql_queries": [
            {"description": "Read the candidate value", "query": "SELECT 1"} for _ in range(6)
        ],
    }
    candidate.update(updates)
    return candidate


def _database(tmp_path: Path) -> Path:
    """Create a controlled SQLite database for verification tests."""
    database_path = tmp_path / "answers.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "CREATE TABLE results (id INTEGER PRIMARY KEY, value INTEGER, label TEXT)"
        )
        connection.executemany(
            "INSERT INTO results (id, value, label) VALUES (?, ?, ?)",
            [
                (1, 7, "Alice"),
                (2, 9, "Bob"),
                (3, 11, "Carol"),
                (4, 9223372036854775807, "Large"),
                (5, 9007199254740993, "Adjacent"),
                (6, "x" * 16385, "Oversized"),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return database_path


def _large_value_database(tmp_path: Path) -> Path:
    """Create a database with one value larger than the tool output budget."""
    database_path = tmp_path / "large.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("CREATE TABLE payloads (value TEXT NOT NULL)")
        connection.execute("INSERT INTO payloads (value) VALUES (?)", ("x" * 4096,))
        connection.commit()
    finally:
        connection.close()
    return database_path


def test_checkout_rejects_unpinned_revision_without_starting_a_process(
    tmp_path: Path,
) -> None:
    """Only a lowercase, full-length commit SHA may reach the process runner."""
    calls: list[tuple[object, ...]] = []

    def unexpected_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Record an unexpected process request before failing the test."""
        del kwargs
        calls.append(args)
        msg = "an unpinned revision must not invoke subprocess"
        raise AssertionError(msg)

    with pytest.raises(HardModeValidationError, match="40-character commit SHA"):
        checkout_letta_generator("main", tmp_path / "letta", run=unexpected_run)

    assert calls == []


def test_checkout_uses_argument_arrays_and_confirms_the_exact_head(tmp_path: Path) -> None:
    """Git checkout is shell-free and rejects a resolved commit mismatch."""
    revision = "a" * 40
    destination = tmp_path / "letta"
    generation_directory = destination / "letta-leaderboard" / "filesystem-agent" / "generation"
    calls: list[tuple[object, dict[str, object]]] = []

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Record each expected Git process without launching it."""
        calls.append((args, kwargs))
        if args[:4] == ["git", "-C", str(destination), "checkout"]:
            generation_directory.mkdir(parents=True)
        stdout = f"{revision}\n" if args[-2:] == ["rev-parse", "HEAD"] else ""
        return subprocess.CompletedProcess(args, 0, stdout=stdout)

    assert checkout_letta_generator(revision, destination, run=fake_run) == generation_directory
    assert [args for args, _ in calls] == [
        ["git", "clone", "--no-checkout", LETTA_REPOSITORY, str(destination)],
        ["git", "-C", str(destination), "checkout", "--detach", revision],
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
    ]
    for args, kwargs in calls:
        assert isinstance(args, list)
        assert kwargs == {
            "check": True,
            "capture_output": True,
            "text": True,
            "timeout": 60,
            "shell": False,
        }


def test_checkout_rejects_a_missing_generation_directory(tmp_path: Path) -> None:
    """A matching Git HEAD is insufficient without the trusted generator path."""
    revision = "a" * 40
    destination = tmp_path / "letta"

    def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        """Return the exact revision without creating a fake checkout tree."""
        del kwargs
        stdout = f"{revision}\n" if args[-2:] == ["rev-parse", "HEAD"] else ""
        return subprocess.CompletedProcess(args, 0, stdout=stdout)

    with pytest.raises(HardModeValidationError, match="generation directory"):
        checkout_letta_generator(revision, destination, run=fake_run)


def test_readonly_sql_tool_truncates_three_fixture_rows_to_two_json_safe_results(
    tmp_path: Path,
) -> None:
    """The model-facing SQL adapter returns only two of three fixture rows."""
    tool = ReadOnlySQLTool(_database(tmp_path))

    result = tool.execute("SELECT id, value FROM results ORDER BY id")

    assert result["success"] is True
    assert result["result"] == [[1, 7], [2, 9]]
    assert result["row_count"] == 2
    assert isinstance(result["execution_time_ms"], float)
    assert result["error"] is None
    json.dumps(result)


def test_readonly_sql_tool_rejects_a_blob_over_the_sqlite_length_limit(tmp_path: Path) -> None:
    """SQLite's per-connection string/blob cap rejects a fixed oversized blob."""
    tool = ReadOnlySQLTool(_database(tmp_path))

    result = tool.execute("SELECT zeroblob(16385)")

    assert result["success"] is False
    assert result["result"] == []
    assert result["row_count"] == 0
    assert result["error"]


def test_readonly_sql_tool_interrupts_an_expensive_recursive_query(tmp_path: Path) -> None:
    """The VM progress budget returns an ordinary error for an expensive query."""
    result = ReadOnlySQLTool(_database(tmp_path)).execute(_EXPENSIVE_RECURSIVE_CTE)

    assert result["success"] is False
    assert result["result"] == []
    assert result["row_count"] == 0
    assert "interrupted" in result["error"]


def test_readonly_sql_tool_returns_a_normal_failure_for_unsafe_sql(tmp_path: Path) -> None:
    """Unsafe model SQL fails as data without opening a writable execution path."""
    tool = ReadOnlySQLTool(_database(tmp_path))

    result = tool.execute("DELETE FROM results")

    assert result["success"] is False
    assert result["result"] == []
    assert result["row_count"] == 0
    assert result["error"] == "SQL query must start with SELECT or WITH"


def test_readonly_sql_tool_rejects_an_oversized_query_before_connecting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The interactive SQL tool applies the query byte cap before SQLite opens."""
    connected = False

    def unexpected_connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        """Record any connection that would violate fail-fast query validation."""
        del args, kwargs
        nonlocal connected
        connected = True
        msg = "oversized SQL must be rejected before opening SQLite"
        raise AssertionError(msg)

    monkeypatch.setattr(context_hard_pilot, "MAX_REASONING_QUERY_BYTES", 7)
    monkeypatch.setattr(context_hard_pilot.sqlite3, "connect", unexpected_connect)

    result = ReadOnlySQLTool(tmp_path / "answers.sqlite").execute("SELECT 1")

    assert result["success"] is False
    assert result["result"] == []
    assert result["row_count"] == 0
    assert "UTF-8 byte limit" in result["error"]
    assert not connected


def test_readonly_sql_tool_caps_an_oversized_unicode_database_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Database errors stay JSON-safe and within the fixed model-facing byte cap."""

    class FailingConnection:
        """Connection that reaches execution then returns one oversized SQLite error."""

        def set_authorizer(self, authorizer: object) -> None:
            """Accept the tool's existing read-only authorizer setup."""
            del authorizer

        def setlimit(self, category: int, limit: int) -> None:
            """Accept the two per-connection resource caps without changing behavior."""
            del category, limit

        def set_progress_handler(self, handler: object, instructions: int) -> None:
            """Accept the resource guard registration before the fixed failure."""
            del handler, instructions

        def execute(self, query: str) -> object:
            """Raise a fixed Unicode error for the fixed read-only query."""
            if query != "SELECT 1":
                msg = "unexpected tool query"
                raise AssertionError(msg)
            raise sqlite3.OperationalError("漢字🚀" * 2048)

        def close(self) -> None:
            """Accept the tool's required connection cleanup."""

    def fake_connect(database_uri: str, *, uri: bool) -> FailingConnection:
        """Return the deterministic failing connection in URI read-only mode."""
        del database_uri
        if not uri:
            msg = "ReadOnlySQLTool must retain URI connection mode"
            raise AssertionError(msg)
        return FailingConnection()

    monkeypatch.setattr(context_hard_pilot.sqlite3, "connect", fake_connect)

    result = ReadOnlySQLTool(tmp_path / "answers.sqlite").execute("SELECT 1")

    assert result["success"] is False
    json.dumps(result)
    error = result["error"]
    assert isinstance(error, str)
    assert len(error.encode("utf-8")) <= context_hard_pilot._MAX_TOOL_ERROR_BYTES  # noqa: SLF001  # test fixed output cap
    assert context_hard_pilot._TOOL_ERROR_TRUNCATION_MARKER in error  # noqa: SLF001  # test fixed marker


def test_readonly_sql_tool_rejects_a_result_set_over_the_column_limit(tmp_path: Path) -> None:
    """A query exceeding the per-connection SQLite column limit fails as data."""
    tool = ReadOnlySQLTool(_database(tmp_path))

    result = tool.execute(
        """SELECT
        1 AS column_01, 1 AS column_02, 1 AS column_03, 1 AS column_04,
        1 AS column_05, 1 AS column_06, 1 AS column_07, 1 AS column_08,
        1 AS column_09, 1 AS column_10, 1 AS column_11, 1 AS column_12,
        1 AS column_13, 1 AS column_14, 1 AS column_15, 1 AS column_16,
        1 AS column_17
        """
    )

    assert result["success"] is False
    assert result["result"] == []
    assert result["row_count"] == 0
    assert result["error"]


def test_readonly_sql_tool_rejects_json_output_over_the_byte_limit(tmp_path: Path) -> None:
    """A bounded row count cannot bypass the model-facing serialized byte budget."""
    tool = ReadOnlySQLTool(_large_value_database(tmp_path))

    result = tool.execute("SELECT value FROM payloads")

    assert result["success"] is False
    assert result["result"] == []
    assert result["row_count"] == 0
    assert "serialized result exceeds" in result["error"]


def test_hard_mode_register_tool_writes_compatible_raw_and_parsed_records(
    tmp_path: Path,
) -> None:
    """Accepted records are locally verified then written in Letta-compatible JSONL."""
    output_path = tmp_path / "generated" / "agent_generated_questions.jsonl"
    tool = HardModeRegisterTool(output_path, _database(tmp_path), [])
    candidate = _candidate(
        answer="7",
        verification_query="SELECT value FROM results WHERE id = 1",
    )

    result = tool.register(
        question=candidate["question"],
        sql_queries=candidate["sql_queries"],
        answer=candidate["answer"],
        answer_reasoning="The result is explicit in the final query.",
        question_type=candidate["question_type"],
        required_files=candidate["required_files"],
        verification_query=candidate["verification_query"],
    )

    query_results = [
        {
            "description": "Read the candidate value",
            "query": "SELECT 1",
            "result": [[1]],
        }
        for _ in range(MIN_REASONING_QUERIES)
    ]
    assert result == {
        "success": True,
        "message": "Question registered successfully. Answer: 7",
        "answer": "7",
        "query_results": query_results,
        "total_questions": 1,
    }
    raw_records = [json.loads(line) for line in output_path.read_text().splitlines()]
    parsed_records = [
        json.loads(line)
        for line in (output_path.parent / "agent_generated_questions_parsed.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(raw_records) == 1
    raw_record = raw_records[0]
    assert raw_record["question"] == candidate["question"]
    assert raw_record["answer"] == candidate["answer"]
    assert raw_record["difficulty"] == "hard"
    assert raw_record["question_type"] == candidate["question_type"]
    assert raw_record["required_files"] == candidate["required_files"]
    assert raw_record["answer_reasoning"] == "The result is explicit in the final query."
    assert raw_record["sql_queries"] == query_results
    assert raw_record["verification_query"] == candidate["verification_query"]
    assert isinstance(raw_record["timestamp"], str)
    assert raw_record["timestamp"]
    assert parsed_records == [
        {
            "input": candidate["question"],
            "ground_truth": candidate["answer"],
            "agent_args": {
                "tags": [],
                "extra": {
                    "required_files": candidate["required_files"],
                    "question_type": candidate["question_type"],
                    "difficulty": "hard",
                    "verification_query": candidate["verification_query"],
                },
            },
        }
    ]


def test_hard_mode_register_tool_rejects_before_writing(tmp_path: Path) -> None:
    """Invalid candidate data produces model-facing failure and no JSONL files."""
    output_path = tmp_path / "generated" / "agent_generated_questions.jsonl"
    tool = HardModeRegisterTool(output_path, _database(tmp_path), [])
    candidate = _candidate(
        sql_queries=[{"description": "Attempt an unsafe mutation", "query": "DELETE FROM results"}]
        * MIN_REASONING_QUERIES
    )

    result = tool.register(
        question=candidate["question"],
        sql_queries=candidate["sql_queries"],
        answer=candidate["answer"],
        answer_reasoning="Unsafe path.",
        question_type=candidate["question_type"],
        required_files=candidate["required_files"],
        verification_query=candidate["verification_query"],
    )

    assert result["success"] is False
    assert "must start with SELECT or WITH" in result["error"]
    assert not output_path.parent.exists()


def test_hard_mode_register_tool_rejects_failed_reasoning_queries_before_writing(
    tmp_path: Path,
) -> None:
    """All six allowed reasoning queries must complete successfully before append."""
    output_path = tmp_path / "generated" / "agent_generated_questions.jsonl"
    tool = HardModeRegisterTool(output_path, _database(tmp_path), [])
    candidate = _candidate(
        sql_queries=[
            {
                "description": "Read a missing table",
                "query": "SELECT value FROM missing_results",
            }
            for _ in range(MIN_REASONING_QUERIES)
        ]
    )

    result = tool.register(
        question=candidate["question"],
        sql_queries=candidate["sql_queries"],
        answer=candidate["answer"],
        answer_reasoning="Each query refers to a missing table.",
        question_type=candidate["question_type"],
        required_files=candidate["required_files"],
        verification_query=candidate["verification_query"],
    )

    assert result["success"] is False
    assert "reasoning query failed" in result["error"]
    assert not output_path.parent.exists()


@pytest.mark.parametrize(
    ("field", "limit_name", "value", "expected_limit"),
    [
        ("query", "MAX_REASONING_QUERY_BYTES", "SELECT 1", 8192),
        ("description", "MAX_REASONING_DESCRIPTION_BYTES", "reasoning", 2048),
    ],
)
def test_validate_candidate_rejects_reasoning_text_over_utf8_byte_limits(
    field: str,
    limit_name: str,
    value: str,
    expected_limit: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reasoning text limits reject model input before SQL validation or execution."""
    assert getattr(context_hard_pilot, limit_name) == expected_limit
    sql_query = {"description": "reasoning", "query": "SELECT 1"}
    sql_query[field] = value
    monkeypatch.setattr(context_hard_pilot, limit_name, 1)

    with pytest.raises(HardModeValidationError, match="UTF-8 byte limit"):
        validate_candidate(
            _candidate(sql_queries=[sql_query] * MIN_REASONING_QUERIES),
            existing_questions=[],
        )


def test_hard_mode_register_tool_rejects_raw_jsonl_over_aggregate_byte_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The aggregate raw JSONL cap rejects a record before either file is appended."""
    output_path = tmp_path / "generated" / "agent_generated_questions.jsonl"
    monkeypatch.setattr(context_hard_pilot, "MAX_RAW_JSONL_BYTES", 1)
    tool = HardModeRegisterTool(output_path, _database(tmp_path), [])
    candidate = _candidate(
        answer="7",
        verification_query="SELECT value FROM results WHERE id = 1",
    )

    result = tool.register(
        question=candidate["question"],
        sql_queries=candidate["sql_queries"],
        answer=candidate["answer"],
        answer_reasoning="The result is explicit in the final query.",
        question_type=candidate["question_type"],
        required_files=candidate["required_files"],
        verification_query=candidate["verification_query"],
    )

    assert result["success"] is False
    assert "raw JSONL" in result["error"]
    assert not output_path.parent.exists()


@pytest.mark.parametrize(
    ("updates", "error_fragment"),
    [
        ({"verification_query": "SELECT 'PERS-0001'"}, "pers-"),
        (
            {"verification_query": "SELECT CASE WHEN 1 THEN 'Alice' ELSE 'Bob' END"},
            "CASE",
        ),
        ({"answer": "No pets"}, "concrete"),
    ],
)
def test_hard_mode_register_tool_rejects_nonconcrete_or_hardcoded_inputs_before_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    updates: dict[str, object],
    error_fragment: str,
) -> None:
    """Registration guardrails reject unsafe candidate inputs before backend work."""
    output_path = tmp_path / "generated" / "agent_generated_questions.jsonl"
    tool = HardModeRegisterTool(output_path, _database(tmp_path), [])
    candidate = _candidate(**updates)
    verify_calls: list[tuple[object, ...]] = []
    reasoning_calls: list[str] = []

    def unexpected_verify(*args: object, **kwargs: object) -> None:
        """Record any verification call that must not occur for rejected data."""
        del kwargs
        verify_calls.append(args)
        msg = "validation must precede verification"
        raise AssertionError(msg)

    def unexpected_execute(self: ReadOnlySQLTool, query: str) -> dict[str, object]:
        """Record any reasoning execution that must not occur for rejected data."""
        del self
        reasoning_calls.append(query)
        msg = "validation must precede reasoning execution"
        raise AssertionError(msg)

    monkeypatch.setattr(context_hard_pilot, "verify_answer", unexpected_verify)
    monkeypatch.setattr(ReadOnlySQLTool, "execute", unexpected_execute)

    result = tool.register(
        question=candidate["question"],
        sql_queries=candidate["sql_queries"],
        answer=candidate["answer"],
        answer_reasoning="Fixed reasoning text.",
        question_type=candidate["question_type"],
        required_files=candidate["required_files"],
        verification_query=candidate["verification_query"],
    )

    assert result["success"] is False
    assert error_fragment.casefold() in result["error"].casefold()
    assert verify_calls == []
    assert reasoning_calls == []
    assert not output_path.parent.exists()


def test_hard_mode_register_tool_rejects_oversized_verification_before_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verification text cap runs before any database or reasoning work."""
    output_path = tmp_path / "generated" / "agent_generated_questions.jsonl"
    assert context_hard_pilot.MAX_VERIFICATION_QUERY_BYTES == 8192
    monkeypatch.setattr(context_hard_pilot, "MAX_VERIFICATION_QUERY_BYTES", 1)
    tool = HardModeRegisterTool(output_path, _database(tmp_path), [])
    candidate = _candidate()
    verify_calls: list[tuple[object, ...]] = []
    reasoning_calls: list[str] = []

    def unexpected_verify(*args: object, **kwargs: object) -> None:
        """Record any forbidden verification call."""
        del kwargs
        verify_calls.append(args)
        msg = "verification byte cap must run first"
        raise AssertionError(msg)

    def unexpected_execute(self: ReadOnlySQLTool, query: str) -> dict[str, object]:
        """Record any forbidden reasoning execution."""
        del self
        reasoning_calls.append(query)
        msg = "verification byte cap must run first"
        raise AssertionError(msg)

    monkeypatch.setattr(context_hard_pilot, "verify_answer", unexpected_verify)
    monkeypatch.setattr(ReadOnlySQLTool, "execute", unexpected_execute)

    result = tool.register(
        question=candidate["question"],
        sql_queries=candidate["sql_queries"],
        answer=candidate["answer"],
        answer_reasoning="Fixed reasoning text.",
        question_type=candidate["question_type"],
        required_files=candidate["required_files"],
        verification_query=candidate["verification_query"],
    )

    assert result["success"] is False
    assert "verification query" in result["error"]
    assert verify_calls == []
    assert reasoning_calls == []
    assert not output_path.parent.exists()


def test_hard_mode_register_tool_rejects_oversized_reasoning_before_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Raw input cap runs before verification or reasoning-query execution."""
    output_path = tmp_path / "generated" / "agent_generated_questions.jsonl"
    monkeypatch.setattr(context_hard_pilot, "MAX_RAW_JSONL_BYTES", 1)
    tool = HardModeRegisterTool(output_path, _database(tmp_path), [])
    candidate = _candidate()
    verify_calls: list[tuple[object, ...]] = []
    reasoning_calls: list[str] = []

    def unexpected_verify(*args: object, **kwargs: object) -> None:
        """Record any forbidden verification call."""
        del kwargs
        verify_calls.append(args)
        msg = "raw input cap must run first"
        raise AssertionError(msg)

    def unexpected_execute(self: ReadOnlySQLTool, query: str) -> dict[str, object]:
        """Record any forbidden reasoning execution."""
        del self
        reasoning_calls.append(query)
        msg = "raw input cap must run first"
        raise AssertionError(msg)

    monkeypatch.setattr(context_hard_pilot, "verify_answer", unexpected_verify)
    monkeypatch.setattr(ReadOnlySQLTool, "execute", unexpected_execute)

    result = tool.register(
        question=candidate["question"],
        sql_queries=candidate["sql_queries"],
        answer=candidate["answer"],
        answer_reasoning="Fixed reasoning text.",
        question_type=candidate["question_type"],
        required_files=candidate["required_files"],
        verification_query=candidate["verification_query"],
    )

    assert result["success"] is False
    assert "raw JSONL" in result["error"]
    assert verify_calls == []
    assert reasoning_calls == []
    assert not output_path.parent.exists()


def test_hard_mode_register_tool_rejects_more_than_twelve_reasoning_queries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reasoning-query ceiling rejects candidates before local SQL or output writes."""
    output_path = tmp_path / "generated" / "agent_generated_questions.jsonl"
    tool = HardModeRegisterTool(output_path, _database(tmp_path), [])
    candidate = _candidate(
        sql_queries=[
            {"description": "Read the candidate value", "query": "SELECT 1"} for _ in range(13)
        ]
    )
    calls: list[str] = []

    def unexpected_execute(self: ReadOnlySQLTool, query: str) -> dict[str, object]:
        """Record any reasoning execution that should be rejected by count first."""
        del self
        calls.append(query)
        msg = "count validation must prevent reasoning execution"
        raise AssertionError(msg)

    monkeypatch.setattr(ReadOnlySQLTool, "execute", unexpected_execute)

    result = tool.register(
        question=candidate["question"],
        sql_queries=candidate["sql_queries"],
        answer=candidate["answer"],
        answer_reasoning="The query count exceeds the hard-mode policy.",
        question_type=candidate["question_type"],
        required_files=candidate["required_files"],
        verification_query=candidate["verification_query"],
    )

    assert result["success"] is False
    assert context_hard_pilot.MAX_REASONING_QUERIES == 12
    assert "at most 12" in result["error"]
    assert calls == []
    assert not output_path.parent.exists()


def test_run_upstream_checkout_imports_only_the_trusted_generator_and_restores_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Trusted on-disk code wins over conflicts without polluting process state."""
    generator_directory = tmp_path / "generator"
    tools_directory = generator_directory / "tools"
    tools_directory.mkdir(parents=True)
    (generator_directory / "question_generator.py").write_text(
        """
from trusted_observer import agents, events, write_records
from trusted_helper import HELPER_ORIGIN
from context import ContextMixin
from display import DisplayMixin
from parallel import ParallelMixin
from tools.register_question_tool import REGISTER_QUESTION_TOOL_DICT
from tools.sql_execute_tool import EXECUTE_SQL_TOOL_DICT


class QuestionGeneratorAgent(DisplayMixin, ContextMixin, ParallelMixin):
    def __init__(self, db_path, output_path, model):
        events.append(("agent", REGISTER_QUESTION_TOOL_DICT["origin"], EXECUTE_SQL_TOOL_DICT["origin"], HELPER_ORIGIN, model))
        self.output_path = output_path
        self.system_prompt = "base system"
        self.type_prompts = {"multi_hop_chain": "base chain", "multi_entity_comparison": "base comparison"}
        agents.append(self)

    def generate_questions(self, count, question_type):
        events.append(("generate", count, question_type))
        write_records(self.output_path, question_type, count)
        """.strip(),
        encoding="utf-8",
    )
    (tools_directory / "__init__.py").write_text("", encoding="utf-8")
    (tools_directory / "register_question_tool.py").write_text(
        'REGISTER_QUESTION_TOOL_DICT = {"origin": "trusted-register"}\n',
        encoding="utf-8",
    )
    (tools_directory / "sql_execute_tool.py").write_text(
        'EXECUTE_SQL_TOOL_DICT = {"origin": "trusted-sql"}\n',
        encoding="utf-8",
    )
    (generator_directory / "trusted_helper.py").write_text(
        'HELPER_ORIGIN = "trusted-helper"\n',
        encoding="utf-8",
    )
    (generator_directory / "display.py").write_text(
        'class DisplayMixin:\n    display_marker = "trusted-display"\n',
        encoding="utf-8",
    )
    (generator_directory / "context.py").write_text(
        'class ContextMixin:\n    context_marker = "trusted-context"\n',
        encoding="utf-8",
    )
    (generator_directory / "parallel.py").write_text(
        'class ParallelMixin:\n    parallel_marker = "trusted-parallel"\n',
        encoding="utf-8",
    )

    events: list[tuple[object, ...]] = []
    agents: list[object] = []
    observer_module = ModuleType("trusted_observer")
    observer_module.events = events
    observer_module.agents = agents

    def write_records(output_path: Path, question_type: str, count: int) -> None:
        """Append the simulated raw registrations requested by the trusted agent."""
        with output_path.open("a", encoding="utf-8") as output_file:
            for index in range(count):
                output_file.write(json.dumps({"question": f"{question_type}-{index}"}) + "\n")

    observer_module.write_records = write_records
    conflicting_agent_module = ModuleType("question_generator")

    class ConflictingQuestionGeneratorAgent:
        """Detect use of the preloaded conflicting generator module."""

        def __init__(self, db_path: Path, output_path: Path, model: str) -> None:
            """Record the unwanted conflicting import if it is selected."""
            del db_path, output_path, model
            events.append(("conflicting-agent",))
            self.system_prompt = "conflicting"
            self.type_prompts = {
                "multi_hop_chain": "conflicting",
                "multi_entity_comparison": "conflicting",
            }

        def generate_questions(self, count: int, question_type: str) -> None:
            """Record unwanted calls on the conflicting preloaded agent."""
            events.append(("conflicting-generate", count, question_type))

    class MetaPathFallbackLoader:
        """Record any forbidden normal-loader execution for a direct trusted module."""

        def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
            """Defer module allocation to importlib's standard implementation."""
            del spec

        def exec_module(self, module: ModuleType) -> None:
            """Expose a conflicting agent only if normal imports invoke this loader."""
            events.append(("meta-path-loader",))
            module.QuestionGeneratorAgent = ConflictingQuestionGeneratorAgent

    class MetaPathFallbackFinder:
        """Offer the forbidden observer loader only for `question_generator`."""

        def find_spec(
            self, fullname: str, path: object, target: object = None
        ) -> importlib.machinery.ModuleSpec | None:
            """Return the observer spec without consulting any filesystem path."""
            del path, target
            if fullname == "question_generator":
                return importlib.machinery.ModuleSpec(fullname, MetaPathFallbackLoader())
            return None

    class ForgedCachedLoader:
        """Record execution if direct module discovery trusts the cache entry."""

        def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
            """Defer module allocation to importlib's standard implementation."""
            del spec

        def exec_module(self, module: ModuleType) -> None:
            """Expose a conflicting agent from a loader that falsely claims trust."""
            events.append(("forged-cached-loader",))
            module.QuestionGeneratorAgent = ConflictingQuestionGeneratorAgent

    class ForgedCachedFinder:
        """Return a forged spec that claims the exact trusted source origin."""

        def __init__(self) -> None:
            """Delegate non-target bare helpers to the real trusted source directory."""
            self.file_finder = importlib.machinery.FileFinder(
                str(generator_directory),
                (importlib.machinery.SourceFileLoader, importlib.machinery.SOURCE_SUFFIXES),
            )

        def find_spec(
            self, fullname: str, target: object = None
        ) -> importlib.machinery.ModuleSpec | None:
            """Serve only the forged direct question-generator module spec."""
            if fullname == "question_generator":
                return importlib.machinery.ModuleSpec(
                    fullname,
                    ForgedCachedLoader(),
                    origin=str(generator_directory / "question_generator.py"),
                )
            return self.file_finder.find_spec(fullname, target)

    conflicting_agent_module.QuestionGeneratorAgent = ConflictingQuestionGeneratorAgent
    conflicting_display_module = ModuleType("display")
    conflicting_context_module = ModuleType("context")
    conflicting_parallel_module = ModuleType("parallel")

    class ConflictingDisplayMixin:
        """Mark an unwanted preloaded display mixin."""

        display_marker = "conflicting-display"

    class ConflictingContextMixin:
        """Mark an unwanted preloaded context mixin."""

        context_marker = "conflicting-context"

    class ConflictingParallelMixin:
        """Mark an unwanted preloaded parallel mixin."""

        parallel_marker = "conflicting-parallel"

    conflicting_display_module.DisplayMixin = ConflictingDisplayMixin
    conflicting_context_module.ContextMixin = ConflictingContextMixin
    conflicting_parallel_module.ParallelMixin = ConflictingParallelMixin
    conflicting_tools_module = ModuleType("tools")
    conflicting_register_module = ModuleType("tools.register_question_tool")
    conflicting_register_module.REGISTER_QUESTION_TOOL_DICT = {"origin": "conflicting-register"}
    conflicting_sql_module = ModuleType("tools.sql_execute_tool")
    conflicting_sql_module.EXECUTE_SQL_TOOL_DICT = {"origin": "conflicting-sql"}
    preloaded_modules = {
        "question_generator": conflicting_agent_module,
        "display": conflicting_display_module,
        "context": conflicting_context_module,
        "parallel": conflicting_parallel_module,
        "tools": conflicting_tools_module,
        "tools.register_question_tool": conflicting_register_module,
        "tools.sql_execute_tool": conflicting_sql_module,
    }
    original_path = list(sys.path)
    original_dont_write_bytecode = sys.dont_write_bytecode
    monkeypatch.setitem(sys.modules, "trusted_observer", observer_module)
    for name, module in preloaded_modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys, "meta_path", [MetaPathFallbackFinder(), *sys.meta_path])
    monkeypatch.setitem(
        sys.path_importer_cache,
        str(generator_directory),
        ForgedCachedFinder(),
    )
    original_modules = dict(sys.modules)
    original_path_importer_cache = dict(sys.path_importer_cache)
    output_path = tmp_path / "output"
    output_path.write_text('{"question": "existing question"}\n', encoding="utf-8")

    run_upstream_checkout(
        generator_directory,
        tmp_path / "fixture.sqlite",
        output_path,
        "test-model",
    )

    assert events == [
        ("agent", "trusted-register", "trusted-sql", "trusted-helper", "test-model"),
        ("generate", 3, "multi_hop_chain"),
        ("generate", 2, "multi_entity_comparison"),
    ]
    assert ("meta-path-loader",) not in events
    assert ("forged-cached-loader",) not in events
    assert len(agents) == 1
    agent = agents[0]
    assert agent.display_marker == "trusted-display"
    assert agent.context_marker == "trusted-context"
    assert agent.parallel_marker == "trusted-parallel"
    assert isinstance(agent.sql_tool, ReadOnlySQLTool)
    assert isinstance(agent.register_tool, HardModeRegisterTool)
    assert "six distinct source files" in agent.system_prompt
    assert "nested subqueries" in agent.system_prompt
    assert "parallel greps" in agent.type_prompts["multi_hop_chain"]
    assert (
        "two sequential, indirect dependency chains"
        in agent.type_prompts["multi_entity_comparison"]
    )
    assert "not CTEs or derived tables" in agent.type_prompts["multi_entity_comparison"]
    raw_records = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(raw_records) == 6
    assert raw_records[0] == {"question": "existing question"}
    assert sys.path == original_path
    assert sys.dont_write_bytecode is original_dont_write_bytecode
    assert not list(generator_directory.rglob("__pycache__"))
    assert set(sys.modules) == set(original_modules)
    for name, module in original_modules.items():
        assert sys.modules[name] is module
    assert set(sys.path_importer_cache) == set(original_path_importer_cache)
    for path, finder in original_path_importer_cache.items():
        assert sys.path_importer_cache[path] is finder


def test_run_upstream_checkout_rejects_any_registration_count_other_than_five(
    tmp_path: Path,
) -> None:
    """The two fixed generation calls must produce exactly five raw registrations."""
    generator_directory = tmp_path / "generator"
    tools_directory = generator_directory / "tools"
    tools_directory.mkdir(parents=True)
    (generator_directory / "display.py").write_text("", encoding="utf-8")
    (generator_directory / "context.py").write_text("", encoding="utf-8")
    (generator_directory / "parallel.py").write_text("", encoding="utf-8")
    (tools_directory / "__init__.py").write_text("", encoding="utf-8")
    (tools_directory / "register_question_tool.py").write_text(
        'REGISTER_QUESTION_TOOL_DICT = {"name": "register"}\n',
        encoding="utf-8",
    )
    (tools_directory / "sql_execute_tool.py").write_text(
        'EXECUTE_SQL_TOOL_DICT = {"name": "execute"}\n',
        encoding="utf-8",
    )
    (generator_directory / "question_generator.py").write_text(
        """
class QuestionGeneratorAgent:
    def __init__(self, db_path, output_path, model):
        self.system_prompt = "base system"
        self.type_prompts = {"multi_hop_chain": "base chain", "multi_entity_comparison": "base comparison"}

    def generate_questions(self, count, question_type):
        return None
        """.strip(),
        encoding="utf-8",
    )

    with pytest.raises(HardModeValidationError, match="exactly five"):
        run_upstream_checkout(
            generator_directory,
            tmp_path / "fixture.sqlite",
            tmp_path / "output",
            "test-model",
        )


def test_run_upstream_checkout_rejects_a_question_generator_from_fallback_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing trusted module cannot silently resolve from the original sys.path."""
    generator_directory = tmp_path / "generator"
    tools_directory = generator_directory / "tools"
    fallback_directory = tmp_path / "fallback"
    tools_directory.mkdir(parents=True)
    fallback_directory.mkdir()
    (tools_directory / "__init__.py").write_text("", encoding="utf-8")
    (tools_directory / "register_question_tool.py").write_text(
        'REGISTER_QUESTION_TOOL_DICT = {"origin": "trusted-register"}\n',
        encoding="utf-8",
    )
    (tools_directory / "sql_execute_tool.py").write_text(
        'EXECUTE_SQL_TOOL_DICT = {"origin": "trusted-sql"}\n',
        encoding="utf-8",
    )
    (fallback_directory / "question_generator.py").write_text(
        """
from fallback_observer import events

events.append("fallback question generator imported")


class QuestionGeneratorAgent:
    def __init__(self, db_path, output_path, model):
        self.system_prompt = "fallback"
        self.type_prompts = {"multi_hop_chain": "fallback", "multi_entity_comparison": "fallback"}

    def generate_questions(self, count, question_type):
        return None
        """.strip(),
        encoding="utf-8",
    )
    events: list[str] = []
    observer_module = ModuleType("fallback_observer")
    observer_module.events = events
    monkeypatch.setitem(sys.modules, "fallback_observer", observer_module)
    monkeypatch.syspath_prepend(str(fallback_directory))
    original_modules = dict(sys.modules)

    with pytest.raises(HardModeValidationError, match="trusted generator directory"):
        run_upstream_checkout(
            generator_directory,
            tmp_path / "fixture.sqlite",
            tmp_path / "output",
            "test-model",
        )

    assert events == []
    assert set(sys.modules) == set(original_modules)
    for name, module in original_modules.items():
        assert sys.modules[name] is module


def test_verify_answer_uses_bounded_fetchmany(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verification requests no more than two rows from the SQLite cursor."""

    class FakeCursor:
        """Cursor that supplies one scalar row only through bounded retrieval."""

        def __init__(self) -> None:
            """Record each requested bounded retrieval size."""
            self.fetchmany_sizes: list[int] = []

        def fetchall(self) -> list[tuple[int]]:
            """Fail if verification attempts to materialize every result row."""
            msg = "verify_answer must not call fetchall"
            raise AssertionError(msg)

        def fetchmany(self, size: int) -> list[tuple[int]]:
            """Return the scalar row after enforcing the expected fetch bound."""
            self.fetchmany_sizes.append(size)
            if size != 2:
                msg = "verify_answer must request exactly two rows"
                raise AssertionError(msg)
            return [(1,)]

    class FakeConnection:
        """Connection that exposes the operations used by `verify_answer`."""

        def __init__(self, cursor: FakeCursor) -> None:
            """Store the supplied cursor and track connection cleanup."""
            self.cursor = cursor
            self.closed = False
            self.authorizer: object | None = None

        def set_authorizer(self, authorizer: object) -> None:
            """Record the authorizer installed before query execution."""
            self.authorizer = authorizer

        def setlimit(self, category: int, limit: int) -> None:
            """Accept verification's SQLite resource limits before execution."""
            del category, limit

        def set_progress_handler(self, handler: object, instructions: int) -> None:
            """Accept the resource guard registration before query execution."""
            del handler, instructions

        def execute(self, query: str) -> FakeCursor:
            """Return the fixed cursor for the fixed test query."""
            if query != "SELECT value FROM results WHERE id = 1":
                msg = "unexpected verification query"
                raise AssertionError(msg)
            if not callable(self.authorizer):
                msg = "verify_answer must install an authorizer before execution"
                raise TypeError(msg)
            self.authorizer(sqlite3.SQLITE_READ, "results", "value", "main", None)
            return self.cursor

        def close(self) -> None:
            """Record `verify_answer`'s required connection cleanup."""
            self.closed = True

    cursor = FakeCursor()
    connection = FakeConnection(cursor)

    def fake_connect(database_uri: str, *, uri: bool) -> FakeConnection:
        """Return the fake connection while checking URI mode is retained."""
        del database_uri
        if not uri:
            msg = "verify_answer must retain URI connection mode"
            raise AssertionError(msg)
        return connection

    monkeypatch.setattr(context_hard_pilot.sqlite3, "connect", fake_connect)

    verify_answer(
        tmp_path / "answers.sqlite",
        "SELECT value FROM results WHERE id = 1",
        "1",
    )

    assert cursor.fetchmany_sizes == [2]
    assert connection.closed


def test_verify_answer_interrupts_an_expensive_recursive_query(tmp_path: Path) -> None:
    """Verification applies the SQLite progress guard to nested model SQL."""
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        verify_answer(_database(tmp_path), _EXPENSIVE_NESTED_SUBQUERY, "7")


def test_verify_answer_rejects_a_source_value_over_the_sqlite_length_limit(
    tmp_path: Path,
) -> None:
    """Verification applies the SQLite length cap to a source-derived value."""
    with pytest.raises(sqlite3.Error):
        verify_answer(_database(tmp_path), "SELECT value FROM results WHERE id = 6", "0")


@pytest.mark.parametrize(
    ("query", "answer"),
    [
        ("SELECT ('Alice')", "Alice"),
        ("SELECT lower('Alice')", "Alice"),
        ("SELECT 7 + 0", "7"),
    ],
)
def test_verify_answer_rejects_literal_or_constant_projections(
    tmp_path: Path, query: str, answer: str
) -> None:
    """Verification rejects final projections that do not derive from source data."""
    with pytest.raises(HardModeValidationError, match="projection"):
        verify_answer(_database(tmp_path), query, answer)


@pytest.mark.parametrize(
    ("query", "error_fragment"),
    [
        (
            "WITH answer_alias AS (SELECT 'Alice' AS answer) SELECT answer FROM answer_alias",
            "top-level SELECT",
        ),
        (
            "SELECT answer FROM (SELECT 'Alice' AS answer) AS answer_alias",
            "derived table",
        ),
        (
            "SELECT answer FROM /* bypass */ (SELECT 'Alice' AS answer) AS answer_alias",
            "derived table",
        ),
        (
            "SELECT answer FROM -- bypass\n(SELECT 'Alice' AS answer) AS answer_alias",
            "derived table",
        ),
    ],
)
def test_verify_answer_rejects_cte_and_final_derived_table_sources(
    tmp_path: Path, query: str, error_fragment: str
) -> None:
    """Verification only permits top-level SELECTs over non-derived final sources."""
    with pytest.raises(HardModeValidationError, match=error_fragment):
        verify_answer(
            _database(tmp_path),
            query,
            "Alice",
        )


def test_verify_answer_rejects_an_unterminated_final_source_comment(tmp_path: Path) -> None:
    """A final source comment must close before verification can inspect the source."""
    with pytest.raises(HardModeValidationError, match="unterminated block comment"):
        verify_answer(_database(tmp_path), "SELECT value FROM /* unterminated", "7")


@pytest.mark.parametrize(
    ("query", "answer"),
    [
        ("SELECT 'Alice' FROM results WHERE id = 1", "Alice"),
        ('SELECT "Alice" FROM results WHERE id = 1', "Alice"),
        ("SELECT ('Alice') FROM results WHERE id = 1", "Alice"),
        ("SELECT lower('Alice') FROM results WHERE id = 1", "Alice"),
    ],
)
def test_verify_answer_rejects_literal_projection_from_a_source_table(
    tmp_path: Path, query: str, answer: str
) -> None:
    """A source-table read cannot justify a literal final answer projection."""
    with pytest.raises(HardModeValidationError, match="projection"):
        verify_answer(_database(tmp_path), query, answer)


@pytest.mark.parametrize(
    ("query", "answer"),
    [
        ("SELECT label FROM results WHERE id = 1", "Alice"),
        ("SELECT label FROM /* direct source */ results WHERE id = 1", "Alice"),
        ("SELECT value FROM results WHERE id = 1", "7"),
        ("SELECT COUNT(*) FROM results", "6"),
        (_NESTED_SUBQUERY_VERIFICATION_QUERY, "Alice"),
    ],
)
def test_verify_answer_allows_data_access_and_nested_subqueries(
    tmp_path: Path, query: str, answer: str
) -> None:
    """Data-derived scalar verification remains usable with nested subqueries."""
    verify_answer(_database(tmp_path), query, answer)


def test_hard_mode_constants_and_valid_candidate_are_accepted() -> None:
    """The two approved question types and a fully formed candidate pass."""
    assert frozenset({"multi_hop_chain", "multi_entity_comparison"}) == HARD_TYPES
    assert MIN_REQUIRED_FILES == 6
    assert MIN_REASONING_QUERIES == 6

    validate_candidate(_candidate(), existing_questions=["A different question"])


@pytest.mark.parametrize(
    "query",
    [
        "SELECT 1",
        "WITH latest AS (SELECT 1 AS value) SELECT value FROM latest",
    ],
)
def test_validate_readonly_sql_accepts_select_and_with(query: str) -> None:
    """A single ordinary SELECT or CTE-backed SELECT is allowed."""
    validate_readonly_sql(query)


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "SELECT 1;",
        "EXPLAIN SELECT 1",
        "ATTACH DATABASE 'other.sqlite' AS other",
        "DETACH DATABASE other",
        "PRAGMA foreign_keys",
        "VACUUM",
        "SELECT load_extension('malicious')",
        "INSERT INTO results VALUES (3, 11)",
        "UPDATE results SET value = 11",
        "DELETE FROM results",
        "DROP TABLE results",
        "CREATE TABLE other (value INTEGER)",
        "ALTER TABLE results RENAME TO renamed_results",
        "REPLACE INTO results VALUES (3, 11)",
        "WITH source AS (SELECT 1) DELETE FROM results",
    ],
)
def test_validate_readonly_sql_rejects_unsafe_or_non_readonly_query(query: str) -> None:
    """SQL policy blocks separators, non-queries, and dangerous operations."""
    with pytest.raises(HardModeValidationError):
        validate_readonly_sql(query)


def test_validate_candidate_rejects_a_non_hard_question_type() -> None:
    """Only the two hard-mode question types are eligible."""
    with pytest.raises(HardModeValidationError):
        validate_candidate(_candidate(question_type="comparison_tiebreak"), existing_questions=[])


@pytest.mark.parametrize("field", ["question", "answer", "verification_query"])
def test_validate_candidate_rejects_blank_required_text(field: str) -> None:
    """Question, answer, and verification SQL must all be non-blank strings."""
    with pytest.raises(HardModeValidationError):
        validate_candidate(_candidate(**{field: "  "}), existing_questions=[])


@pytest.mark.parametrize(
    "required_files",
    [
        [f"record-{index}.txt" for index in range(5)],
        ["same.txt"] * 6,
    ],
)
def test_validate_candidate_requires_six_distinct_files(
    required_files: list[str],
) -> None:
    """Candidates need six distinct source files to force multi-hop reasoning."""
    with pytest.raises(HardModeValidationError):
        validate_candidate(_candidate(required_files=required_files), existing_questions=[])


@pytest.mark.parametrize(
    "required_files",
    [
        [
            "record-0.txt",
            "record-1.txt",
            "record-2.txt",
            "record-3.txt",
            "record-4.txt",
            "nested/record-5.txt",
        ],
        [
            "record-0.txt",
            "record-1.txt",
            "record-2.txt",
            "record-3.txt",
            "record-4.txt",
            "nested\\record-5.txt",
        ],
        [
            "record-0.txt",
            "record-1.txt",
            "record-2.txt",
            "record-3.txt",
            "record-4.txt",
            "record-5.csv",
        ],
    ],
)
def test_validate_candidate_rejects_non_basename_text_files(
    required_files: list[str],
) -> None:
    """Generated file references are names only and are never traversed."""
    with pytest.raises(HardModeValidationError):
        validate_candidate(_candidate(required_files=required_files), existing_questions=[])


def test_validate_candidate_requires_six_reasoning_queries() -> None:
    """At least six reasoning queries demonstrate the requested depth."""
    with pytest.raises(HardModeValidationError):
        validate_candidate(
            _candidate(
                sql_queries=[
                    {"description": "Read the candidate value", "query": "SELECT 1"}
                    for _ in range(5)
                ]
            ),
            existing_questions=[],
        )


@pytest.mark.parametrize(
    "sql_queries",
    [
        ["SELECT 1"] * 6,
        [{"description": "Read the candidate value", "query": ""}] * 6,
    ],
)
def test_validate_candidate_requires_mapped_nonempty_queries(
    sql_queries: list[object],
) -> None:
    """Each reasoning entry must expose a usable SQL query field."""
    with pytest.raises(HardModeValidationError):
        validate_candidate(_candidate(sql_queries=sql_queries), existing_questions=[])


@pytest.mark.parametrize(
    "sql_query",
    [
        {"query": "SELECT 1"},
        {"description": " ", "query": "SELECT 1"},
    ],
)
def test_validate_candidate_requires_nonempty_reasoning_query_descriptions(
    sql_query: dict[str, str],
) -> None:
    """Every raw Letta reasoning query retains a nonempty audit description."""
    with pytest.raises(HardModeValidationError, match="description"):
        validate_candidate(
            _candidate(sql_queries=[sql_query] * MIN_REASONING_QUERIES),
            existing_questions=[],
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"verification_query": "PRAGMA user_version"},
        {
            "sql_queries": [
                {"description": "Read value one", "query": "SELECT 1"},
                {"description": "Read value two", "query": "SELECT 1"},
                {"description": "Read value three", "query": "SELECT 1"},
                {"description": "Read value four", "query": "SELECT 1"},
                {"description": "Read value five", "query": "SELECT 1"},
                {"description": "Attempt mutation", "query": "DELETE FROM results"},
            ]
        },
    ],
)
def test_validate_candidate_validates_every_sql_statement(
    updates: dict[str, object],
) -> None:
    """Both the final and intermediate generated SQL must pass the query gate."""
    with pytest.raises(HardModeValidationError):
        validate_candidate(_candidate(**updates), existing_questions=[])


def test_validate_candidate_rejects_a_normalized_duplicate_question() -> None:
    """Case and whitespace changes cannot evade exact duplicate detection."""
    with pytest.raises(HardModeValidationError):
        validate_candidate(
            _candidate(question="  WHAT   is total revenue?  "),
            existing_questions=["what is total revenue?"],
        )


def test_validate_candidate_rejects_jaccard_duplicate_at_threshold() -> None:
    """The near-duplicate rule rejects a Jaccard token score of exactly 0.8."""
    with pytest.raises(HardModeValidationError):
        validate_candidate(
            _candidate(question="What is total revenue"),
            existing_questions=["What is total revenue amount"],
        )


def test_validate_candidate_accepts_jaccard_similarity_below_threshold() -> None:
    """A 7/9 token overlap remains below the near-duplicate threshold."""
    validate_candidate(
        _candidate(question="Which customer has highest total sales revenue"),
        existing_questions=["Which customer has highest total sales revenue this year"],
    )


def test_verify_answer_accepts_case_insensitive_exact_string(tmp_path: Path) -> None:
    """Text answers compare case-insensitively after a one-cell query."""
    verify_answer(_database(tmp_path), "SELECT label FROM results WHERE id = 1", "aLiCe")


def test_verify_answer_accepts_numeric_result_within_tolerance(tmp_path: Path) -> None:
    """Numeric answers allow the approved absolute tolerance of one hundredth."""
    verify_answer(_database(tmp_path), "SELECT value FROM results WHERE id = 1", "7.009")


def test_verify_answer_accepts_numeric_result_at_inclusive_tolerance(tmp_path: Path) -> None:
    """A difference of exactly one hundredth remains an accepted match."""
    verify_answer(_database(tmp_path), "SELECT value FROM results WHERE id = 1", "7.01")


def test_answers_match_rejects_large_decimal_difference_just_over_tolerance() -> None:
    """High-precision Decimal results must not round down to the tolerance."""
    assert not context_hard_pilot._answers_match(  # noqa: SLF001  # test exact private helper
        Decimal("9223372036854775807.0100000000000000000000000000001"),
        "9223372036854775807",
    )


def test_verify_answer_rejects_large_difference_just_over_tolerance(tmp_path: Path) -> None:
    """A SQLite integer result must retain decimal digits beyond the threshold."""
    with pytest.raises(HardModeValidationError):
        verify_answer(
            _database(tmp_path),
            "SELECT value FROM results WHERE id = 4",
            "9223372036854775806.9899999999999999999999999999999",
        )


@pytest.mark.parametrize("answer", ["NaN", "Infinity", "-Infinity", "not-a-number"])
def test_verify_answer_rejects_nonfinite_or_unparseable_numeric_answer(
    tmp_path: Path, answer: str
) -> None:
    """Numeric answers must be finite, parseable decimal values."""
    with pytest.raises(HardModeValidationError):
        verify_answer(_database(tmp_path), "SELECT value FROM results WHERE id = 1", answer)


def test_verify_answer_rejects_nonfinite_numeric_result(tmp_path: Path) -> None:
    """A non-finite SQLite numeric result cannot verify a candidate answer."""
    with pytest.raises(HardModeValidationError):
        verify_answer(_database(tmp_path), "SELECT value * 1e999 FROM results WHERE id = 1", "7")


def test_verify_answer_rejects_zero_rows(tmp_path: Path) -> None:
    """A scalar query must return a row rather than an empty result set."""
    with pytest.raises(HardModeValidationError):
        verify_answer(_database(tmp_path), "SELECT value FROM results WHERE id = 404", "0")


def test_verify_answer_rejects_adjacent_large_integers(tmp_path: Path) -> None:
    """Adjacent integers beyond the float exact range must not compare equal."""
    with pytest.raises(HardModeValidationError):
        verify_answer(
            _database(tmp_path),
            "SELECT value FROM results WHERE id = 5",
            "9007199254740992",
        )


@pytest.mark.parametrize(
    ("query", "answer"),
    [
        ("SELECT label FROM results WHERE id = 1", "Bob"),
        ("SELECT value FROM results WHERE id = 1", "7.011"),
        ("SELECT value FROM results", "7"),
        ("SELECT value, value FROM results WHERE id = 1", "7"),
    ],
)
def test_verify_answer_rejects_mismatched_or_non_scalar_results(
    tmp_path: Path, query: str, answer: str
) -> None:
    """Verification requires one matching scalar result, not a partial answer."""
    with pytest.raises(HardModeValidationError):
        verify_answer(_database(tmp_path), query, answer)


def test_verify_answer_authorizer_denies_pragma_after_validator_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SQLite still denies a PRAGMA if the outer SQL validator regresses."""

    def allow_any_query(query: str) -> None:
        """Bypass only the first defense layer to exercise the authorizer."""
        del query

    monkeypatch.setattr(context_hard_pilot, "_validate_verification_query", allow_any_query)
    monkeypatch.setattr(
        context_hard_pilot,
        "_validate_verification_projection",
        allow_any_query,
        raising=False,
    )

    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        verify_answer(_database(tmp_path), "PRAGMA user_version", "0")


@pytest.mark.parametrize(
    "query",
    [
        "INSERT INTO results (id, value) VALUES (3, 11)",
        "CREATE TABLE blocked (value INTEGER)",
        "ATTACH DATABASE 'other.sqlite' AS other",
    ],
)
def test_verify_answer_authorizer_denies_write_schema_and_attach_after_validator_bypass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    """The SQLite authorizer denies writes, schema changes, and attaches."""

    def allow_any_query(query: str) -> None:
        """Bypass only lexical validation to exercise the authorizer."""
        del query

    monkeypatch.setattr(context_hard_pilot, "_validate_verification_query", allow_any_query)
    monkeypatch.setattr(
        context_hard_pilot,
        "_validate_verification_projection",
        allow_any_query,
        raising=False,
    )

    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        verify_answer(_database(tmp_path), query, "0")


def test_verify_answer_readonly_connection_denies_write_when_authorizer_allows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read-only URI mode remains a final defense if the authorizer regresses."""

    def allow_any_query(query: str) -> None:
        """Bypass only lexical validation to reach SQLite's read-only mode."""
        del query

    def allow_all_operations(
        action: int,
        argument_one: str | None,
        argument_two: str | None,
        database_name: str | None,
        trigger_name: str | None,
    ) -> int:
        """Permit every authorizer action to exercise the read-only connection."""
        del action, argument_one, argument_two, database_name, trigger_name
        return sqlite3.SQLITE_OK

    monkeypatch.setattr(context_hard_pilot, "_validate_verification_query", allow_any_query)
    monkeypatch.setattr(
        context_hard_pilot,
        "_validate_verification_projection",
        allow_any_query,
        raising=False,
    )
    monkeypatch.setattr(context_hard_pilot, "_sqlite_authorizer", allow_all_operations)

    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        verify_answer(_database(tmp_path), "INSERT INTO results (id, value) VALUES (3, 11)", "0")
