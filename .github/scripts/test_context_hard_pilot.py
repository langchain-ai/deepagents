"""Tests for the Context-Hard candidate validation helpers."""

from __future__ import annotations

import sqlite3
from decimal import Decimal
from typing import TYPE_CHECKING

import context_hard_pilot
import pytest
from context_hard_pilot import (
    HARD_TYPES,
    MIN_REASONING_QUERIES,
    MIN_REQUIRED_FILES,
    HardModeValidationError,
    validate_candidate,
    validate_readonly_sql,
    verify_answer,
)

if TYPE_CHECKING:
    from pathlib import Path


def _candidate(**updates: object) -> dict[str, object]:
    """Build a valid candidate, allowing each test to override one field."""
    candidate: dict[str, object] = {
        "question_type": "multi_hop_chain",
        "question": "Which account has the highest verified balance?",
        "answer": "Alice",
        "verification_query": "SELECT 'Alice'",
        "required_files": [f"record-{index}.txt" for index in range(6)],
        "sql_queries": [{"query": "SELECT 1"} for _ in range(6)],
    }
    candidate.update(updates)
    return candidate


def _database(tmp_path: Path) -> Path:
    """Create a controlled SQLite database for verification tests."""
    database_path = tmp_path / "answers.sqlite"
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("CREATE TABLE results (id INTEGER PRIMARY KEY, value INTEGER)")
        connection.executemany(
            "INSERT INTO results (id, value) VALUES (?, ?)",
            [(1, 7), (2, 9)],
        )
        connection.commit()
    finally:
        connection.close()
    return database_path


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

        def execute(self, query: str) -> FakeCursor:
            """Return the fixed cursor for the fixed test query."""
            if query != "SELECT 1":
                msg = "unexpected verification query"
                raise AssertionError(msg)
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

    verify_answer(tmp_path / "answers.sqlite", "SELECT 1", "1")

    assert cursor.fetchmany_sizes == [2]
    assert connection.closed


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
            _candidate(sql_queries=[{"query": "SELECT 1"} for _ in range(5)]),
            existing_questions=[],
        )


@pytest.mark.parametrize(
    "sql_queries",
    [
        ["SELECT 1"] * 6,
        [{"query": ""}] * 6,
    ],
)
def test_validate_candidate_requires_mapped_nonempty_queries(
    sql_queries: list[object],
) -> None:
    """Each reasoning entry must expose a usable SQL query field."""
    with pytest.raises(HardModeValidationError):
        validate_candidate(_candidate(sql_queries=sql_queries), existing_questions=[])


@pytest.mark.parametrize(
    "updates",
    [
        {"verification_query": "PRAGMA user_version"},
        {
            "sql_queries": [
                {"query": "SELECT 1"},
                {"query": "SELECT 1"},
                {"query": "SELECT 1"},
                {"query": "SELECT 1"},
                {"query": "SELECT 1"},
                {"query": "DELETE FROM results"},
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
    verify_answer(_database(tmp_path), "SELECT 'Alice'", "aLiCe")


def test_verify_answer_accepts_numeric_result_within_tolerance(tmp_path: Path) -> None:
    """Numeric answers allow the approved absolute tolerance of one hundredth."""
    verify_answer(_database(tmp_path), "SELECT 7.0", "7.009")


def test_verify_answer_accepts_numeric_result_at_inclusive_tolerance(tmp_path: Path) -> None:
    """A difference of exactly one hundredth remains an accepted match."""
    verify_answer(_database(tmp_path), "SELECT 1.0", "1.01")


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
            "SELECT 9223372036854775807",
            "9223372036854775806.9899999999999999999999999999999",
        )


@pytest.mark.parametrize("answer", ["NaN", "Infinity", "-Infinity", "not-a-number"])
def test_verify_answer_rejects_nonfinite_or_unparseable_numeric_answer(
    tmp_path: Path, answer: str
) -> None:
    """Numeric answers must be finite, parseable decimal values."""
    with pytest.raises(HardModeValidationError):
        verify_answer(_database(tmp_path), "SELECT 7.0", answer)


def test_verify_answer_rejects_nonfinite_numeric_result(tmp_path: Path) -> None:
    """A non-finite SQLite numeric result cannot verify a candidate answer."""
    with pytest.raises(HardModeValidationError):
        verify_answer(_database(tmp_path), "SELECT 1e999", "7")


def test_verify_answer_rejects_zero_rows(tmp_path: Path) -> None:
    """A scalar query must return a row rather than an empty result set."""
    with pytest.raises(HardModeValidationError):
        verify_answer(_database(tmp_path), "SELECT value FROM results WHERE id = 404", "0")


def test_verify_answer_rejects_adjacent_large_integers(tmp_path: Path) -> None:
    """Adjacent integers beyond the float exact range must not compare equal."""
    with pytest.raises(HardModeValidationError):
        verify_answer(_database(tmp_path), "SELECT 9007199254740993", "9007199254740992")


@pytest.mark.parametrize(
    ("query", "answer"),
    [
        ("SELECT 'Alice'", "Bob"),
        ("SELECT 7.0", "7.011"),
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

    monkeypatch.setattr(context_hard_pilot, "validate_readonly_sql", allow_any_query)

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

    monkeypatch.setattr(context_hard_pilot, "validate_readonly_sql", allow_any_query)

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

    monkeypatch.setattr(context_hard_pilot, "validate_readonly_sql", allow_any_query)
    monkeypatch.setattr(context_hard_pilot, "_sqlite_authorizer", allow_all_operations)

    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        verify_answer(_database(tmp_path), "INSERT INTO results (id, value) VALUES (3, 11)", "0")
