"""Validate and verify Deep Agents Context-Hard pilot candidates."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Mapping
from decimal import Context, Decimal, DecimalException, localcontext
from pathlib import Path, PurePosixPath, PureWindowsPath

HARD_TYPES = frozenset({"multi_hop_chain", "multi_entity_comparison"})
MIN_REQUIRED_FILES = 6
MIN_REASONING_QUERIES = 6
_JACCARD_SIMILARITY_THRESHOLD = 0.8
_NUMERIC_ABSOLUTE_TOLERANCE = Decimal("0.01")

_READONLY_SQL_START = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)
_FORBIDDEN_SQL = re.compile(
    r"\b(?:ATTACH|DETACH|PRAGMA|VACUUM|LOAD_EXTENSION|"
    r"INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|REPLACE)\b",
    re.IGNORECASE,
)
_QUESTION_TOKEN = re.compile(r"\w+")
_DENIED_SQLITE_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_ATTACH,
        sqlite3.SQLITE_DETACH,
        sqlite3.SQLITE_PRAGMA,
        sqlite3.SQLITE_INSERT,
        sqlite3.SQLITE_UPDATE,
        sqlite3.SQLITE_DELETE,
        sqlite3.SQLITE_CREATE_INDEX,
        sqlite3.SQLITE_CREATE_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_INDEX,
        sqlite3.SQLITE_CREATE_TEMP_TABLE,
        sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
        sqlite3.SQLITE_CREATE_TEMP_VIEW,
        sqlite3.SQLITE_CREATE_TRIGGER,
        sqlite3.SQLITE_CREATE_VIEW,
        sqlite3.SQLITE_CREATE_VTABLE,
        sqlite3.SQLITE_DROP_INDEX,
        sqlite3.SQLITE_DROP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_INDEX,
        sqlite3.SQLITE_DROP_TEMP_TABLE,
        sqlite3.SQLITE_DROP_TEMP_TRIGGER,
        sqlite3.SQLITE_DROP_TEMP_VIEW,
        sqlite3.SQLITE_DROP_TRIGGER,
        sqlite3.SQLITE_DROP_VIEW,
        sqlite3.SQLITE_DROP_VTABLE,
        sqlite3.SQLITE_ALTER_TABLE,
        sqlite3.SQLITE_REINDEX,
        sqlite3.SQLITE_ANALYZE,
        sqlite3.SQLITE_TRANSACTION,
        sqlite3.SQLITE_SAVEPOINT,
    }
)


class HardModeValidationError(ValueError):
    """Raised when a candidate does not satisfy the hard-mode policy."""


def validate_readonly_sql(query: str) -> None:
    """Ensure SQL is a single read-only SELECT or WITH query.

    Args:
        query: SQL submitted by the candidate generator.

    Raises:
        HardModeValidationError: If `query` can perform a non-read-only operation.
    """
    if not isinstance(query, str) or not query.strip():
        msg = "SQL query must be a non-empty string"
        raise HardModeValidationError(msg)
    if ";" in query:
        msg = "SQL query must not contain a statement separator"
        raise HardModeValidationError(msg)
    if not _READONLY_SQL_START.match(query):
        msg = "SQL query must start with SELECT or WITH"
        raise HardModeValidationError(msg)
    if _FORBIDDEN_SQL.search(query):
        msg = "SQL query contains a forbidden operation"
        raise HardModeValidationError(msg)


def validate_candidate(candidate: dict[str, object], *, existing_questions: list[str]) -> None:
    """Validate the deterministic hard-mode requirements for a candidate.

    Args:
        candidate: Generated candidate data to validate without executing paths or SQL.
        existing_questions: Questions from accepted or source candidates to compare against.

    Raises:
        HardModeValidationError: If the candidate fails hard-mode validation.
    """
    question_type = candidate.get("question_type")
    if question_type not in HARD_TYPES:
        msg = f"question_type must be one of {sorted(HARD_TYPES)}"
        raise HardModeValidationError(msg)

    question = _require_nonempty_string(candidate, "question")
    _require_nonempty_string(candidate, "answer")
    verification_query = _require_nonempty_string(candidate, "verification_query")
    validate_readonly_sql(verification_query)

    required_files = _require_list(candidate, "required_files")
    if len(required_files) < MIN_REQUIRED_FILES:
        msg = f"required_files must contain at least {MIN_REQUIRED_FILES} entries"
        raise HardModeValidationError(msg)
    if not all(_is_basename_txt_file(value) for value in required_files):
        msg = "required_files must contain basename-only .txt files"
        raise HardModeValidationError(msg)
    if len(set(required_files)) < MIN_REQUIRED_FILES:
        msg = f"required_files must contain {MIN_REQUIRED_FILES} distinct files"
        raise HardModeValidationError(msg)

    sql_queries = _require_list(candidate, "sql_queries")
    if len(sql_queries) < MIN_REASONING_QUERIES:
        msg = f"sql_queries must contain at least {MIN_REASONING_QUERIES} entries"
        raise HardModeValidationError(msg)
    for sql_query in sql_queries:
        if not isinstance(sql_query, Mapping):
            msg = "each sql_queries entry must be a mapping"
            raise HardModeValidationError(msg)
        query = sql_query.get("query")
        if not isinstance(query, str) or not query.strip():
            msg = "each sql_queries entry must contain a non-empty query string"
            raise HardModeValidationError(msg)
        validate_readonly_sql(query)

    _validate_question_is_distinct(question, existing_questions)


def verify_answer(database_path: Path, verification_query: str, answer: str) -> None:
    """Verify a candidate answer through a read-only SQLite query.

    Args:
        database_path: Trusted SQLite database containing the source fixture.
        verification_query: Candidate query that derives the final answer.
        answer: Candidate answer to compare with the scalar query result.

    Raises:
        HardModeValidationError: If the query result is not one matching scalar.
    """
    validate_readonly_sql(verification_query)
    if not isinstance(answer, str):
        msg = "answer must be a string"
        raise HardModeValidationError(msg)

    database_uri = f"{database_path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    try:
        connection.set_authorizer(_sqlite_authorizer)
        rows = connection.execute(verification_query).fetchmany(2)
    finally:
        connection.close()

    if len(rows) != 1 or len(rows[0]) != 1:
        msg = "verification query must return exactly one row and one column"
        raise HardModeValidationError(msg)
    if not _answers_match(rows[0][0], answer):
        msg = "verification query result does not match the candidate answer"
        raise HardModeValidationError(msg)


def _require_nonempty_string(candidate: Mapping[str, object], field: str) -> str:
    """Return a required non-empty string field from a candidate."""
    value = candidate.get(field)
    if not isinstance(value, str) or not value.strip():
        msg = f"{field} must be a non-empty string"
        raise HardModeValidationError(msg)
    return value


def _require_list(candidate: Mapping[str, object], field: str) -> list[object]:
    """Return a required list field from a candidate."""
    value = candidate.get(field)
    if not isinstance(value, list):
        msg = f"{field} must be a list"
        raise HardModeValidationError(msg)
    return value


def _is_basename_txt_file(value: object) -> bool:
    """Return whether a value is a cross-platform basename-only `.txt` file."""
    if not isinstance(value, str):
        return False

    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    return (
        value == posix_path.name == windows_path.name
        and posix_path.suffix == ".txt"
        and windows_path.suffix == ".txt"
        and bool(posix_path.stem)
        and bool(windows_path.stem)
    )


def _validate_question_is_distinct(question: str, existing_questions: list[str]) -> None:
    """Reject exact or token-similar questions from an existing question set."""
    normalized_question = _normalize_question(question)
    candidate_tokens = _question_tokens(normalized_question)
    for existing_question in existing_questions:
        normalized_existing = _normalize_question(existing_question)
        if normalized_question == normalized_existing:
            msg = "question duplicates an existing question"
            raise HardModeValidationError(msg)
        existing_tokens = _question_tokens(normalized_existing)
        if _jaccard_similarity(candidate_tokens, existing_tokens) >= _JACCARD_SIMILARITY_THRESHOLD:
            msg = "question is too similar to an existing question"
            raise HardModeValidationError(msg)


def _normalize_question(question: str) -> str:
    """Normalize case and whitespace for deterministic duplicate checks."""
    return " ".join(question.casefold().split())


def _question_tokens(question: str) -> set[str]:
    """Extract normalized word tokens for the Jaccard similarity check."""
    return set(_QUESTION_TOKEN.findall(question.casefold()))


def _jaccard_similarity(left: set[str], right: set[str]) -> float:
    """Return token-set Jaccard similarity, avoiding an empty-set match."""
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _sqlite_authorizer(
    action: int,
    argument_one: str | None,
    argument_two: str | None,
    database_name: str | None,
    trigger_name: str | None,
) -> int:
    """Deny write, schema, attach, pragma, and extension SQL operations."""
    del database_name, trigger_name
    if action in _DENIED_SQLITE_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and any(
        argument is not None and argument.casefold() == "load_extension"
        for argument in (argument_one, argument_two)
    ):
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def _answers_match(result: object, answer: str) -> bool:
    """Compare a scalar SQLite result with the candidate answer."""
    if isinstance(result, str):
        return result.casefold() == answer.casefold()
    if type(result) not in {int, float, Decimal}:
        return False
    try:
        result_decimal = Decimal(str(result))
        answer_decimal = Decimal(answer)
        if not result_decimal.is_finite() or not answer_decimal.is_finite():
            return False
        comparison_context = _decimal_comparison_context(result_decimal, answer_decimal)
        with localcontext(comparison_context):
            difference = abs(result_decimal - answer_decimal)
    except (DecimalException, ValueError):
        return False
    return difference <= _NUMERIC_ABSOLUTE_TOLERANCE


def _decimal_comparison_context(left: Decimal, right: Decimal) -> Context:
    """Build a context that preserves an exact finite subtraction."""
    values = (left, right)
    minimum_exponent = min(value.as_tuple().exponent for value in values)
    precision = (
        max(
            len(value.as_tuple().digits) + value.as_tuple().exponent - minimum_exponent
            for value in values
        )
        + 1
    )
    maximum_adjusted_exponent = max(value.adjusted() for value in values)
    return Context(
        prec=precision,
        Emax=max(0, maximum_adjusted_exponent + 1),
        Emin=min(0, minimum_exponent),
    )
