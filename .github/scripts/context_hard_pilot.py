"""Validate and verify Deep Agents Context-Hard pilot candidates."""

from __future__ import annotations

import argparse
import base64
import importlib
import importlib.machinery
import importlib.util
import json
import math
import re
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from decimal import Context, Decimal, DecimalException, localcontext
from pathlib import Path, PurePosixPath, PureWindowsPath

HARD_TYPES = frozenset({"multi_hop_chain", "multi_entity_comparison"})
MIN_REQUIRED_FILES = 6
MIN_REASONING_QUERIES = 6
MAX_REASONING_QUERIES = 12
MAX_REASONING_QUERY_BYTES = 8 * 1024
MAX_REASONING_DESCRIPTION_BYTES = 2 * 1024
MAX_RAW_JSONL_BYTES = 64 * 1024
MAX_VERIFICATION_QUERY_BYTES = 8 * 1024
LETTA_REPOSITORY = "https://github.com/letta-ai/letta-evals.git"
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_JACCARD_SIMILARITY_THRESHOLD = 0.8
_NUMERIC_ABSOLUTE_TOLERANCE = Decimal("0.01")
_GIT_TIMEOUT_SECONDS = 60
_RESULT_ROW_LIMIT = 2
_SQLITE_MAX_STRING_OR_BLOB_BYTES = 16 * 1024
_SQLITE_MAX_RESULT_COLUMNS = 16
_MAX_SERIALIZED_RESULT_BYTES = 1024
_MAX_TOOL_ERROR_BYTES = 512
_TOOL_ERROR_TRUNCATION_MARKER = "... [truncated]"
_SQLITE_PROGRESS_HANDLER_OPCODES = 100
_SQLITE_MAX_VM_STEPS = 10_000
_SQLITE_MAX_QUERY_SECONDS = 0.1
_RAW_OUTPUT_NAME = "agent_generated_questions.jsonl"
_PARSED_OUTPUT_NAME = "agent_generated_questions_parsed.jsonl"
_HARD_MODE_PROMPT = """
Hard-mode requirements: use at least six distinct source files and at least six SQL
steps. Build two sequential, indirect dependency chains that converge in a final
comparison or tiebreak. The answer must be one unique, concrete value. Independent
parallel greps or unrelated lookups do not satisfy this requirement.
The verification query must start with SELECT and use nested subqueries, not CTEs or derived tables.
""".strip()

Run = Callable[..., subprocess.CompletedProcess[str]]

_TRUSTED_MODULE_ORDER = (
    "display",
    "context",
    "parallel",
    "tools",
    "tools.register_question_tool",
    "tools.sql_execute_tool",
    "question_generator",
)

_READONLY_SQL_START = re.compile(r"^\s*(?:SELECT|WITH)\b", re.IGNORECASE)
_VERIFICATION_SQL_START = re.compile(r"^\s*SELECT\b", re.IGNORECASE)
_HARDCODED_CASE_VALUE = re.compile(r"\bCASE\b.*\bTHEN\s+['\"]", re.IGNORECASE | re.DOTALL)
_SQL_IDENTIFIER = r'(?:[A-Za-z_][A-Za-z0-9_]*|"(?:[^"]|"")*"|`(?:[^`]|``)*`|\[[^\]]+\])'
_SOURCE_REFERENCE = re.compile(
    rf"(?:{_SQL_IDENTIFIER}\s*\.\s*)*(?:{_SQL_IDENTIFIER}|\*)",
    re.IGNORECASE,
)
_SOURCE_AGGREGATE = re.compile(
    r"(?P<name>COUNT|SUM|AVG|MIN|MAX|TOTAL|GROUP_CONCAT)\s*"
    r"\((?P<argument>.*)\)",
    re.IGNORECASE | re.DOTALL,
)
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


def checkout_letta_generator(
    revision: str, destination: Path, *, run: Run = subprocess.run
) -> Path:
    """Clone and detach Letta's generator at one immutable commit SHA.

    Args:
        revision: Lowercase 40-character Git commit SHA to check out.
        destination: New local checkout directory.
        run: Process runner injected by tests.

    Returns:
        The checked-out generator directory.

    Raises:
        HardModeValidationError: If `revision` is not pinned or resolves differently.
        subprocess.CalledProcessError: If a Git command fails.
    """
    if not REVISION_RE.fullmatch(revision):
        msg = "revision must be a 40-character commit SHA in lowercase hexadecimal"
        raise HardModeValidationError(msg)

    process_options = {
        "check": True,
        "capture_output": True,
        "text": True,
        "timeout": _GIT_TIMEOUT_SECONDS,
        "shell": False,
    }
    run(
        ["git", "clone", "--no-checkout", LETTA_REPOSITORY, str(destination)],
        **process_options,
    )
    run(
        ["git", "-C", str(destination), "checkout", "--detach", revision],
        **process_options,
    )
    resolved_head = run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        **process_options,
    ).stdout.strip()
    if resolved_head != revision:
        msg = "checked-out revision does not match the requested commit SHA"
        raise HardModeValidationError(msg)
    generation_directory = destination / "letta-leaderboard" / "filesystem-agent" / "generation"
    if not generation_directory.is_dir():
        msg = "checked-out generation directory does not exist"
        raise HardModeValidationError(msg)
    return generation_directory


class ReadOnlySQLTool:
    """Execute model-facing SQL only through the local read-only policy."""

    def __init__(self, database_path: Path) -> None:
        """Store the trusted SQLite database path.

        Args:
            database_path: SQLite database copied into the trusted generation area.
        """
        self.database_path = database_path

    def execute(self, query: str) -> dict[str, object]:
        """Execute one bounded, read-only query and return a JSON-safe result.

        Args:
            query: Model-provided SQL to validate and execute.

        Returns:
            A Letta-compatible result dictionary. Errors are returned as data.
        """
        started_at = time.perf_counter()
        connection: sqlite3.Connection | None = None
        try:
            validate_readonly_sql(query)
            database_uri = f"{self.database_path.resolve().as_uri()}?mode=ro"
            connection = sqlite3.connect(database_uri, uri=True)
            connection.set_authorizer(_sqlite_authorizer)
            connection.setlimit(
                sqlite3.SQLITE_LIMIT_LENGTH,
                _SQLITE_MAX_STRING_OR_BLOB_BYTES,
            )
            connection.setlimit(
                sqlite3.SQLITE_LIMIT_COLUMN,
                _SQLITE_MAX_RESULT_COLUMNS,
            )
            _install_sqlite_resource_guard(connection)
            rows = connection.execute(query).fetchmany(_RESULT_ROW_LIMIT)
            result = [_json_safe_row(row) for row in rows]
            _validate_serialized_result_size(result)
            return _tool_result(True, result, len(result), started_at, None)
        except (HardModeValidationError, sqlite3.Error, TypeError, ValueError) as error:
            return _tool_result(False, [], 0, started_at, str(error))
        finally:
            if connection is not None:
                connection.close()


class HardModeRegisterTool:
    """Verify and write hard-mode candidates without upstream write access."""

    def __init__(
        self, output_path: Path, database_path: Path, existing_questions: list[str]
    ) -> None:
        """Configure local output and validation inputs.

        Args:
            output_path: Raw JSONL destination for accepted records.
            database_path: Trusted SQLite database used to verify answers.
            existing_questions: Questions that new candidates must not duplicate.
        """
        self.output_path = output_path
        self.database_path = database_path
        self.existing_questions = existing_questions
        self.sql_tool = ReadOnlySQLTool(database_path)
        self.registered_count = 0

    def register(
        self,
        question: str,
        sql_queries: list[dict[str, str]],
        answer: str,
        answer_reasoning: str,
        question_type: str,
        required_files: list[str],
        verification_query: str = "",
    ) -> dict[str, object]:
        """Validate, verify, and append one locally controlled candidate.

        Args:
            question: Candidate question text.
            sql_queries: The model's SQL reasoning steps.
            answer: Candidate's concrete answer.
            answer_reasoning: Audit explanation from the model.
            question_type: Requested hard-mode category.
            required_files: Basename-only source files required by the question.
            verification_query: Scalar read-only SQL used to verify `answer`.

        Returns:
            A model-facing success or error dictionary without raising validation errors.
        """
        candidate: dict[str, object] = {
            "question": question,
            "sql_queries": sql_queries,
            "answer": answer,
            "question_type": question_type,
            "required_files": required_files,
            "verification_query": verification_query,
        }
        try:
            _validate_answer_reasoning(answer_reasoning)
            validate_candidate(candidate, existing_questions=self.existing_questions)
            preflight_raw_line = (
                json.dumps({**candidate, "answer_reasoning": answer_reasoning}) + "\n"
            )
            _validate_raw_jsonl_size(preflight_raw_line)
            verify_answer(self.database_path, verification_query, answer)
            query_results, all_queries_succeeded = self._execute_reasoning_queries(sql_queries)
            _require_successful_reasoning_queries(all_queries_succeeded)
            raw_record = {
                "question": question,
                "answer": answer,
                "difficulty": "hard",
                "question_type": question_type,
                "required_files": required_files,
                "answer_reasoning": answer_reasoning,
                "sql_queries": query_results,
                "verification_query": verification_query,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            parsed_record = {
                "input": question,
                "ground_truth": answer,
                "agent_args": {
                    "tags": [],
                    "extra": {
                        "required_files": required_files,
                        "question_type": question_type,
                        "difficulty": "hard",
                        "verification_query": verification_query,
                    },
                },
            }
            raw_line = json.dumps(raw_record) + "\n"
            parsed_line = json.dumps(parsed_record) + "\n"
            _validate_raw_jsonl_size(raw_line)
            self._append_records(raw_line, parsed_line)
        except (HardModeValidationError, OSError, TypeError, ValueError, sqlite3.Error) as error:
            return {"success": False, "error": str(error)}
        self.existing_questions.append(question)
        self.registered_count += 1
        return {
            "success": True,
            "message": f"Question registered successfully. Answer: {answer}",
            "answer": answer,
            "query_results": query_results,
            "total_questions": self.registered_count,
        }

    def _execute_reasoning_queries(
        self, sql_queries: list[dict[str, str]]
    ) -> tuple[list[dict[str, object]], bool]:
        """Execute validated reasoning SQL through the local read-only adapter."""
        query_results: list[dict[str, object]] = []
        all_succeeded = True
        for sql_query in sql_queries:
            query_result = self.sql_tool.execute(sql_query["query"])
            record: dict[str, object] = {
                "description": sql_query["description"],
                "query": sql_query["query"],
            }
            if query_result["success"] is True:
                record["result"] = query_result["result"]
            else:
                all_succeeded = False
                record["error"] = query_result["error"]
            query_results.append(record)
        return query_results, all_succeeded

    def _append_records(self, raw_line: str, parsed_line: str) -> None:
        """Append paired records and roll back either file if a write fails."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        parsed_path = self.output_path.parent / _PARSED_OUTPUT_NAME
        paths_and_lines = ((self.output_path, raw_line), (parsed_path, parsed_line))
        offsets = {path: path.stat().st_size if path.exists() else 0 for path, _ in paths_and_lines}
        try:
            for path, line in paths_and_lines:
                with path.open("a", encoding="utf-8") as output_file:
                    output_file.write(line)
        except OSError:
            for path, offset in offsets.items():
                if path.exists():
                    with path.open("r+", encoding="utf-8") as output_file:
                        output_file.truncate(offset)
            raise


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
    _validate_utf8_byte_limit(query, MAX_REASONING_QUERY_BYTES, "SQL query")
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
    answer = _require_nonempty_string(candidate, "answer")
    _validate_concrete_answer(answer)
    verification_query = _require_nonempty_string(candidate, "verification_query")
    _validate_verification_query(verification_query)

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
    if len(sql_queries) > MAX_REASONING_QUERIES:
        msg = f"sql_queries must contain at most {MAX_REASONING_QUERIES} entries"
        raise HardModeValidationError(msg)
    for sql_query in sql_queries:
        if not isinstance(sql_query, Mapping):
            msg = "each sql_queries entry must be a mapping"
            raise HardModeValidationError(msg)
        description = sql_query.get("description")
        if not isinstance(description, str) or not description.strip():
            msg = "each sql_queries entry must contain a non-empty description string"
            raise HardModeValidationError(msg)
        _validate_utf8_byte_limit(
            description,
            MAX_REASONING_DESCRIPTION_BYTES,
            "reasoning description",
        )
        query = sql_query.get("query")
        if not isinstance(query, str) or not query.strip():
            msg = "each sql_queries entry must contain a non-empty query string"
            raise HardModeValidationError(msg)
        _validate_utf8_byte_limit(
            query,
            MAX_REASONING_QUERY_BYTES,
            "reasoning query",
        )
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
    _validate_verification_query(verification_query)
    _validate_verification_projection(verification_query)
    if not isinstance(answer, str):
        msg = "answer must be a string"
        raise HardModeValidationError(msg)

    database_uri = f"{database_path.resolve().as_uri()}?mode=ro"
    source_data_read = False

    def recording_authorizer(
        action: int,
        argument_one: str | None,
        argument_two: str | None,
        database_name: str | None,
        trigger_name: str | None,
    ) -> int:
        """Record reads of non-system source tables while retaining deny policy."""
        nonlocal source_data_read
        if (
            action == sqlite3.SQLITE_READ
            and isinstance(argument_one, str)
            and not argument_one.casefold().startswith("sqlite_")
        ):
            source_data_read = True
        return _sqlite_authorizer(
            action,
            argument_one,
            argument_two,
            database_name,
            trigger_name,
        )

    connection = sqlite3.connect(database_uri, uri=True)
    try:
        connection.set_authorizer(recording_authorizer)
        connection.setlimit(
            sqlite3.SQLITE_LIMIT_LENGTH,
            _SQLITE_MAX_STRING_OR_BLOB_BYTES,
        )
        connection.setlimit(
            sqlite3.SQLITE_LIMIT_COLUMN,
            _SQLITE_MAX_RESULT_COLUMNS,
        )
        _install_sqlite_resource_guard(connection)
        rows = connection.execute(verification_query).fetchmany(2)
    finally:
        connection.close()

    if not source_data_read:
        msg = "verification query must read source data from a non-system table"
        raise HardModeValidationError(msg)
    if len(rows) != 1 or len(rows[0]) != 1:
        msg = "verification query must return exactly one row and one column"
        raise HardModeValidationError(msg)
    if not _answers_match(rows[0][0], answer):
        msg = "verification query result does not match the candidate answer"
        raise HardModeValidationError(msg)


def run_upstream_checkout(
    generator_directory: Path,
    database_path: Path,
    output_path: Path,
    model: str,
) -> None:
    """Run the trusted upstream generator with local hard-mode tool adapters.

    Args:
        generator_directory: SHA-pinned Letta generation directory.
        database_path: Trusted copied SQLite database for the generator.
        output_path: Raw local JSONL destination for accepted candidates.
        model: Resolved model identifier passed to Letta's generator.
    """
    resolved_generator_directory = generator_directory.resolve()
    generator_module_path = str(resolved_generator_directory)
    original_path = list(sys.path)
    original_path_importer_cache = dict(sys.path_importer_cache)
    original_dont_write_bytecode = sys.dont_write_bytecode
    original_modules = dict(sys.modules)
    try:
        _remove_upstream_modules()
        sys.path[:] = [
            generator_module_path,
            *(path for path in original_path if path != generator_module_path),
        ]
        sys.dont_write_bytecode = True
        importlib.invalidate_caches()
        trusted_specs = _preflight_trusted_module_specs(resolved_generator_directory)
        trusted_modules = _load_trusted_modules(trusted_specs)
        _validate_trusted_module_origins(resolved_generator_directory)

        question_generator_agent = trusted_modules["question_generator"].QuestionGeneratorAgent
        register_question_tool_schema = trusted_modules[
            "tools.register_question_tool"
        ].REGISTER_QUESTION_TOOL_DICT
        sql_execute_tool_schema = trusted_modules["tools.sql_execute_tool"].EXECUTE_SQL_TOOL_DICT

        if not register_question_tool_schema or not sql_execute_tool_schema:
            msg = "upstream tool schemas must be non-empty"
            raise HardModeValidationError(msg)

        agent = question_generator_agent(
            db_path=database_path,
            output_path=output_path,
            model=model,
        )
        agent.system_prompt = f"{agent.system_prompt}\n\n{_HARD_MODE_PROMPT}"
        for question_type in ("multi_hop_chain", "multi_entity_comparison"):
            agent.type_prompts[question_type] = (
                f"{agent.type_prompts[question_type]}\n\n{_HARD_MODE_PROMPT}"
            )
        existing_questions = _read_existing_questions(output_path)
        existing_record_count = len(existing_questions)
        agent.sql_tool = ReadOnlySQLTool(database_path)
        agent.register_tool = HardModeRegisterTool(
            output_path,
            database_path,
            existing_questions,
        )
        agent.generate_questions(3, "multi_hop_chain")
        agent.generate_questions(2, "multi_entity_comparison")
        new_record_count = len(_read_existing_questions(output_path)) - existing_record_count
        if new_record_count != 5:
            msg = f"expected exactly five new registrations, found {new_record_count}"
            raise HardModeValidationError(msg)
    finally:
        importlib.invalidate_caches()
        sys.modules.clear()
        sys.modules.update(original_modules)
        sys.path[:] = original_path
        sys.path_importer_cache.clear()
        sys.path_importer_cache.update(original_path_importer_cache)
        sys.dont_write_bytecode = original_dont_write_bytecode


def main(argv: list[str] | None = None) -> None:
    """Run the internal pinned-upstream generator mode.

    Args:
        argv: Optional CLI argument list, using process arguments by default.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-upstream-checkout", action="store_true")
    parser.add_argument(
        "--generator-directory",
        "--generation-directory",
        dest="generator_directory",
        type=Path,
    )
    parser.add_argument("--database-path", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--model")
    arguments = parser.parse_args(argv)
    if not arguments.run_upstream_checkout:
        parser.error("--run-upstream-checkout is required")
    required_arguments = (
        arguments.generator_directory,
        arguments.database_path,
        arguments.output_path,
        arguments.model,
    )
    if any(argument is None for argument in required_arguments):
        parser.error(
            "--generator-directory, --database-path, --output-path, and --model are required"
        )
    run_upstream_checkout(
        arguments.generator_directory,
        arguments.database_path,
        arguments.output_path,
        arguments.model,
    )


def _tool_result(
    success: bool,
    result: list[list[object]],
    row_count: int,
    started_at: float,
    error: str | None,
) -> dict[str, object]:
    """Build the bounded model-facing response for a SQL tool invocation."""
    return {
        "success": success,
        "result": result,
        "row_count": row_count,
        "execution_time_ms": (time.perf_counter() - started_at) * 1000,
        "error": _truncate_tool_error(error),
    }


def _json_safe_row(row: tuple[object, ...]) -> list[object]:
    """Convert a SQLite row into values that standard JSON can safely encode."""
    return [_json_safe_value(value) for value in row]


def _json_safe_value(value: object) -> object:
    """Convert a SQLite cell value to a JSON-safe primitive without evaluation."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, memoryview):
        return base64.b64encode(value.tobytes()).decode("ascii")
    return str(value)


def _validate_serialized_result_size(result: list[list[object]]) -> None:
    """Reject model-facing SQL output that exceeds the fixed JSON byte budget."""
    serialized_result = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if len(serialized_result.encode("utf-8")) > _MAX_SERIALIZED_RESULT_BYTES:
        msg = "serialized result exceeds the model-facing byte limit"
        raise HardModeValidationError(msg)


def _validate_raw_jsonl_size(raw_line: str) -> None:
    """Reject a raw audit record that exceeds its fixed aggregate byte budget."""
    if len(raw_line.encode("utf-8")) > MAX_RAW_JSONL_BYTES:
        msg = "raw JSONL record exceeds the aggregate byte limit"
        raise HardModeValidationError(msg)


def _require_successful_reasoning_queries(all_queries_succeeded: bool) -> None:
    """Require every locally executed reasoning query to complete successfully."""
    if not all_queries_succeeded:
        msg = "reasoning query failed"
        raise HardModeValidationError(msg)


def _validate_utf8_byte_limit(value: str, maximum_bytes: int, field: str) -> None:
    """Reject model-supplied text that exceeds a fixed UTF-8 byte limit."""
    if len(value.encode("utf-8")) > maximum_bytes:
        msg = f"{field} exceeds the UTF-8 byte limit"
        raise HardModeValidationError(msg)


def _validate_concrete_answer(answer: str) -> None:
    """Reject answers that are too long, absent, or restate question language."""
    _validate_utf8_byte_limit(answer, 100, "answer")
    normalized_answer = answer.casefold().strip()
    negation_phrases = (
        "does not own",
        "do not own",
        "doesn't own",
        "don't own",
        "no record",
        "no pets",
        "no vehicles",
        "no bank",
        "no credit",
        "no insurance",
        "not found",
        "none",
    )
    if any(phrase in normalized_answer for phrase in negation_phrases):
        msg = "answer must be a concrete value, not a negation or absence"
        raise HardModeValidationError(msg)
    question_phrases = (
        "the person with",
        "among",
        "residents of",
        "who has",
        "who owns",
    )
    if any(phrase in normalized_answer for phrase in question_phrases):
        msg = "answer must be a concrete value, not question text"
        raise HardModeValidationError(msg)


def _validate_verification_query_guardrails(verification_query: str) -> None:
    """Reject verification SQL that encodes an answer or source identifier."""
    if "pers-" in verification_query.casefold():
        msg = "verification query must not contain pers- identifiers"
        raise HardModeValidationError(msg)
    if _HARDCODED_CASE_VALUE.search(verification_query):
        msg = "verification query must not use CASE with a quoted THEN value"
        raise HardModeValidationError(msg)


def _validate_verification_query(verification_query: str) -> None:
    """Reject non-concrete verification SQL before SQLite receives it."""
    validate_readonly_sql(verification_query)
    if not _VERIFICATION_SQL_START.match(verification_query):
        msg = "verification query must start with a top-level SELECT"
        raise HardModeValidationError(msg)
    _validate_utf8_byte_limit(
        verification_query,
        MAX_VERIFICATION_QUERY_BYTES,
        "verification query",
    )
    _validate_verification_query_guardrails(verification_query)


def _validate_verification_projection(verification_query: str) -> None:
    """Require one source-derived scalar in the final top-level SELECT projection."""
    projection = _final_top_level_select_projection(verification_query)
    if '"' in projection:
        msg = "verification query final projection must not contain double quotes"
        raise HardModeValidationError(msg)
    expressions = _split_top_level_projection(projection)
    if len(expressions) != 1 or not _is_source_derived_projection(expressions[0]):
        msg = "verification query final projection must derive from source data"
        raise HardModeValidationError(msg)


def _final_top_level_select_projection(query: str) -> str:
    """Return the projection between the final depth-zero SELECT and FROM keywords."""
    words, _, _ = _scan_sql_structure(query)
    select_spans = [span for span in words if span[0] == "select"]
    if not select_spans:
        msg = "verification query must contain a top-level SELECT projection"
        raise HardModeValidationError(msg)
    _, _, select_end = select_spans[-1]
    from_spans = [span for span in words if span[0] == "from" and span[1] > select_end]
    if from_spans:
        _, from_start, from_end = from_spans[0]
        if _next_final_source_character(query[from_end:]) == "(":
            msg = "verification query must not use a final derived table source"
            raise HardModeValidationError(msg)
        projection_end = from_start
    else:
        projection_end = len(query)
    return query[select_end:projection_end].strip()


def _next_final_source_character(source_text: str) -> str | None:
    """Return the first final-source character after whitespace and SQL comments."""
    index = 0
    while index < len(source_text):
        while index < len(source_text) and source_text[index].isspace():
            index += 1
        if source_text.startswith("/*", index):
            comment_end = source_text.find("*/", index + 2)
            if comment_end == -1:
                msg = "verification query contains an unterminated block comment"
                raise HardModeValidationError(msg)
            index = comment_end + 2
            continue
        if source_text.startswith("--", index):
            while index < len(source_text) and source_text[index] not in {"\r", "\n"}:
                index += 1
            continue
        if index < len(source_text):
            return source_text[index]
    return None


def _split_top_level_projection(projection: str) -> list[str]:
    """Split a SELECT projection only on commas outside quotes and parentheses."""
    _, comma_positions, _ = _scan_sql_structure(projection)
    expressions: list[str] = []
    expression_start = 0
    for comma_position in comma_positions:
        expressions.append(projection[expression_start:comma_position].strip())
        expression_start = comma_position + 1
    expressions.append(projection[expression_start:].strip())
    return expressions


def _is_source_derived_projection(projection: str) -> bool:
    """Return whether a scalar projection is a direct source reference or aggregate."""
    expression = _strip_top_level_projection_alias(projection)
    expression = _strip_enclosing_parentheses(expression)
    if expression.casefold().startswith("distinct "):
        expression = expression[9:].strip()
    if _SOURCE_REFERENCE.fullmatch(expression):
        return True

    aggregate = _SOURCE_AGGREGATE.fullmatch(expression)
    if aggregate is None:
        return False
    aggregate_name = aggregate["name"].casefold()
    argument = _strip_enclosing_parentheses(aggregate["argument"].strip())
    if argument.casefold().startswith("distinct "):
        argument = argument[9:].strip()
    if argument == "*":
        return aggregate_name == "count"
    return _SOURCE_REFERENCE.fullmatch(argument) is not None


def _strip_top_level_projection_alias(projection: str) -> str:
    """Remove one top-level AS alias without treating quoted text as SQL syntax."""
    words, _, _ = _scan_sql_structure(projection)
    aliases = [span for span in words if span[0] == "as"]
    if not aliases:
        return projection.strip()
    _, alias_start, alias_end = aliases[-1]
    alias = projection[alias_end:].strip()
    if _SOURCE_REFERENCE.fullmatch(alias) is None:
        return projection.strip()
    return projection[:alias_start].strip()


def _strip_enclosing_parentheses(expression: str) -> str:
    """Strip balanced parentheses only when they wrap the entire expression."""
    stripped_expression = expression.strip()
    while stripped_expression.startswith("(") and stripped_expression.endswith(")"):
        _, _, closing_parenthesis = _scan_sql_structure(stripped_expression)
        if closing_parenthesis != len(stripped_expression) - 1:
            break
        stripped_expression = stripped_expression[1:-1].strip()
    return stripped_expression


def _scan_sql_structure(
    value: str,
) -> tuple[list[tuple[str, int, int]], list[int], int | None]:
    """Scan SQL text while tracking quotes, parentheses, depth-zero words, and commas."""
    words: list[tuple[str, int, int]] = []
    comma_positions: list[int] = []
    outer_closing_parenthesis: int | None = None
    depth = 0
    quote: str | None = None
    index = 0
    while index < len(value):
        character = value[index]
        if quote is not None:
            if character == quote:
                if quote != "]" and index + 1 < len(value) and value[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        if character in {"'", '"', "`"}:
            quote = character
        elif character == "[":
            quote = "]"
        elif character == "(":
            depth += 1
        elif character == ")":
            if depth > 0:
                depth -= 1
                if value.startswith("(") and depth == 0 and outer_closing_parenthesis is None:
                    outer_closing_parenthesis = index
        elif depth == 0 and character == ",":
            comma_positions.append(index)
        elif depth == 0 and (character.isalpha() or character == "_"):
            word_start = index
            index += 1
            while index < len(value) and (value[index].isalnum() or value[index] == "_"):
                index += 1
            words.append((value[word_start:index].casefold(), word_start, index))
            continue
        index += 1
    return words, comma_positions, outer_closing_parenthesis


def _truncate_tool_error(error: str | None) -> str | None:
    """Bound a tool error to the fixed UTF-8 output budget."""
    if error is None:
        return None
    encoded_error = error.encode("utf-8")
    if len(encoded_error) <= _MAX_TOOL_ERROR_BYTES:
        return error
    marker_bytes = _TOOL_ERROR_TRUNCATION_MARKER.encode("utf-8")
    prefix = encoded_error[: _MAX_TOOL_ERROR_BYTES - len(marker_bytes)].decode(
        "utf-8",
        errors="ignore",
    )
    return f"{prefix}{_TOOL_ERROR_TRUNCATION_MARKER}"


def _is_upstream_module_name(name: str) -> bool:
    """Return whether a module name belongs to the imported Letta generator tree."""
    return name in _TRUSTED_MODULE_ORDER or name.startswith("tools.")


def _remove_upstream_modules() -> None:
    """Evict only the Letta generator modules selected by the trusted import boundary."""
    for name in tuple(sys.modules):
        if _is_upstream_module_name(name):
            sys.modules.pop(name)


def _validate_trusted_module_origins(generator_directory: Path) -> None:
    """Require every direct Letta generator module to come from the checkout."""
    for module_name in _TRUSTED_MODULE_ORDER:
        module = sys.modules.get(module_name)
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str) or not _is_path_within_directory(
            Path(origin), generator_directory
        ):
            msg = f"{module_name} must load from the trusted generator directory"
            raise HardModeValidationError(msg)


def _preflight_trusted_module_specs(
    generator_directory: Path,
) -> dict[str, importlib.machinery.ModuleSpec]:
    """Build direct Letta specs from exact source files inside the checkout."""
    specs: dict[str, importlib.machinery.ModuleSpec] = {}
    for module_name, source_path, is_package in (
        ("display", generator_directory / "display.py", False),
        ("context", generator_directory / "context.py", False),
        ("parallel", generator_directory / "parallel.py", False),
        ("tools", generator_directory / "tools" / "__init__.py", True),
        (
            "tools.register_question_tool",
            generator_directory / "tools" / "register_question_tool.py",
            False,
        ),
        (
            "tools.sql_execute_tool",
            generator_directory / "tools" / "sql_execute_tool.py",
            False,
        ),
        ("question_generator", generator_directory / "question_generator.py", False),
    ):
        if not source_path.is_file() or not _is_path_within_directory(
            source_path, generator_directory
        ):
            msg = f"{module_name} must load from the trusted generator directory"
            raise HardModeValidationError(msg)
        loader = importlib.machinery.SourceFileLoader(module_name, str(source_path))
        submodule_search_locations = [str(source_path.parent)] if is_package else None
        spec = importlib.util.spec_from_file_location(
            module_name,
            str(source_path),
            loader=loader,
            submodule_search_locations=submodule_search_locations,
        )
        origin = spec.origin if spec is not None else None
        if (
            not isinstance(origin, str)
            or spec.loader is not loader
            or not callable(getattr(loader, "exec_module", None))
            or not _is_path_within_directory(Path(origin), generator_directory)
        ):
            msg = f"{module_name} must load from the trusted generator directory"
            raise HardModeValidationError(msg)
        specs[module_name] = spec
    return specs


def _load_trusted_modules(
    specs: Mapping[str, importlib.machinery.ModuleSpec],
) -> dict[str, object]:
    """Load the prevalidated Letta modules without consulting normal import finders."""
    modules: dict[str, object] = {}
    for module_name in _TRUSTED_MODULE_ORDER:
        spec = specs[module_name]
        loader = spec.loader
        if loader is None or not callable(getattr(loader, "exec_module", None)):
            msg = f"{module_name} must load from the trusted generator directory"
            raise HardModeValidationError(msg)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        loader.exec_module(module)
        modules[module_name] = module
    return modules


def _install_sqlite_resource_guard(connection: sqlite3.Connection) -> None:
    """Interrupt SQLite work that exceeds fixed VM-step or elapsed-time budgets."""
    started_at = time.monotonic()
    vm_steps = 0

    def should_interrupt() -> int:
        nonlocal vm_steps
        vm_steps += _SQLITE_PROGRESS_HANDLER_OPCODES
        elapsed_time = time.monotonic() - started_at
        return int(vm_steps >= _SQLITE_MAX_VM_STEPS or elapsed_time >= _SQLITE_MAX_QUERY_SECONDS)

    connection.set_progress_handler(should_interrupt, _SQLITE_PROGRESS_HANDLER_OPCODES)


def _is_path_within_directory(path: Path, directory: Path) -> bool:
    """Return whether a resolved module path is contained by the trusted directory."""
    try:
        path.resolve().relative_to(directory)
    except ValueError:
        return False
    return True


def _read_existing_questions(output_path: Path) -> list[str]:
    """Load previously written local questions without trusting malformed records."""
    if not output_path.exists():
        return []
    questions: list[str] = []
    with output_path.open(encoding="utf-8") as output_file:
        for line in output_file:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, Mapping):
                question = record.get("question")
                if isinstance(question, str):
                    questions.append(question)
    return questions


def _validate_answer_reasoning(answer_reasoning: object) -> None:
    """Require an audit explanation that can be safely serialized as JSON text."""
    if not isinstance(answer_reasoning, str):
        msg = "answer_reasoning must be a string"
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


if __name__ == "__main__":
    main()
