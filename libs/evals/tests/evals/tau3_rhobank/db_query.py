"""Database query utilities for the tau3 Rho-Bank banking knowledge domain.

Provides functions to query and modify a TransactionalDB instance with
flexible constraint-based filtering. All operations are in-memory.

Based on τ-bench / τ²-bench by Sierra Research (MIT License).
See LICENSE in this directory. Source: https://github.com/sierra-research/tau2-bench
"""

from __future__ import annotations

import json
import operator
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests.evals.tau3_rhobank.domain import TransactionalDB


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_databases(db: TransactionalDB) -> dict[str, dict[str, Any]]:
    """Convert TransactionalDB to dict format for querying.

    Args:
        db: TransactionalDB instance.

    Returns:
        Dictionary mapping database names to their records.
    """
    result: dict[str, dict[str, Any]] = {}
    for field_name in db.model_fields:
        table = getattr(db, field_name, None)
        if table is not None and hasattr(table, "data"):
            result[field_name] = {
                "data": table.data,
                "notes": table.notes,
            }
    return result


def _get_comparison_op(op_name: str) -> Callable[[Any, Any], bool]:
    """Get a comparison function by operator name.

    Args:
        op_name: Name of the operator.

    Returns:
        Comparison function.
    """
    ops: dict[str, Callable[[Any, Any], bool]] = {
        "eq": operator.eq,
        "ne": operator.ne,
        "gt": operator.gt,
        "gte": operator.ge,
        "lt": operator.lt,
        "lte": operator.le,
        "contains": lambda a, b: b in a if a is not None else False,
        "startswith": lambda a, b: str(a).startswith(str(b)) if a is not None else False,
        "endswith": lambda a, b: str(a).endswith(str(b)) if a is not None else False,
        "in": lambda a, b: a in b,
        "nin": lambda a, b: a not in b,
    }
    return ops.get(op_name, operator.eq)


def _parse_constraint(key: str, value: Any) -> tuple[str, str, Any]:  # noqa: ANN401
    """Parse a constraint key into field name and operator.

    Args:
        key: Constraint key, e.g. `amount__gt` or `status`.
        value: The value to compare against.

    Returns:
        Tuple of (field_name, operator_name, value).
    """
    if "__" in key:
        parts = key.rsplit("__", 1)
        field_name = parts[0]
        op_name = parts[1]
    else:
        field_name = key
        op_name = "eq"
    return field_name, op_name, value


def _record_matches(record: dict[str, Any], constraints: dict[str, Any]) -> bool:
    """Check if a record matches all constraints.

    Args:
        record: The record to check.
        constraints: Dictionary of constraints (field__op: value).

    Returns:
        True if record matches all constraints.
    """
    for key, value in constraints.items():
        field_name, op_name, expected = _parse_constraint(key, value)
        actual = record.get(field_name)
        compare = _get_comparison_op(op_name)
        try:
            if not compare(actual, expected):
                return False
        except (TypeError, ValueError):
            return False
    return True


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------


def list_databases(db: TransactionalDB) -> list[str]:
    """List all available database names.

    Args:
        db: TransactionalDB instance.

    Returns:
        List of database names.
    """
    return list(_load_databases(db).keys())


def get_database(db_name: str, db: TransactionalDB) -> dict[str, dict[str, Any]] | None:
    """Get all records from a database.

    Args:
        db_name: Name of the database.
        db: TransactionalDB instance.

    Returns:
        Dictionary mapping record IDs to records, or None if not found.
    """
    databases = _load_databases(db)
    db_entry = databases.get(db_name)
    if db_entry is None:
        return None
    if isinstance(db_entry, dict) and "data" in db_entry:
        return db_entry["data"]
    return db_entry


def query_db(
    db_name: str,
    db: TransactionalDB,
    return_ids: bool = False,
    limit: int | None = None,
    **constraints: Any,
) -> Union[list[dict[str, Any]], list[tuple[str, dict[str, Any]]]]:  # noqa: UP007
    """Query a database with flexible constraints.

    Args:
        db_name: Name of the database to query.
        db: TransactionalDB instance.
        return_ids: If True, returns list of (record_id, record) tuples.
        limit: Maximum number of results to return.
        **constraints: Field constraints as keyword arguments.

    Returns:
        List of matching records (or tuples with IDs if return_ids=True).
    """
    database = get_database(db_name, db)
    if database is None:
        return []

    results: list[Any] = []
    for record_id, record in database.items():
        if _record_matches(record, constraints):
            if return_ids:
                results.append((record_id, record))
            else:
                results.append(record)
            if limit is not None and len(results) >= limit:
                break
    return results


def add_to_db(db_name: str, record_id: str, record: dict[str, Any], db: TransactionalDB) -> bool:
    """Add a record to a database.

    Args:
        db_name: Name of the database.
        record_id: ID for the new record.
        record: The record data to add.
        db: TransactionalDB instance.

    Returns:
        True if successful, False if database not found or record exists.
    """
    table = getattr(db, db_name, None)
    if table is None:
        return False
    if record_id in table.data:
        return False
    table.data[record_id] = record
    return True


def update_record_in_db(
    db_name: str,
    db: TransactionalDB,
    record_id: str,
    updates: dict[str, Any],
) -> tuple[bool, dict[str, Any] | None]:
    """Update fields in an existing record.

    Args:
        db_name: Name of the database.
        db: TransactionalDB instance.
        record_id: ID of the record to update.
        updates: Dictionary of field names to new values.

    Returns:
        Tuple of (success, updated_record).
    """
    table = getattr(db, db_name, None)
    if table is None:
        return False, None
    if record_id not in table.data:
        return False, None
    for field, value in updates.items():
        table.data[record_id][field] = value
    return True, table.data[record_id]


def query_database_tool(
    database_name: str,
    constraints: str = "{}",
    db: TransactionalDB | None = None,
) -> str:
    """Tool wrapper for query_db -- handles JSON parsing and formatting.

    Args:
        database_name: Name of the database to query.
        constraints: JSON string of field constraints.
        db: TransactionalDB instance.

    Returns:
        Formatted string of query results.
    """
    if db is None:
        return "Error: TransactionalDB instance required"

    try:
        available_dbs = list_databases(db)
        if database_name not in available_dbs:
            return f"Error: Database '{database_name}' not found. Available: {available_dbs}"

        try:
            constraint_dict = json.loads(constraints) if constraints else {}
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON: {e}"

        results = query_db(database_name, db=db, return_ids=True, **constraint_dict)

        if not results:
            return f"No records found in '{database_name}'."

        formatted_lines = [f"Found {len(results)} record(s) in '{database_name}':\n"]
        for i, (record_id, record) in enumerate(results, 1):
            formatted_lines.append(f"{i}. Record ID: {record_id}")
            for field, value in record.items():
                formatted_lines.append(f"   {field}: {value}")
            formatted_lines.append("")

        return "\n".join(formatted_lines)

    except Exception as e:  # noqa: BLE001
        return f"Error querying database: {e!s}"
