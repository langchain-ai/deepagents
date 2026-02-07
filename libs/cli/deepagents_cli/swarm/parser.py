"""Parser for swarm task files (JSONL and CSV formats)."""

import csv
import json
from pathlib import Path
from typing import Any

from deepagents_cli.swarm.types import SwarmTask


class TaskFileError(Exception):
    """Error parsing a task file."""

    pass


def parse_task_file(path: str | Path) -> list[SwarmTask]:
    """Parse a task file (JSONL or CSV) into SwarmTask objects.

    Args:
        path: Path to the task file. Format is auto-detected from extension.
              - .jsonl: JSON Lines format (one JSON object per line)
              - .csv: CSV format with headers

    Returns:
        List of SwarmTask objects.

    Raises:
        TaskFileError: If the file cannot be parsed or has invalid format.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        return _parse_jsonl(path)
    elif suffix == ".csv":
        return _parse_csv(path)
    else:
        # Try to detect format from content
        content = path.read_text().strip()
        if content.startswith("{"):
            return _parse_jsonl(path)
        else:
            return _parse_csv(path)


def _parse_jsonl(path: Path) -> list[SwarmTask]:
    """Parse a JSONL task file.

    Expected format (one JSON object or string per line):
    {"id": "1", "description": "Task 1"}
    {"id": "2", "description": "Task 2", "type": "analyst"}
    "Task 4"
    """
    tasks: list[SwarmTask] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                raw_data = json.loads(line)
            except json.JSONDecodeError as e:
                raise TaskFileError(f"Invalid JSON on line {line_num}: {e}")

            data = _normalize_task_payload(raw_data, line_num)
            task = _validate_and_convert_task(
                data,
                line_num,
                default_id=f"auto-{line_num}",
            )
            tasks.append(task)

    if not tasks:
        raise TaskFileError("Task file is empty")

    _validate_task_ids(tasks)
    return tasks


def _parse_csv(path: Path) -> list[SwarmTask]:
    """Parse a CSV task file.

    Expected format:
    id,description,type
    1,Task 1,,
    2,Task 2,analyst,
    3,Task 3,writer
    """
    tasks: list[SwarmTask] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise TaskFileError("CSV file has no headers")

        # Check required columns (id is optional and auto-generated if missing)
        required = {"description"}
        description_aliases = {"task", "prompt"}
        fieldnames = set(reader.fieldnames)
        missing = required - fieldnames
        if missing and fieldnames.intersection(description_aliases):
            missing = set()
        if missing:
            raise TaskFileError(f"CSV missing required columns: {missing}")

        for line_num, row in enumerate(reader, start=2):  # Start at 2 (after header)
            data = _csv_row_to_dict(row)
            data = _normalize_task_payload(data, line_num)
            task = _validate_and_convert_task(
                data,
                line_num,
                default_id=f"auto-{line_num}",
            )
            tasks.append(task)

    if not tasks:
        raise TaskFileError("Task file is empty")

    _validate_task_ids(tasks)
    return tasks


def _csv_row_to_dict(row: dict[str, str]) -> dict:
    """Convert a CSV row to a task dict, handling special fields."""
    data: dict = {}

    for key, value in row.items():
        if value is None or value.strip() == "":
            continue

        value = value.strip()

        if key == "blocked_by":
            msg = (
                "Field 'blocked_by' is not supported in simplified swarm mode. "
                "All tasks run independently in parallel."
            )
            raise TaskFileError(msg)
        elif key == "metadata":
            # Parse JSON for metadata
            try:
                data[key] = json.loads(value)
            except json.JSONDecodeError:
                raise TaskFileError(f"Invalid JSON in metadata column: {value}")
        else:
            data[key] = value

    return data


def _normalize_task_payload(raw_data: dict[str, Any] | str, line_num: int) -> dict[str, Any]:
    """Normalize a task payload into canonical dict form."""
    if isinstance(raw_data, str):
        return {"description": raw_data}
    if not isinstance(raw_data, dict):
        msg = f"Line {line_num}: task entry must be a JSON object or string"
        raise TaskFileError(msg)

    data = dict(raw_data)
    if "description" not in data:
        for alias in ("task", "prompt"):
            if alias in data:
                data["description"] = data[alias]
                break
    return data


def _validate_and_convert_task(
    data: dict[str, Any], line_num: int, *, default_id: str
) -> SwarmTask:
    """Validate task data and convert to SwarmTask."""
    # Check required fields
    if "description" not in data or not str(data["description"]).strip():
        raise TaskFileError(f"Line {line_num}: missing required field 'description'")

    task_id = str(data.get("id", "")).strip() or default_id

    # Build the task
    task: SwarmTask = {
        "id": task_id,
        "description": str(data["description"]).strip(),
    }

    # Optional fields
    if "type" in data:
        task["type"] = str(data["type"])

    if "blocked_by" in data:
        msg = (
            f"Line {line_num}: field 'blocked_by' is not supported in simplified swarm mode. "
            "All tasks run independently in parallel."
        )
        raise TaskFileError(msg)

    if "metadata" in data:
        if not isinstance(data["metadata"], dict):
            raise TaskFileError(f"Line {line_num}: metadata must be a dict")
        task["metadata"] = data["metadata"]

    return task


def _validate_task_ids(tasks: list[SwarmTask]) -> None:
    """Validate that all task IDs are unique."""
    task_ids = set()
    for task in tasks:
        if task["id"] in task_ids:
            raise TaskFileError(f"Duplicate task ID: {task['id']}")
        task_ids.add(task["id"])
