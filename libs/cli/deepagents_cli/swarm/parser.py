"""Parser for swarm task files (JSONL and CSV formats)."""

import csv
import json
from pathlib import Path

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

    Expected format (one JSON object per line):
    {"id": "1", "description": "Task 1"}
    {"id": "2", "description": "Task 2", "type": "analyst"}
    {"id": "3", "description": "Task 3", "blocked_by": ["1", "2"]}
    """
    tasks: list[SwarmTask] = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise TaskFileError(f"Invalid JSON on line {line_num}: {e}")

            task = _validate_and_convert_task(data, line_num)
            tasks.append(task)

    if not tasks:
        raise TaskFileError("Task file is empty")

    _validate_task_ids(tasks)
    return tasks


def _parse_csv(path: Path) -> list[SwarmTask]:
    """Parse a CSV task file.

    Expected format:
    id,description,type,blocked_by
    1,Task 1,,
    2,Task 2,analyst,
    3,Task 3,writer,"1,2"
    """
    tasks: list[SwarmTask] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise TaskFileError("CSV file has no headers")

        # Check required columns
        required = {"id", "description"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise TaskFileError(f"CSV missing required columns: {missing}")

        for line_num, row in enumerate(reader, start=2):  # Start at 2 (after header)
            data = _csv_row_to_dict(row)
            task = _validate_and_convert_task(data, line_num)
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
            # Parse comma-separated list
            data[key] = [v.strip() for v in value.split(",") if v.strip()]
        elif key == "metadata":
            # Parse JSON for metadata
            try:
                data[key] = json.loads(value)
            except json.JSONDecodeError:
                raise TaskFileError(f"Invalid JSON in metadata column: {value}")
        else:
            data[key] = value

    return data


def _validate_and_convert_task(data: dict, line_num: int) -> SwarmTask:
    """Validate task data and convert to SwarmTask."""
    # Check required fields
    if "id" not in data:
        raise TaskFileError(f"Line {line_num}: missing required field 'id'")
    if "description" not in data:
        raise TaskFileError(f"Line {line_num}: missing required field 'description'")

    # Ensure id is a string
    task_id = str(data["id"])

    # Build the task
    task: SwarmTask = {
        "id": task_id,
        "description": str(data["description"]),
    }

    # Optional fields
    if "type" in data:
        task["type"] = str(data["type"])

    if "blocked_by" in data:
        blocked_by = data["blocked_by"]
        if isinstance(blocked_by, str):
            # Handle single ID as string
            task["blocked_by"] = [blocked_by]
        elif isinstance(blocked_by, list):
            task["blocked_by"] = [str(b) for b in blocked_by]
        else:
            raise TaskFileError(f"Line {line_num}: blocked_by must be a list or string")

    if "metadata" in data:
        if not isinstance(data["metadata"], dict):
            raise TaskFileError(f"Line {line_num}: metadata must be a dict")
        task["metadata"] = data["metadata"]

    return task


def _validate_task_ids(tasks: list[SwarmTask]) -> None:
    """Validate that all task IDs are unique and blocked_by references exist."""
    task_ids = set()
    for task in tasks:
        if task["id"] in task_ids:
            raise TaskFileError(f"Duplicate task ID: {task['id']}")
        task_ids.add(task["id"])

    # Check blocked_by references
    for task in tasks:
        blocked_by = task.get("blocked_by", [])
        for dep_id in blocked_by:
            if dep_id not in task_ids:
                raise TaskFileError(
                    f"Task '{task['id']}' references non-existent task '{dep_id}' in blocked_by"
                )
