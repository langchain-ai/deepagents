"""Parser for JSONL swarm task files."""

import json
from pathlib import Path
from typing import Any

from deepagents_cli.swarm.types import SwarmTask


class TaskFileError(Exception):
    """Error parsing a task file."""


def parse_task_file(path: str | Path) -> list[SwarmTask]:
    """Parse a JSONL task file into SwarmTask objects.

    Args:
        path: Path to a JSONL task file.

    Returns:
        List of parsed task definitions.

    Raises:
        FileNotFoundError: If the file does not exist.
        TaskFileError: If the file has invalid format or content.
    """
    task_path = Path(path)
    if not task_path.exists():
        msg = f"Task file not found: {task_path}"
        raise FileNotFoundError(msg)

    if task_path.suffix.lower() != ".jsonl":
        msg = f"Task file must be JSONL (.jsonl). Got: {task_path.name}"
        raise TaskFileError(msg)

    tasks: list[SwarmTask] = []

    with task_path.open("r", encoding="utf-8") as file_handle:
        for line_num, raw_line in enumerate(file_handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                raw_data = json.loads(line)
            except json.JSONDecodeError as exc:
                msg = f"Invalid JSON on line {line_num}: {exc}"
                raise TaskFileError(msg) from exc

            data = _normalize_task_payload(raw_data, line_num)
            task = _validate_and_convert_task(
                data,
                line_num,
                default_id=f"auto-{line_num}",
            )
            tasks.append(task)

    if not tasks:
        msg = "Task file is empty"
        raise TaskFileError(msg)

    _validate_task_ids(tasks)
    return tasks


def _normalize_task_payload(
    raw_data: dict[str, Any] | str,
    line_num: int,
) -> dict[str, Any]:
    """Normalize a task payload into canonical dict form.

    Returns:
        Normalized task dictionary.

    Raises:
        TaskFileError: If the payload is not a JSON object or string.
    """
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
    """Validate task data and convert to SwarmTask.

    Returns:
        Validated task dictionary.

    Raises:
        TaskFileError: If required fields or optional structures are invalid.
    """
    if "description" not in data or not str(data["description"]).strip():
        msg = f"Line {line_num}: missing required field 'description'"
        raise TaskFileError(msg)

    if "blocked_by" in data:
        msg = (
            "Field 'blocked_by' is not supported in simplified swarm mode. "
            "All tasks run independently in parallel."
        )
        raise TaskFileError(msg)

    task: SwarmTask = {
        "id": str(data.get("id", "")).strip() or default_id,
        "description": str(data["description"]).strip(),
    }

    if "type" in data:
        task["type"] = str(data["type"]).strip()

    if "metadata" in data:
        if not isinstance(data["metadata"], dict):
            msg = f"Line {line_num}: metadata must be a dict"
            raise TaskFileError(msg)
        task["metadata"] = data["metadata"]

    return task


def _validate_task_ids(tasks: list[SwarmTask]) -> None:
    """Validate that all task IDs are unique.

    Raises:
        TaskFileError: If duplicate IDs are found.
    """
    task_ids: set[str] = set()
    for task in tasks:
        task_id = task["id"]
        if task_id in task_ids:
            msg = f"Duplicate task ID: {task_id}"
            raise TaskFileError(msg)
        task_ids.add(task_id)
