"""CSV enrichment module for swarm execution.

This module provides functionality to "enrich" a CSV by filling in empty columns
using subagents. Each row becomes a task where the subagent receives context from
filled columns and returns values for empty columns.
"""

import csv
import json
import re
from pathlib import Path

from deepagents_cli.swarm.types import SwarmTask


class EnrichmentError(Exception):
    """Error during CSV enrichment."""

    pass


def parse_csv_for_enrichment(path: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    """Parse a CSV file for enrichment, identifying which columns need filling.

    Args:
        path: Path to the CSV file.

    Returns:
        Tuple of (headers, rows) where rows are dicts with column values.

    Raises:
        EnrichmentError: If the file cannot be parsed.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    rows: list[dict[str, str]] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise EnrichmentError("CSV file has no headers")

        headers = list(reader.fieldnames)

        for row in reader:
            rows.append(dict(row))

    if not rows:
        raise EnrichmentError("CSV file has no data rows")

    return headers, rows


def create_enrichment_tasks(
    headers: list[str],
    rows: list[dict[str, str]],
    *,
    id_column: str | None = None,
) -> list[SwarmTask]:
    """Create SwarmTasks for enriching each row.

    For each row, creates a task that:
    - Provides context from filled columns
    - Requests values for empty columns
    - Expects JSON output with the empty column names as keys

    Args:
        headers: Column headers from the CSV.
        rows: List of row dicts.
        id_column: Column to use as task ID. If None, uses row index.

    Returns:
        List of SwarmTask objects.
    """
    tasks: list[SwarmTask] = []

    for idx, row in enumerate(rows):
        # Determine task ID
        if id_column and id_column in row and row[id_column]:
            task_id = str(row[id_column])
        else:
            task_id = str(idx + 1)

        # Separate filled and empty columns
        context_parts: list[str] = []
        empty_columns: list[str] = []

        for col in headers:
            value = row.get(col, "").strip()
            if value:
                context_parts.append(f"- {col}: {value}")
            else:
                empty_columns.append(col)

        if not empty_columns:
            # Nothing to fill for this row, skip
            continue

        # Build the task description
        context_str = "\n".join(context_parts) if context_parts else "(no context provided)"
        columns_str = ", ".join(f'"{col}"' for col in empty_columns)

        # Build JSON template for expected output
        json_lines = []
        for i, col in enumerate(empty_columns):
            comma = "," if i < len(empty_columns) - 1 else ""
            json_lines.append(f'  "{col}": "..."{comma}')
        json_template = "\n".join(json_lines)

        description = f"""Research and fill in the missing information.

**Known information:**
{context_str}

**Columns to fill:** {columns_str}

**Instructions:**
1. Use the known information as context to research the missing values
2. Return ONLY a valid JSON object with the missing column names as keys
3. If you cannot find a value, use null or "N/A"

**Required output format (JSON only, no other text):**
```json
{{
{json_template}
}}
```"""

        task: SwarmTask = {
            "id": task_id,
            "description": description,
            "metadata": {
                "row_index": idx,
                "empty_columns": empty_columns,
                "original_row": row,
            },
        }
        tasks.append(task)

    return tasks


def parse_enrichment_output(output: str) -> dict[str, str]:
    """Parse JSON output from an enrichment task.

    Handles various formats:
    - Pure JSON
    - JSON in markdown code blocks
    - JSON with surrounding text

    Args:
        output: Raw output from the subagent.

    Returns:
        Dict mapping column names to values.
    """
    if not output:
        return {}

    # Try to extract JSON from markdown code blocks
    json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", output, re.DOTALL)
    if json_match:
        output = json_match.group(1)

    # Try to find JSON object pattern
    json_obj_match = re.search(r"\{[^{}]*\}", output, re.DOTALL)
    if json_obj_match:
        output = json_obj_match.group(0)

    try:
        result = json.loads(output)
        if isinstance(result, dict):
            # Convert all values to strings
            return {k: str(v) if v is not None else "" for k, v in result.items()}
    except json.JSONDecodeError:
        pass

    return {}


def merge_enrichment_results(
    headers: list[str],
    rows: list[dict[str, str]],
    results: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    """Merge enrichment results back into the original rows.

    Args:
        headers: Original CSV headers.
        rows: Original row data.
        results: Dict mapping task_id -> parsed output dict.

    Returns:
        List of enriched row dicts.
    """
    enriched_rows: list[dict[str, str]] = []

    for idx, row in enumerate(rows):
        enriched = dict(row)

        # Find the result for this row (by index or by ID column value)
        task_id = str(idx + 1)
        result = results.get(task_id, {})

        # Also try to find by any column value that might be the ID
        if not result:
            for col_value in row.values():
                if col_value and col_value in results:
                    result = results[col_value]
                    break

        # Merge results into empty columns
        for col in headers:
            if not enriched.get(col, "").strip() and col in result:
                enriched[col] = result[col]

        enriched_rows.append(enriched)

    return enriched_rows


def write_enriched_csv(
    path: Path,
    headers: list[str],
    rows: list[dict[str, str]],
) -> Path:
    """Write enriched data to a CSV file.

    Args:
        path: Output path for the enriched CSV.
        headers: Column headers.
        rows: Enriched row data.

    Returns:
        Path to the written file.
    """
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    return path
