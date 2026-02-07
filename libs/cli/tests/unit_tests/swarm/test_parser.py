"""Unit tests for swarm task file parser."""

import json

import pytest

from deepagents_cli.swarm.parser import TaskFileError, parse_task_file


@pytest.fixture
def jsonl_file(tmp_path):
    """Create a sample JSONL task file."""
    path = tmp_path / "tasks.jsonl"
    tasks = [
        {"id": "1", "description": "Analyze Q1 data"},
        {"id": "2", "description": "Analyze Q2 data", "type": "analyst"},
        {"id": "3", "description": "Summarize both analyses"},
    ]
    path.write_text("\n".join(json.dumps(t) for t in tasks))
    return path


@pytest.fixture
def csv_file(tmp_path):
    """Create a sample CSV task file."""
    path = tmp_path / "tasks.csv"
    content = """\
id,description,type
1,Analyze Q1 data,
2,Analyze Q2 data,analyst
3,Compare results,writer
"""
    path.write_text(content)
    return path


class TestParseJsonl:
    def test_parses_basic_tasks(self, jsonl_file):
        tasks = parse_task_file(jsonl_file)

        assert len(tasks) == 3
        assert tasks[0]["id"] == "1"
        assert tasks[0]["description"] == "Analyze Q1 data"

    def test_parses_optional_type(self, jsonl_file):
        tasks = parse_task_file(jsonl_file)

        assert "type" not in tasks[0]
        assert tasks[1]["type"] == "analyst"

    def test_parses_metadata(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        path.write_text('{"id": "1", "description": "Test", "metadata": {"key": "value"}}')

        tasks = parse_task_file(path)

        assert tasks[0]["metadata"] == {"key": "value"}

    def test_skips_empty_lines(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        path.write_text(
            '{"id": "1", "description": "Task 1"}\n\n'
            '{"id": "2", "description": "Task 2"}\n'
        )

        tasks = parse_task_file(path)

        assert len(tasks) == 2

    def test_auto_generates_ids_for_missing_id(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        path.write_text(
            '{"description": "Task 1"}\n'
            '{"description": "Task 2"}\n'
        )

        tasks = parse_task_file(path)

        assert [task["id"] for task in tasks] == ["auto-1", "auto-2"]

    def test_supports_description_aliases(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        path.write_text(
            '{"id": "a", "prompt": "Task from prompt"}\n'
            '{"id": "b", "task": "Task from task"}\n'
        )

        tasks = parse_task_file(path)

        assert tasks[0]["description"] == "Task from prompt"
        assert tasks[1]["description"] == "Task from task"

    def test_supports_string_entries(self, tmp_path):
        path = tmp_path / "tasks.jsonl"
        path.write_text('"Task from string"')

        tasks = parse_task_file(path)

        assert tasks[0]["id"] == "auto-1"
        assert tasks[0]["description"] == "Task from string"


class TestParseCsv:
    def test_parses_basic_tasks(self, csv_file):
        tasks = parse_task_file(csv_file)

        assert len(tasks) == 3
        assert tasks[0]["id"] == "1"
        assert tasks[0]["description"] == "Analyze Q1 data"

    def test_parses_optional_type(self, csv_file):
        tasks = parse_task_file(csv_file)

        assert "type" not in tasks[0]
        assert tasks[1]["type"] == "analyst"


class TestValidation:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_task_file(tmp_path / "nonexistent.jsonl")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")

        with pytest.raises(TaskFileError, match="empty"):
            parse_task_file(path)

    def test_missing_description(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": "1"}')

        with pytest.raises(TaskFileError, match="missing required field 'description'"):
            parse_task_file(path)

    def test_duplicate_ids(self, tmp_path):
        path = tmp_path / "dup.jsonl"
        path.write_text(
            '{"id": "1", "description": "Task 1"}\n'
            '{"id": "1", "description": "Task 2"}\n'
        )

        with pytest.raises(TaskFileError, match="Duplicate task ID"):
            parse_task_file(path)

    def test_blocked_by_is_not_supported(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"id": "1", "description": "Test", "blocked_by": ["2"]}')

        with pytest.raises(TaskFileError, match="not supported"):
            parse_task_file(path)

    def test_csv_blocked_by_is_not_supported(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("id,description,blocked_by\n1,Test,2\n")

        with pytest.raises(TaskFileError, match="not supported"):
            parse_task_file(path)

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("not valid json")

        with pytest.raises(TaskFileError, match="Invalid JSON"):
            parse_task_file(path)

    def test_invalid_task_payload_type(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text("123")

        with pytest.raises(TaskFileError, match="must be a JSON object or string"):
            parse_task_file(path)


class TestAutoDetection:
    def test_detects_jsonl_by_extension(self, jsonl_file):
        tasks = parse_task_file(jsonl_file)
        assert len(tasks) == 3

    def test_detects_csv_by_extension(self, csv_file):
        tasks = parse_task_file(csv_file)
        assert len(tasks) == 3

    def test_detects_jsonl_by_content(self, tmp_path):
        # File with unknown extension but JSONL content
        path = tmp_path / "tasks.txt"
        path.write_text('{"id": "1", "description": "Test"}')

        tasks = parse_task_file(path)
        assert len(tasks) == 1
