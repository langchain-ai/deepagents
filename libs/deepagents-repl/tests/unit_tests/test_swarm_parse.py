"""Tests for generic table JSONL parse/serialize."""

from __future__ import annotations

import json

import pytest

from deepagents_repl._swarm.parse import parse_table_jsonl, serialize_table_jsonl


class TestParseTableJsonl:
    def test_parses_valid_rows(self) -> None:
        content = '{"id":"a","x":1}\n{"id":"b","x":2}\n'
        rows = parse_table_jsonl(content)
        assert rows == [{"id": "a", "x": 1}, {"id": "b", "x": 2}]

    def test_returns_empty_for_empty_content(self) -> None:
        assert parse_table_jsonl("") == []
        assert parse_table_jsonl("   \n  \n") == []

    def test_ignores_blank_lines(self) -> None:
        content = '{"id":"a"}\n\n\n{"id":"b"}\n'
        rows = parse_table_jsonl(content)
        assert len(rows) == 2

    def test_raises_on_invalid_json(self) -> None:
        with pytest.raises(ValueError, match="Line 2: invalid JSON"):
            parse_table_jsonl('{"id":"a"}\nnot json\n')

    def test_raises_on_non_object(self) -> None:
        with pytest.raises(ValueError, match="Line 1: expected a JSON object, got array"):
            parse_table_jsonl("[1, 2, 3]\n")
        with pytest.raises(ValueError, match="Line 1: expected a JSON object, got str"):
            parse_table_jsonl('"string"\n')
        with pytest.raises(ValueError, match="Line 1: expected a JSON object, got NoneType"):
            parse_table_jsonl("null\n")

    def test_collects_multiple_errors(self) -> None:
        with pytest.raises(ValueError) as exc:
            parse_table_jsonl("not json\n[1]\n{\"id\":\"ok\"}\n")
        msg = str(exc.value)
        assert "Line 1:" in msg
        assert "Line 2:" in msg
        assert "Line 3:" not in msg


class TestSerializeTableJsonl:
    def test_round_trips(self) -> None:
        rows = [{"id": "a", "x": 1}, {"id": "b", "nested": {"k": "v"}}]
        serialized = serialize_table_jsonl(rows)
        assert serialized.endswith("\n")
        assert parse_table_jsonl(serialized) == rows

    def test_empty_is_empty_string(self) -> None:
        assert serialize_table_jsonl([]) == ""

    def test_one_row_per_line(self) -> None:
        rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        lines = serialize_table_jsonl(rows).split("\n")
        # 3 rows + trailing empty from newline
        assert len(lines) == 4
        assert [json.loads(line)["id"] for line in lines[:3]] == ["a", "b", "c"]
