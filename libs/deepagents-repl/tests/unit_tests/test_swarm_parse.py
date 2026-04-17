"""Port of ``libs/deepagents/src/swarm/parse.test.ts``."""

from __future__ import annotations

import pytest

from deepagents_repl._swarm import (
    SwarmTaskResult,
    SwarmTaskSpec,
    parse_tasks_jsonl,
    serialize_results_jsonl,
    serialize_tasks_jsonl,
)


class TestParseTasksJsonl:
    def test_parses_valid_single_line(self) -> None:
        tasks = parse_tasks_jsonl('{"id":"t1","description":"do thing"}\n')
        assert tasks == [SwarmTaskSpec(id="t1", description="do thing")]

    def test_parses_multiple_lines(self) -> None:
        content = "\n".join(
            [
                '{"id":"t1","description":"first"}',
                '{"id":"t2","description":"second"}',
                '{"id":"t3","description":"third"}',
            ]
        )
        tasks = parse_tasks_jsonl(content)
        assert len(tasks) == 3
        assert tasks[2].id == "t3"

    def test_preserves_subagent_type(self) -> None:
        tasks = parse_tasks_jsonl(
            '{"id":"t1","description":"do thing","subagentType":"analyst"}\n'
        )
        assert tasks[0].subagent_type == "analyst"

    def test_ignores_blank_lines(self) -> None:
        tasks = parse_tasks_jsonl(
            '{"id":"t1","description":"first"}\n\n\n'
            '{"id":"t2","description":"second"}\n'
        )
        assert len(tasks) == 2

    def test_raises_on_empty(self) -> None:
        with pytest.raises(ValueError, match="tasks.jsonl is empty"):
            parse_tasks_jsonl("")

    def test_raises_on_whitespace_only(self) -> None:
        with pytest.raises(ValueError, match="tasks.jsonl is empty"):
            parse_tasks_jsonl("  \n  \n  ")

    def test_raises_on_invalid_json(self) -> None:
        with pytest.raises(ValueError, match="Line 2: invalid JSON"):
            parse_tasks_jsonl('{"id":"t1","description":"ok"}\nnot json\n')

    def test_raises_on_missing_id(self) -> None:
        with pytest.raises(ValueError, match="validation failed"):
            parse_tasks_jsonl('{"description":"no id"}\n')

    def test_raises_on_empty_id(self) -> None:
        with pytest.raises(ValueError, match="validation failed"):
            parse_tasks_jsonl('{"id":"","description":"empty id"}\n')

    def test_raises_on_missing_description(self) -> None:
        with pytest.raises(ValueError, match="validation failed"):
            parse_tasks_jsonl('{"id":"t1"}\n')

    def test_raises_on_empty_description(self) -> None:
        with pytest.raises(ValueError, match="validation failed"):
            parse_tasks_jsonl('{"id":"t1","description":""}\n')

    def test_raises_on_duplicate_ids(self) -> None:
        content = "\n".join(
            [
                '{"id":"dup","description":"first"}',
                '{"id":"dup","description":"second"}',
            ]
        )
        with pytest.raises(ValueError, match='duplicate task id "dup"'):
            parse_tasks_jsonl(content)

    def test_collects_multiple_errors(self) -> None:
        content = "\n".join(
            [
                "not json",
                '{"id":"","description":""}',
                '{"id":"t1","description":"ok"}',
            ]
        )
        with pytest.raises(ValueError) as exc:
            parse_tasks_jsonl(content)
        msg = str(exc.value)
        assert "Line 1:" in msg
        assert "Line 2:" in msg
        assert "Line 3:" not in msg

    def test_strips_extra_properties(self) -> None:
        tasks = parse_tasks_jsonl(
            '{"id":"t1","description":"d","extra":"field"}\n'
        )
        # SwarmTaskSpec only has id/description/subagent_type.
        assert tasks[0] == SwarmTaskSpec(id="t1", description="d")


class TestSerializeTasksJsonl:
    def test_serializes_with_trailing_newline(self) -> None:
        tasks = [
            SwarmTaskSpec(id="t1", description="first"),
            SwarmTaskSpec(id="t2", description="second"),
        ]
        import json as _json

        output = serialize_tasks_jsonl(tasks)
        lines = output.split("\n")
        assert len(lines) == 3  # 2 lines + trailing empty
        assert lines[2] == ""
        assert _json.loads(lines[0]) == {"id": "t1", "description": "first"}
        assert _json.loads(lines[1]) == {"id": "t2", "description": "second"}

    def test_includes_subagent_type(self) -> None:
        import json as _json

        tasks = [SwarmTaskSpec(id="t1", description="d", subagent_type="analyst")]
        output = serialize_tasks_jsonl(tasks)
        parsed = _json.loads(output.strip())
        assert parsed["subagentType"] == "analyst"

    def test_handles_empty_iter(self) -> None:
        assert serialize_tasks_jsonl([]) == "\n"

    def test_round_trips(self) -> None:
        tasks = [
            SwarmTaskSpec(id="a", description="alpha"),
            SwarmTaskSpec(id="b", description="beta", subagent_type="custom"),
        ]
        assert parse_tasks_jsonl(serialize_tasks_jsonl(tasks)) == tasks


class TestSerializeResultsJsonl:
    def test_serializes_results(self) -> None:
        import json as _json

        results = [
            SwarmTaskResult(
                id="t1",
                subagent_type="general-purpose",
                status="completed",
                result="done",
            ),
            SwarmTaskResult(
                id="t2",
                subagent_type="general-purpose",
                status="failed",
                error="timeout",
            ),
        ]
        output = serialize_results_jsonl(results)
        lines = [ln for ln in output.split("\n") if ln.strip()]
        assert len(lines) == 2
        first = _json.loads(lines[0])
        assert first["status"] == "completed"
        assert first["result"] == "done"
        assert first["subagentType"] == "general-purpose"
        second = _json.loads(lines[1])
        assert second["status"] == "failed"
        assert second["error"] == "timeout"

    def test_empty(self) -> None:
        assert serialize_results_jsonl([]) == "\n"
