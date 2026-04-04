"""Tests for the eval failure analysis script."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / ".github" / "scripts"))

from analyze_eval_failures import _format_markdown, analyze_one, run


_SAMPLE_FAILURE = {
    "test_name": "tests/evals/test_memory.py::test_recall[anthropic:claude-sonnet-4-6]",
    "category": "memory",
    "failure_message": (
        "success check failed: Expected final text to contain 'TurboWidget', "
        "got: 'I cannot determine the project name'\n\n"
        "trajectory:\nstep 1:\n  text: I cannot determine the project name"
    ),
}


class TestFormatMarkdown:
    def test_single_failure(self):
        results = [{**_SAMPLE_FAILURE, "analysis": "The agent ignored memory context."}]
        md = _format_markdown(results)
        assert "## Failure analysis (1 failure)" in md
        assert "test_recall" in md
        assert "memory" in md
        assert "The agent ignored memory context." in md

    def test_multiple_failures_plural_header(self):
        results = [
            {**_SAMPLE_FAILURE, "analysis": "analysis 1"},
            {**_SAMPLE_FAILURE, "analysis": "analysis 2"},
        ]
        md = _format_markdown(results)
        assert "2 failures" in md

    def test_empty_category_omitted(self):
        results = [{"test_name": "test_x", "category": "", "failure_message": "f", "analysis": "a"}]
        md = _format_markdown(results)
        assert "**Category:**" not in md

    def test_category_present_when_set(self):
        results = [
            {"test_name": "test_x", "category": "tool_use", "failure_message": "f", "analysis": "a"}
        ]
        md = _format_markdown(results)
        assert "**Category:** tool_use" in md


class TestAnalyzeOne:
    async def test_returns_analysis(self):
        model = AsyncMock()
        model.ainvoke.return_value = AsyncMock(content="Root cause: hallucination")
        result = await analyze_one(model, _SAMPLE_FAILURE)

        assert result["analysis"] == "Root cause: hallucination"
        assert result["test_name"] == _SAMPLE_FAILURE["test_name"]

    async def test_handles_exception_gracefully(self):
        model = AsyncMock()
        model.ainvoke.side_effect = RuntimeError("API timeout")
        result = await analyze_one(model, _SAMPLE_FAILURE)

        assert "Analysis failed" in result["analysis"]
        assert "API timeout" in result["analysis"]


class TestRun:
    async def test_no_failures_exits_early(self, tmp_path, capsys):
        report = {"passed": 5, "failed": 0, "failures": []}
        report_path = tmp_path / "evals_report.json"
        report_path.write_text(json.dumps(report))

        await run(report_path)

        assert "No failures to analyze" in capsys.readouterr().out
        assert not (tmp_path / "failure_analysis.json").exists()

    async def test_missing_failures_key_exits_early(self, tmp_path, capsys):
        report = {"passed": 5, "failed": 0}
        report_path = tmp_path / "evals_report.json"
        report_path.write_text(json.dumps(report))

        await run(report_path)

        assert "No failures to analyze" in capsys.readouterr().out
