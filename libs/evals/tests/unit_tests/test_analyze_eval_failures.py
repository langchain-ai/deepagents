"""Tests for the eval failure analysis script (`.github/scripts/analyze_eval_failures.py`).

Adds the script directory to `sys.path` for import since it lives outside
the package tree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / ".github" / "scripts"))

from analyze_eval_failures import (  # ty: ignore[unresolved-import]
    _format_markdown,
    analyze_one,
    main,
    run,
)

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

    def test_wrapped_in_details_toggle(self):
        results = [{**_SAMPLE_FAILURE, "analysis": "analysis"}]
        md = _format_markdown(results)
        # Heading stays outside the toggle so the count is visible collapsed.
        heading_idx = md.index("## Failure analysis")
        details_idx = md.index("<details>")
        summary_idx = md.index("<summary>(click to expand)</summary>")
        close_idx = md.index("</details>")
        assert heading_idx < details_idx < summary_idx < close_idx


class TestAnalyzeOne:
    async def test_returns_analysis(self):
        model = AsyncMock()
        model.ainvoke.return_value = AsyncMock(text="Root cause: hallucination")
        result = await analyze_one(model, _SAMPLE_FAILURE)

        assert result["analysis"] == "Root cause: hallucination"
        assert result["test_name"] == _SAMPLE_FAILURE["test_name"]

    async def test_handles_exception_gracefully(self):
        model = AsyncMock()
        model.ainvoke.side_effect = RuntimeError("API timeout")
        result = await analyze_one(model, _SAMPLE_FAILURE)

        assert "Analysis failed" in result["analysis"]
        assert "RuntimeError" in result["analysis"]
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

    async def test_success_path_writes_outputs(self, tmp_path, capsys, monkeypatch):
        report = {
            "passed": 3,
            "failed": 1,
            "failures": [_SAMPLE_FAILURE],
        }
        report_path = tmp_path / "evals_report.json"
        report_path.write_text(json.dumps(report))

        summary_file = tmp_path / "step_summary.md"
        monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary_file))

        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = AsyncMock(text="Root cause: hallucination")

        with patch("langchain.chat_models.init_chat_model", return_value=mock_model):
            await run(report_path)

        # Verify JSON artifact
        analysis_path = tmp_path / "failure_analysis.json"
        assert analysis_path.exists()
        results = json.loads(analysis_path.read_text())
        assert len(results) == 1
        assert results[0]["analysis"] == "Root cause: hallucination"
        assert results[0]["test_name"] == _SAMPLE_FAILURE["test_name"]

        # Verify GITHUB_STEP_SUMMARY
        assert summary_file.exists()
        summary = summary_file.read_text()
        assert "Failure analysis" in summary
        assert "test_recall" in summary

        # Verify stdout
        out = capsys.readouterr().out
        assert "Failure analysis" in out

    async def test_success_path_without_summary_env(self, tmp_path, capsys, monkeypatch):
        report = {"passed": 0, "failed": 1, "failures": [_SAMPLE_FAILURE]}
        report_path = tmp_path / "evals_report.json"
        report_path.write_text(json.dumps(report))

        monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

        mock_model = AsyncMock()
        mock_model.ainvoke.return_value = AsyncMock(text="analysis")

        with patch("langchain.chat_models.init_chat_model", return_value=mock_model):
            await run(report_path)

        # JSON artifact should still be written
        assert (tmp_path / "failure_analysis.json").exists()
        # Markdown still printed to stdout
        assert "Failure analysis" in capsys.readouterr().out


class TestMain:
    def test_missing_file_exits_nonzero(self, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.argv", ["script", str(tmp_path / "nonexistent.json")])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_no_args_missing_default_exits_nonzero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["script"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
