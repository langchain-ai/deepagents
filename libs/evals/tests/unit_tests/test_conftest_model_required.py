"""End-to-end tests for the eval suite's `--model` requirement and report-skip behaviour.

These spin up an isolated pytest invocation via `pytester` to exercise the
real `pytest_configure` and `pytest_sessionfinish` hooks in
`tests/evals/conftest.py` and `tests/evals/pytest_reporter.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["pytester"]


@pytest.fixture
def evals_pytester(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> pytest.Pytester:
    """Pytester pre-configured to load the real eval conftest and reporter plugin."""
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-test")

    repo_root = Path(__file__).resolve().parents[2]
    evals_conftest = (repo_root / "tests" / "evals" / "conftest.py").read_text(encoding="utf-8")
    reporter_src = (repo_root / "tests" / "evals" / "pytest_reporter.py").read_text(
        encoding="utf-8"
    )
    utils_src = (repo_root / "tests" / "evals" / "utils.py").read_text(encoding="utf-8")

    tests_dir = pytester.mkpydir("tests")
    evals_dir = tests_dir / "evals"
    evals_dir.mkdir()
    (evals_dir / "__init__.py").write_text("", encoding="utf-8")
    (evals_dir / "conftest.py").write_text(evals_conftest, encoding="utf-8")
    (evals_dir / "pytest_reporter.py").write_text(reporter_src, encoding="utf-8")
    (evals_dir / "utils.py").write_text(utils_src, encoding="utf-8")

    pytester.makepyfile(
        **{
            "tests/evals/test_smoke.py": (
                "import pytest\n\n@pytest.mark.langsmith\ndef test_smoke() -> None:\n    pass\n"
            )
        }
    )
    return pytester


def test_missing_model_aborts_session(evals_pytester: pytest.Pytester) -> None:
    """Without `--model`, the session must abort with a clear message."""
    result = evals_pytester.runpytest_subprocess("tests/evals", "--no-header")
    assert result.ret == 1
    combined = "\n".join(result.outlines + result.errlines)
    assert "--model is required" in combined


def test_present_model_does_not_abort_for_model_reason(evals_pytester: pytest.Pytester) -> None:
    """With `--model` set, conftest must not bail with the model-required message.

    The session may still fail downstream (langsmith plugin etc.), but the
    `--model` guard itself should not trigger.
    """
    result = evals_pytester.runpytest_subprocess(
        "tests/evals",
        "--model",
        "claude-opus-4-7",
        "--no-header",
    )
    combined = "\n".join(result.outlines + result.errlines)
    assert "--model is required" not in combined


def test_report_not_written_when_session_aborted(
    evals_pytester: pytest.Pytester, tmp_path: Path
) -> None:
    """When the session aborts before `--model` validation, the report file
    must not be clobbered with a `model: null` payload."""
    report_path = tmp_path / "report.json"
    pre_existing = '{"existing": "report"}\n'
    report_path.write_text(pre_existing, encoding="utf-8")

    result = evals_pytester.runpytest_subprocess(
        "tests/evals",
        f"--evals-report-file={report_path}",
        "--no-header",
    )
    assert result.ret == 1
    # The pre-existing report must be preserved.
    assert report_path.read_text(encoding="utf-8") == pre_existing
