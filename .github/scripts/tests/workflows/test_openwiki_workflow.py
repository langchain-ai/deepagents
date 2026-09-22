"""Behavioral tests for merging generated OpenWiki pull requests."""

import json
import shutil
import subprocess
import sys
from itertools import pairwise
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / ".github" / "workflows" / "openwiki-update.yml"
HEAD_SHA = "d77258dee4f5f0d7bbe6bb913e69767415948751"
PR_ENDPOINT = "repos/langchain-ai/deepagents/pulls/6404"
BASH = shutil.which("bash")
JQ = shutil.which("jq")
pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or not BASH or not JQ,
    reason="Requires POSIX bash and jq",
)
STUB = """
import json
import os
import sys
from pathlib import Path

command = Path(sys.argv[0]).name
calls_path = Path(os.environ["CALLS"])
calls = json.loads(calls_path.read_text())
call = [command, *sys.argv[1:]]
previous = sum(item == call for item in calls)
calls.append(call)
calls_path.write_text(json.dumps(calls))
if command == "sleep":
    sys.exit(0)
scenario = json.loads(Path(os.environ["SCENARIO"]).read_text())
responses = scenario["put" if "PUT" in sys.argv else "get"]
response = responses[min(previous, len(responses) - 1)]
sys.stdout.write(response["stdout"])
sys.stderr.write(response.get("stderr", ""))
sys.exit(response["exitcode"])
"""


def _pr(field: str = "", value: str | bool | None = None) -> str:
    data = {
        "base": {"ref": "main"},
        "head": {
            "label": "langchain-ai:openwiki/update",
            "repo": {"full_name": "langchain-ai/deepagents"},
            "sha": HEAD_SHA,
        },
        "state": "open",
        "mergeable": True,
    }
    if field:
        target = data
        *parents, name = field.split(".")
        for parent in parents:
            target = target[parent]
        target[name] = value
    return json.dumps(data)


def _response(
    status: int = 200, *, body: str | None = None, crlf: bool = False
) -> dict[str, str | int]:
    newline = "\r\n" if crlf else "\n"
    if body is None:
        body = json.dumps({"merged": status == 200, "message": f"status {status}"})
    return {
        "stdout": f"HTTP/2.0 {status} Status{newline}Content-Type: application/json"
        f"{newline}{newline}{body}",
        "exitcode": 0 if status == 200 else 1,
    }


def _run_merge(
    tmp_path: Path,
    responses: list[dict[str, str | int]],
    *,
    prs: list[str] | None = None,
    overrides: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("gh", "sleep"):
        stub = bin_dir / name
        stub.write_text(f"#!{sys.executable}\n{STUB}")
        stub.chmod(0o755)
    (bin_dir / "jq").symlink_to(JQ)
    scenario = tmp_path / "scenario.json"
    scenario.write_text(
        json.dumps(
            {
                "get": [{"stdout": pr, "exitcode": 0} for pr in (prs or [_pr()])],
                "put": responses,
            }
        )
    )
    calls_path = tmp_path / "calls.json"
    calls_path.write_text("[]")
    workflow = yaml.safe_load(WORKFLOW.read_text())
    step = next(
        step
        for step in workflow["jobs"]["update"]["steps"]
        if step.get("name") == "Merge OpenWiki update pull request"
    )
    result = subprocess.run(
        [BASH, "-c", step["run"]],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
        env={
            "PATH": str(bin_dir),
            "CALLS": str(calls_path),
            "SCENARIO": str(scenario),
            "GITHUB_REPOSITORY": "langchain-ai/deepagents",
            "GITHUB_REPOSITORY_OWNER": "langchain-ai",
            "PR_NUMBER": "6404",
            "HEAD_SHA": HEAD_SHA,
            "EXPECTED_BASE": "main",
            "EXPECTED_HEAD": "langchain-ai:openwiki/update",
            **(overrides or {}),
        },
    )
    return result, json.loads(calls_path.read_text())


def _merges(calls: list[list[str]]) -> list[list[str]]:
    return [call for call in calls if "PUT" in call]


def _assert_pinned_merge(call: list[str]) -> None:
    assert call[:2] == ["gh", "api"]
    assert call[call.index("--method") + 1] == "PUT"
    assert f"{PR_ENDPOINT}/merge" in call
    fields = [value for flag, value in pairwise(call) if flag == "-f"]
    assert set(fields) == {"merge_method=squash", f"sha={HEAD_SHA}"}


@pytest.mark.parametrize("crlf", [False, True])
def test_merge_succeeds_with_pinned_sha(tmp_path: Path, crlf: bool) -> None:
    result, calls = _run_merge(tmp_path, [_response(crlf=crlf)])
    assert result.returncode == 0, result.stderr
    assert "Merged OpenWiki PR #6404." in result.stdout
    assert len(_merges(calls)) == 1
    _assert_pinned_merge(_merges(calls)[0])
    assert not any(call[0] == "sleep" for call in calls)


@pytest.mark.parametrize("crlf", [False, True])
def test_merge_retries_unsatisfied_requirements(tmp_path: Path, crlf: bool) -> None:
    result, calls = _run_merge(
        tmp_path,
        [_response(405, crlf=crlf), _response(crlf=crlf)],
        prs=[_pr("mergeable", None), _pr()],
    )
    assert result.returncode == 0, result.stderr
    assert len(_merges(calls)) == 2
    for call in _merges(calls):
        _assert_pinned_merge(call)
    assert [call for call in calls if call[0] == "sleep"] == [["sleep", "15"]]


def test_merge_retry_budget_expires_without_final_sleep(tmp_path: Path) -> None:
    result, calls = _run_merge(tmp_path, [_response(405)])
    assert result.returncode != 0
    assert "still cannot be merged after 60 attempts" in result.stdout
    assert len(_merges(calls)) == 60
    assert [call for call in calls if call[0] == "sleep"] == [["sleep", "15"]] * 59
    assert calls[-1] in _merges(calls)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("base.ref", "other"),
        ("head.label", "langchain-ai:other"),
        ("head.repo.full_name", "other/deepagents"),
        ("head.sha", "a" * 40),
        ("state", "closed"),
        ("mergeable", False),
    ],
)
def test_merge_rejects_changed_or_unmergeable_pr(
    tmp_path: Path, field: str, value: str | bool
) -> None:
    result, calls = _run_merge(tmp_path, [_response()], prs=[_pr(field, value)])
    assert result.returncode != 0
    assert not _merges(calls)
    assert not any(call[0] == "sleep" for call in calls)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("PR_NUMBER", "6404/merge"),
        ("HEAD_SHA", "not-a-sha"),
        ("EXPECTED_BASE", "other"),
        ("EXPECTED_HEAD", "langchain-ai:other"),
    ],
)
def test_merge_rejects_invalid_inputs(tmp_path: Path, field: str, value: str) -> None:
    result, calls = _run_merge(tmp_path, [_response()], overrides={field: value})
    assert result.returncode != 0
    assert calls == []


def test_merge_stops_when_head_changes_between_attempts(tmp_path: Path) -> None:
    result, calls = _run_merge(
        tmp_path, [_response(405), _response()], prs=[_pr(), _pr("head.sha", "a" * 40)]
    )
    assert result.returncode != 0
    assert len(_merges(calls)) == 1
    _assert_pinned_merge(_merges(calls)[0])
    assert [call for call in calls if call[0] == "sleep"] == [["sleep", "15"]]


@pytest.mark.parametrize("status", [401, 403, 409, 500])
def test_merge_does_not_retry_terminal_errors(tmp_path: Path, status: int) -> None:
    result, calls = _run_merge(tmp_path, [_response(status), _response()])
    assert result.returncode != 0
    assert len(_merges(calls)) == 1
    _assert_pinned_merge(_merges(calls)[0])
    assert not any(call[0] == "sleep" for call in calls)


def test_merge_does_not_retry_transport_errors(tmp_path: Path) -> None:
    result, calls = _run_merge(
        tmp_path, [{"stdout": "", "stderr": "connection failed", "exitcode": 1}]
    )
    assert result.returncode != 0
    assert "connection failed" in result.stderr
    assert len(_merges(calls)) == 1
    assert not any(call[0] == "sleep" for call in calls)


@pytest.mark.parametrize("body", ['{"merged": false}', "not json", "{}"])
def test_merge_requires_confirmation_in_success_response(
    tmp_path: Path, body: str
) -> None:
    result, calls = _run_merge(tmp_path, [_response(body=body)])
    assert result.returncode != 0
    assert "Merged OpenWiki PR" not in result.stdout
    assert len(_merges(calls)) == 1
    assert not any(call[0] == "sleep" for call in calls)
