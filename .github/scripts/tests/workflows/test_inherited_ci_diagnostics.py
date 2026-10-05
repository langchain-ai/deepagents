"""Exercise inherited CI diagnostics and the unchanged release gate."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = Path(__file__).resolve().parents[3] / "workflows" / "ci.yml"
REPO_URL = "https://github.com/langchain-ai/deepagents"
RUN_URL = f"{REPO_URL}/actions/runs/123"
CHECK = {
    "id": 456,
    "conclusion": "failure",
    "details_url": f"{RUN_URL}/job/456",
    "check_suite": {"id": 789},
}


def _step(job: str, name: str) -> dict:
    steps = yaml.safe_load(WORKFLOW.read_text())["jobs"][job]["steps"]
    return next(step for step in steps if step["name"] == name)


def _report(checks: list[dict], *, error: bool = False, metadata: str = "") -> dict:
    script = _step("ci_success", "Report inherited CI failure")["with"]["script"]
    harness = """
const input = JSON.parse(process.argv[1]);
const result = {info: [], error: [], warning: [], summary: ''};
const core = Object.fromEntries(['info', 'error', 'warning'].map(level =>
  [level, message => result[level].push(message)]));
core.summary = {
  addHeading(text) { result.summary += text + '\\n'; return this; },
  addRaw(text) { result.summary += text; return this; },
  async write() {},
};
const context = {repo: {owner: 'langchain-ai', repo: 'deepagents'}};
const github = {
  rest: {checks: {listForSuite: 'listForSuite'}},
  async paginate(method, params) {
    result.request = {method, params};
    if (input.error) throw new Error('private API response must not be logged');
    return input.checks;
  },
};
(async () => {
  SCRIPT
})().then(() => process.stdout.write(JSON.stringify(result)));
""".replace("SCRIPT", script)
    result = subprocess.run(
        ["node", "-e", harness, json.dumps({"checks": checks, "error": error})],
        env={
            **os.environ,
            "GITHUB_SERVER_URL": "https://github.com",
            "PARENT_SHA": "abc123",
            "PARENT_CHECK": metadata or json.dumps(CHECK),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_report_links_parent_run_and_unsuccessful_jobs() -> None:
    failed = {
        "id": 100,
        "name": "code Python 3.12 <script>\n::error::injected",
        "conclusion": "failure",
        "details_url": f"{RUN_URL}/job/100",
    }
    result = _report(
        [
            CHECK,
            failed,
            {**failed, "id": 101, "name": "cancelled job", "conclusion": "cancelled"},
            {**failed, "id": 102, "name": "passed job", "conclusion": "success"},
            {**failed, "id": 103, "name": "skipped job", "conclusion": "skipped"},
        ]
    )
    summary = result["summary"]
    assert f"{REPO_URL}/commit/abc123" in summary
    assert f"Parent CI run: {RUN_URL}" in summary
    assert CHECK["details_url"] in summary
    assert failed["details_url"] in summary
    assert "cancelled job" in summary
    assert "passed job" not in summary
    assert "skipped job" not in summary
    assert "<script>" not in summary
    assert "&lt;script&gt;" in summary
    assert "\n::error::" not in "\n".join(result["info"])
    assert len(result["error"]) == 1
    assert "skipped, not passed" in result["error"][0]
    assert result["request"]["params"] == {
        "owner": "langchain-ai",
        "repo": "deepagents",
        "check_suite_id": 789,
        "filter": "latest",
        "per_page": 100,
    }


@pytest.mark.parametrize("case", ["unavailable", "empty", "malformed", "unknown-url"])
def test_report_keeps_failure_visible_when_details_are_missing(case: str) -> None:
    metadata = json.dumps({**CHECK, "details_url": "https://example.com/untrusted"})
    result = _report(
        [],
        error=case == "unavailable",
        metadata="not json"
        if case == "malformed"
        else metadata
        if case == "unknown-url"
        else "",
    )
    assert "inherits failed parent CI" in result["error"][0]
    assert f"{REPO_URL}/commit/abc123" in result["summary"]
    assert "inherited failure is unchanged" in result["summary"]
    assert "private API response" not in json.dumps(result)
    assert "https://example.com" not in result["summary"]
    if case == "unavailable":
        assert RUN_URL in result["summary"]
        assert result["warning"]


@pytest.mark.parametrize(
    "conclusion", ["success", "failure", "cancelled", None, "api-error"]
)
def test_detection_and_gate_preserve_parent_result(
    tmp_path: Path, conclusion: str | None
) -> None:
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    response = {**CHECK, "conclusion": conclusion}
    for name, body in {
        "python3": "printf true",
        "git": "printf abc123",
        "gh": "exit 1"
        if conclusion == "api-error"
        else "printf '%s' \"$CHECK_RESPONSE\"",
    }.items():
        executable = bin_path / name
        executable.write_text(f"#!/bin/sh\n{body}\n")
        executable.chmod(0o755)
    output = tmp_path / "output"
    env = {
        **os.environ,
        "PATH": f"{bin_path}:{os.environ['PATH']}",
        "CHECK_RESPONSE": json.dumps(response, indent=2),
        "GITHUB_OUTPUT": str(output),
        "GITHUB_WORKSPACE": str(tmp_path),
        "GITHUB_REPOSITORY": "langchain-ai/deepagents",
    }
    detector = _step("changes", "📝 Detect a changelog-only curated-notes apply")
    subprocess.run(["bash", "-e", "-c", detector["run"]], env=env, check=True)
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    conclusive = conclusion in {"success", "failure"}
    assert outputs["only"] == str(conclusive).lower()
    assert outputs["prior-conclusion"] == (conclusion if conclusive else "")
    if not conclusive:
        return
    assert outputs["parent-sha"] == "abc123"
    assert json.loads(outputs["parent-check"]) == response
    gate = _step("ci_success", "🎉 All Checks Passed")
    result = subprocess.run(
        ["bash", "-e", "-c", gate["run"]],
        env={
            **env,
            "CURATED_APPLY_ONLY": outputs["only"],
            "CURATED_APPLY_PARENT_CONCLUSION": outputs["prior-conclusion"],
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (0 if conclusion == "success" else 1)


def test_diagnostics_cannot_replace_the_gate_or_require_write_permissions() -> None:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    report = _step("ci_success", "Report inherited CI failure")
    assert report["continue-on-error"] is True
    assert report["if"].strip() == (
        "needs.changes.outputs.curated-apply-only == 'true' && "
        "needs.changes.outputs.curated-apply-parent-conclusion == 'failure'"
    )
    assert workflow["permissions"]["checks"] == "read"
    assert "write" not in workflow["permissions"].values()
    changes = workflow["jobs"]["changes"]["outputs"]
    for env, output in [("PARENT_SHA", "parent-sha"), ("PARENT_CHECK", "parent-check")]:
        assert (
            report["env"][env]
            == "${{ needs.changes.outputs.curated-apply-" + output + " }}"
        )
        assert (
            changes[f"curated-apply-{output}"]
            == "${{ steps.curated-apply.outputs." + output + " }}"
        )
