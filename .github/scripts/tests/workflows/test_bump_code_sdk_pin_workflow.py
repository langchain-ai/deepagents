"""Structural contracts for the Code SDK pin bump workflow."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / ".github" / "workflows" / "bump_code_sdk_pin.yml"


def _load_workflow() -> dict:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    workflow["on"] = workflow.pop(True, workflow.get("on"))
    return workflow


def _find_step(name: str) -> dict:
    steps = _load_workflow()["jobs"]["bump"]["steps"]
    matches = [step for step in steps if step.get("name") == name]
    assert len(matches) == 1, f"expected one {name!r} step, found {len(matches)}"
    return matches[0]


def test_existing_pr_lookup_uses_tested_selector() -> None:
    step = _find_step("Find existing automated pin bump PR")

    assert "gh pr list --state open" in step["run"]
    assert ".github/scripts/release/find_code_sdk_pin_pr.py" in step["run"]
    assert 'echo "branch=$branch" >> "$GITHUB_OUTPUT"' in step["run"]


def test_existing_pr_update_preserves_branch_history() -> None:
    step = _find_step("Create or update pin bump PR")
    run = step["run"]
    checkout = _load_workflow()["jobs"]["bump"]["steps"][0]

    assert checkout["with"]["fetch-depth"] == 0
    assert step["env"]["BRANCH"] == "${{ steps.existing.outputs.branch }}"
    assert 'git merge --no-edit "$DEFAULT_BRANCH"' in run
    assert 'git push "https://x-access-token:${GH_TOKEN}' in run
    assert '"$BRANCH:$BRANCH"' in run
    assert "--force" not in run
    assert 'gh pr edit "$EXISTING_PR"' in run
