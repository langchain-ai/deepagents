"""Contract tests for the Code SDK pin bump workflow.

The PR selection logic is tested directly in
`.github/scripts/tests/release/test_find_code_sdk_pin_pr.py`. What cannot be
tested there, and is tested here, is the wiring between the workflow and that
selector: the body marker the selector matches must be the marker the workflow
writes, the branch prefix it requires must be the prefix the workflow builds,
and the API query must request every field it reads. Each of those is a silent
failure if it drifts — the selector stops recognising the PR it maintains, and
the workflow opens a duplicate PR on every release.

These tests parse the YAML and locate assertions inside the shell block they
belong to. A bare substring match on the whole `run:` script cannot tell a
merge in the update path from the same merge moved into the create path.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from find_code_sdk_pin_pr import BODY_MARKER, BRANCH_PREFIX

ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / ".github" / "workflows" / "bump_code_sdk_pin.yml"

STALE_GATE = "steps.versions.outputs.stale == 'true'"
LOOKUP_STEP = "Find existing automated pin bump PR"
WRITE_STEP = "Create or update pin bump PR"


def _load_workflow() -> dict:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    # PyYAML resolves the bare `on:` key to the boolean `True`.
    workflow["on"] = workflow.pop(True, workflow.get("on"))
    return workflow


def _steps() -> list[dict]:
    return _load_workflow()["jobs"]["bump"]["steps"]


def _find_step(name: str) -> dict:
    matches = [step for step in _steps() if step.get("name") == name]
    assert len(matches) == 1, f"expected one {name!r} step, found {len(matches)}"
    return matches[0]


def _checkout_step() -> dict:
    matches = [
        step
        for step in _steps()
        if step.get("uses", "").startswith("actions/checkout@")
    ]
    assert len(matches) == 1, f"expected one checkout step, found {len(matches)}"
    return matches[0]


def _shell_block(run: str, opening: str, *, index: int = 0, count: int = 1) -> str:
    """Return the lines governed by `opening`, up to its `else`/`fi`.

    Selecting the branch of the conditional, rather than the whole script,
    is what lets a test distinguish a command in the update path from the
    same command moved into the create path. `count` is asserted so that a
    new occurrence of the same test makes a deliberate update necessary.
    """
    lines = run.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == opening]
    assert len(starts) == count, (
        f"expected {count} {opening!r}, found {len(starts)}"
    )
    start = starts[index]
    indent = len(lines[start]) - len(lines[start].lstrip())
    for offset, line in enumerate(lines[start + 1 :], start=start + 1):
        stripped = line.strip()
        same_level = len(line) - len(line.lstrip()) == indent
        if same_level and stripped in {"else", "fi"}:
            return "\n".join(lines[start + 1 : offset])
    msg = f"unterminated block for {opening!r}"
    raise AssertionError(msg)


def test_workflow_is_dispatch_only() -> None:
    # `release.yml` dispatches this workflow post-publish so the new SDK is
    # installable from PyPI before CI on the PR resolves the pin.
    assert _load_workflow()["on"] == {"workflow_dispatch": None}


def test_lookup_requests_every_field_the_selector_reads() -> None:
    run = _find_step(LOOKUP_STEP)["run"]

    # Dropping a field makes the selector match nothing, so the workflow
    # silently opens a duplicate PR instead of updating the one it owns.
    for field in ("body", "headRefName", "isCrossRepository", "number", "url"):
        assert field in run, f"the API projection must request {field!r}"


def test_lookup_delegates_to_the_tested_selector() -> None:
    run = _find_step(LOOKUP_STEP)["run"]

    assert ".github/scripts/release/find_code_sdk_pin_pr.py" in run
    # `gh pr list` caps results client-side, and the maintained PR is the
    # oldest open one, so it falls off the page first.
    assert "gh pr list" not in run
    assert "--paginate" in run
    assert 'echo "branch=$branch" >> "$GITHUB_OUTPUT"' in run


def test_pr_body_carries_the_marker_the_selector_matches() -> None:
    run = _find_step(WRITE_STEP)["run"]

    assert BODY_MARKER in run, (
        "the PR body must carry the marker the selector matches, or the "
        "workflow cannot find the PR it maintains on the next run"
    )


def test_new_branch_carries_the_prefix_the_selector_requires() -> None:
    run = _find_step("Resolve workspace SDK version and current Code pin")["run"]

    branch_lines = [
        line for line in run.splitlines() if line.strip().startswith("echo \"branch=")
    ]
    assert len(branch_lines) == 1
    # The selector adopts a PR only when its head branch carries this prefix.
    assert f'echo "branch={BRANCH_PREFIX}' in branch_lines[0]


def test_update_path_merges_the_remote_default_branch() -> None:
    step = _find_step(WRITE_STEP)
    # The second occurrence chooses between `gh pr edit` and `gh pr create`.
    update_path = _shell_block(
        step["run"], 'if [ -n "$EXISTING_PR" ]; then', count=2
    )

    assert _checkout_step()["with"]["fetch-depth"] == 0
    assert step["env"]["BRANCH"] == "${{ steps.existing.outputs.branch }}"
    # A local `$DEFAULT_BRANCH` exists only on a default-branch dispatch.
    assert 'git merge --no-edit "origin/$DEFAULT_BRANCH"' in update_path
    assert 'git merge --no-edit "$DEFAULT_BRANCH"' not in step["run"]


def test_merge_conflict_is_confined_to_the_generated_pin_files() -> None:
    run = _find_step(WRITE_STEP)["run"]

    # Resolving in favour of the default branch is safe only for the two
    # files this workflow regenerates. Anything else must fail the run.
    assert '"$code_pyproject" | "$lockfile")' in run
    assert 'git checkout --theirs -- "$conflict"' in run
    assert 'echo "::error::Unexpected merge conflict in $conflict"' in run
    assert "git commit --no-edit" in run


def test_branch_update_never_rewrites_history() -> None:
    run = _find_step(WRITE_STEP)["run"]
    pushes = [line.strip() for line in run.splitlines() if "git push " in line]

    updates = [line for line in pushes if '"$BRANCH:$BRANCH"' in line]
    assert len(updates) == 1, "expected exactly one branch-update push"
    # A `+` refspec prefix forces just as `--force` does, and `--force` in a
    # different push would not be caught by scanning the whole script.
    assert "+$BRANCH" not in updates[0]
    forcing = re.findall(r"--force(?!-with-lease)|(?<!\S)-f(?!\S)", updates[0])
    assert not forcing, f"branch update must not force: {updates[0]}"


def test_merge_only_updates_are_still_pushed() -> None:
    run = _find_step(WRITE_STEP)["run"]

    # The merge runs on every update. Confining the push to the `behind`
    # path discarded it and left the PR behind the default branch.
    behind = _shell_block(run, 'if [ "$relation" = "behind" ]; then')
    assert '"$BRANCH:$BRANCH"' not in behind


def test_pr_metadata_describes_the_pin_on_the_branch() -> None:
    run = _find_step(WRITE_STEP)["run"]

    # `ahead` leaves the files alone, so the title and body must name the
    # pin the branch carries rather than the version this run targets.
    assert 'pr_version="$branch_code_pin"' in run
    assert 'to $pr_version"' in run
    assert 'to $SDK_VERSION"' not in run.split("body_file=")[-1]


def test_write_steps_gate_on_staleness_alone() -> None:
    # A title-derived gate let a run exit green on a PR whose title named a
    # version its branch did not pin.
    for name in ("Generate GitHub App token", WRITE_STEP):
        assert _find_step(name)["if"] == STALE_GATE
    assert "steps.existing.outputs.current" not in WORKFLOW.read_text()
