"""Tests for the release-please workflow."""

import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / ".github/workflows/release-please.yml"
CONFIG = ROOT / "release-please-config.json"
RELEASE_WORKFLOW = ROOT / ".github/workflows/release.yml"
COMMENT_UPDATE_WORKFLOW = ROOT / ".github/workflows/release_comment_update.yml"
PENDING_SCRIPT = ROOT / ".github/scripts/release/list_pending_releases.sh"


def _load_workflow() -> dict:
    return yaml.safe_load(WORKFLOW.read_text())


def _needs(job: dict) -> set[str]:
    needs = job.get("needs", [])
    if isinstance(needs, str):
        return {needs}
    return set(needs)


def _condition(job: dict) -> str:
    return " ".join(str(job.get("if", "")).split())


def _step(job: dict, step_id: str) -> dict:
    return next(step for step in job["steps"] if step.get("id") == step_id)


def _evaluate(
    condition: str, values: dict[str, str], *, cancelled: bool = False
) -> bool:
    """Evaluate an Actions `if:` expression against concrete `needs` values.

    Substring assertions cannot tell correct nesting from no nesting: flattening
    the parentheses, or swapping `||` for `&&`, leaves every clause present while
    inverting the meaning. Only the boolean structure catches that, so evaluate it.

    Supports just the subset the workflow uses — `!cancelled()`, `always()`,
    `&&`, `||`, parentheses, and `needs.<job>.result` /
    `needs.<job>.outputs.<name>` compared against single-quoted literals.
    """
    expr = condition
    for ref, value in values.items():
        expr = expr.replace(ref, repr(value))
    expr = expr.replace("!cancelled()", repr(not cancelled))
    # always() unconditionally disables the implicit success() gate; the
    # hand-checked needs.*.result clauses beside it carry the fail-closed load.
    expr = expr.replace("always()", repr(True))
    expr = expr.replace("&&", " and ").replace("||", " or ")
    leftover = re.findall(r"needs\.[\w.-]+", expr)
    assert not leftover, f"unsubstituted references (update the test): {leftover}"
    return bool(eval(expr))  # noqa: S307 - workflow-derived, no external input


def test_condition_references_only_declared_needs() -> None:
    """Every job referenced in a job-level `if:` must appear in its `needs:`.

    The Actions `needs` context only contains declared dependencies: a
    reference to an undeclared job evaluates to an empty string instead of
    erroring, so a `result == 'success'` clause silently fails and the job
    skips even when its real dependency succeeded. This is what let the
    `update-lockfiles` fix reference the guard jobs without depending on
    them. Checking for undeclared references in `env:`/`run:` steps is out of
    scope — there it is a runtime bug, not an always-skip gate.
    """
    for job_name, job in _load_workflow()["jobs"].items():
        condition = str(job.get("if", ""))
        declared = _needs(job)
        undeclared = {
            ref.split(".")[1]
            for ref in re.findall(r"needs\.[\w-]+(?=\.(?:result|outputs)\b)", condition)
            if ref.split(".")[1] not in declared
        }
        assert not undeclared, f"{job_name} if: references undeclared needs: {undeclared}"


def test_trigger_releases_can_comment_on_release_pr() -> None:
    """Grant only the permissions needed to dispatch and report releases."""
    workflow = _load_workflow()

    assert workflow["jobs"]["trigger-releases"]["permissions"] == {
        "actions": "write",
        "issues": "read",
        "pull-requests": "write",
    }


def test_release_dispatch_precedes_release_please_maintenance() -> None:
    """Publish dispatch stays first; maintenance waits after successful dispatch."""
    workflow = _load_workflow()
    jobs = workflow["jobs"]

    assert "concurrency" not in workflow

    trigger = jobs["trigger-releases"]
    assert _needs(trigger) == {"detect-release-commit"}
    # The empty-commit guard runs inside the detector, before detection, so a
    # failed guard also blocks the dispatch.
    detect_steps = [step.get("name") for step in jobs["detect-release-commit"]["steps"]]
    assert detect_steps.index("Reject empty commits") < detect_steps.index(
        "Check if release PRs were merged"
    )

    guard = jobs["guard-pending-release"]
    assert _needs(guard) == {
        "detect-release-commit",
        "trigger-releases",
    }

    # `!cancelled()` is required: `trigger-releases` is legitimately skipped on
    # normal pushes, so the implicit success() gate on `needs` cannot be used.
    # Because it also disables that gate, every upstream result is checked by
    # hand — see test_guard_condition_fails_closed for the resulting truth table.
    guard_if = _condition(guard)
    assert "!cancelled()" in guard_if
    # Malformed detector output must fail closed (no loose != 'true').
    assert "release-commit != 'true'" not in guard_if

    release_please = jobs["release-please"]
    assert "guard-pending-release" in _needs(release_please)
    # Must stay off this job: `trigger-releases` is skipped on ordinary pushes.
    # The guard already sequences dispatch-before-maintenance; listing the
    # skipped job here would reintroduce the #5161 skip poison that left main
    # green while never refreshing release PRs.
    assert "trigger-releases" not in _needs(release_please)
    # `!cancelled()` is required for the same reason as on the guard: an
    # upstream-skipped `trigger-releases` would otherwise trip the implicit
    # success() gate on this job even after a successful `skip=false` guard.
    # Maintenance may (and should) run after release commits, so this must not
    # re-test `release-commit`.
    release_please_if = _condition(release_please)
    assert "!cancelled()" in release_please_if
    # Soft-reject only the detector's release-commit *output* gate — job ids may
    # contain the words "release-commit" (e.g. detect-release-commit).
    assert "outputs.release-commit" not in release_please_if
    assert "needs.guard-pending-release.outputs.skip == 'false'" in release_please_if

    update_lockfiles = jobs["update-lockfiles"]
    assert "release-please" in _needs(update_lockfiles)
    # The `needs` context only carries declared dependencies, so every
    # ancestor whose result the `if:` hand-checks must be listed — an
    # undeclared reference compares an empty string against 'success' and
    # the job skips even when release-please opened PRs.
    assert {
        "detect-release-commit",
        "guard-pending-release",
    } <= _needs(update_lockfiles)
    # Same skip poison as on `release-please`: this job's needs chain also ends
    # at a legitimately-skipped `trigger-releases`, so a bare outputs-only `if:`
    # keeps the implicit success() gate and the job silently skips even when
    # release-please produced PRs. It must override the gate and hand-check
    # every dependency in the chain instead.
    update_lockfiles_if = _condition(update_lockfiles)
    assert "!cancelled()" in update_lockfiles_if
    assert "needs.release-please.result == 'success'" in update_lockfiles_if
    assert "needs.release-please.outputs.prs != '[]'" in update_lockfiles_if
    # `release-please` succeeding already implies the guard saw the dispatch
    # succeed on release commits, so re-checking the skipped job is unnecessary
    # — and a `'skipped'` comparison would wrongly block this job on them.
    assert "trigger-releases" not in update_lockfiles_if

    # The notes-check dispatch sits behind the same chain (and `always()` alone
    # would tolerate red ancestors), so it must pin the same hand-checked
    # dependency results — except `update-lockfiles` itself, which is a
    # sequencing-only need whose own failure must not block the check.
    # Same declared-needs requirement as update-lockfiles: the hand-checked
    # ancestors must be listed, while `update-lockfiles` itself stays a
    # sequencing-only need whose result the `if:` does not check.
    assert {
        "detect-release-commit",
        "guard-pending-release",
        "update-lockfiles",
    } <= _needs(jobs["dispatch-release-notes-check"])
    dispatch_if = _condition(jobs["dispatch-release-notes-check"])
    assert "always()" in dispatch_if
    assert "!cancelled()" in dispatch_if
    assert "needs.release-please.result == 'success'" in dispatch_if
    assert "update-lockfiles" not in dispatch_if


def test_release_please_condition_fails_closed() -> None:
    """Pin maintenance's boolean structure after the #5161 skip-gate fix."""
    release_please_if = _condition(_load_workflow()["jobs"]["release-please"])

    def runs(
        skip: str,
        *,
        detect: str = "success",
        guard: str = "success",
        cancelled: bool = False,
    ) -> bool:
        return _evaluate(
            release_please_if,
            {
                "needs.detect-release-commit.result": detect,
                "needs.guard-pending-release.result": guard,
                "needs.guard-pending-release.outputs.skip": skip,
            },
            cancelled=cancelled,
        )

    # Guard cleared the path — the normal push and post-dispatch release path.
    assert runs("false")
    # Guard deferred (in-flight / stuck publish / operator recovery).
    assert not runs("true")
    # Unset/empty skip (crash, hard timeout) must not look like all-clear.
    assert not runs("")
    # Direct deps: any red/skipped guardian job fails closed.
    assert not runs("false", detect="failure")
    assert not runs("false", guard="failure")
    assert not runs("false", guard="skipped")
    assert not runs("false", cancelled=True)


def test_update_lockfiles_condition_fails_closed() -> None:
    """Pin lockfile regeneration's boolean structure after the #5161 skip-gate fix.

    `update-lockfiles` needs `release-please`, which transitively needs a
    legitimately-skipped `trigger-releases` on ordinary pushes — the same
    implicit-success() poison #5169 fixed for `release-please`. Without
    `!cancelled()` and hand-checked dependency results, this job silently
    skipped on a green workflow and release PR lockfiles went stale.
    """
    update_lockfiles_if = _condition(
        _load_workflow()["jobs"]["update-lockfiles"]
    )

    def runs(
        prs: str,
        *,
        detect: str = "success",
        guard: str = "success",
        release_please: str = "success",
        cancelled: bool = False,
    ) -> bool:
        return _evaluate(
            update_lockfiles_if,
            {
                "needs.detect-release-commit.result": detect,
                "needs.guard-pending-release.result": guard,
                "needs.release-please.result": release_please,
                "needs.release-please.outputs.prs": prs,
            },
            cancelled=cancelled,
        )

    # release-please opened/updated release PRs — the normal refresh path.
    assert runs(
        '[{"headBranchName": "release-please--branches--main--components--deepagents"}]'
    )
    # No PRs produced (empty output or the empty-array literal) — nothing to do.
    assert not runs("[]")
    # Unset/empty prs (release-please crashed before writing outputs) must not
    # look like work to do.
    assert not runs("")
    # Direct deps: any red guardian job fails closed.
    assert not runs("[1]", detect="failure")
    assert not runs("[1]", guard="failure")
    assert not runs("[1]", guard="skipped")
    assert not runs("[1]", release_please="failure")
    assert not runs("[1]", release_please="skipped")
    assert not runs("[1]", cancelled=True)


def test_label_release_prs_condition_fails_closed() -> None:
    """Labeling runs beside `update-lockfiles` with the same skip-gate handling.

    It needs `release-please`, so it inherits the same implicit-success()
    poison from the legitimately-skipped `trigger-releases` (#5161/#5169).
    """
    label_if = _condition(_load_workflow()["jobs"]["label-release-prs"])

    def runs(
        prs: str,
        *,
        detect: str = "success",
        guard: str = "success",
        release_please: str = "success",
        cancelled: bool = False,
    ) -> bool:
        return _evaluate(
            label_if,
            {
                "needs.detect-release-commit.result": detect,
                "needs.guard-pending-release.result": guard,
                "needs.release-please.result": release_please,
                "needs.release-please.outputs.prs": prs,
            },
            cancelled=cancelled,
        )

    assert runs('[{"number": 1, "title": "release(deepagents): 1.0.0"}]')
    assert not runs("[]")
    assert not runs("")
    assert not runs("[1]", detect="failure")
    assert not runs("[1]", guard="failure")
    assert not runs("[1]", guard="skipped")
    assert not runs("[1]", release_please="failure")
    assert not runs("[1]", release_please="skipped")
    assert not runs("[1]", cancelled=True)


def test_label_release_prs_is_off_the_lockfile_path() -> None:
    """Labeling must not sit between release-please and the lockfile refresh."""
    jobs = _load_workflow()["jobs"]
    assert "label-release-prs" not in _needs(jobs["update-lockfiles"])
    assert "label-release-prs" not in _needs(jobs["dispatch-release-notes-check"])
    assert not any(
        step.get("name") == "Label release PRs" for step in jobs["release-please"]["steps"]
    )


def test_dispatch_release_notes_check_condition_fails_closed() -> None:
    """Pin the notes-check dispatch gate: `always()` plus hand-checked deps.

    `update-lockfiles` is a sequencing-only need whose own failure must not
    block the dispatch (its comment says so), but every other ancestor in the
    chain must be checked by hand — bare `always()` would also tolerate red
    guardians and a cancelled workflow.
    """
    dispatch_if = _condition(_load_workflow()["jobs"]["dispatch-release-notes-check"])

    def runs(
        prs: str,
        *,
        detect: str = "success",
        guard: str = "success",
        release_please: str = "success",
        cancelled: bool = False,
    ) -> bool:
        return _evaluate(
            dispatch_if,
            {
                "needs.detect-release-commit.result": detect,
                "needs.guard-pending-release.result": guard,
                "needs.release-please.result": release_please,
                "needs.release-please.outputs.prs": prs,
            },
            cancelled=cancelled,
        )

    # Release PRs exist and every guardian succeeded — dispatch the check.
    assert runs("[1]")
    # No PRs / unset prs — nothing to validate.
    assert not runs("[]")
    assert not runs("")
    # Red ancestors fail closed even though the condition starts with always().
    assert not runs("[1]", detect="failure")
    assert not runs("[1]", guard="failure")
    assert not runs("[1]", guard="skipped")
    assert not runs("[1]", release_please="failure")
    assert not runs("[1]", release_please="skipped")
    assert not runs("[1]", cancelled=True)


def test_guard_condition_fails_closed() -> None:
    """Pin the guard's boolean structure, not just the presence of its clauses."""
    guard_if = _condition(_load_workflow()["jobs"]["guard-pending-release"])

    def runs(
        release_commit: str,
        trigger: str,
        *,
        detect: str = "success",
        cancelled: bool = False,
    ) -> bool:
        return _evaluate(
            guard_if,
            {
                "needs.detect-release-commit.result": detect,
                "needs.detect-release-commit.outputs.release-commit": release_commit,
                "needs.trigger-releases.result": trigger,
            },
            cancelled=cancelled,
        )

    # Normal push: `trigger-releases` is skipped and must be ignored, not awaited.
    assert runs("false", "skipped")
    assert runs("false", "success")
    # Release commit: maintenance only after the publish actually dispatched.
    assert runs("true", "success")
    assert not runs("true", "failure")
    assert not runs("true", "skipped")
    assert not runs("true", "cancelled")
    # Malformed/unset detector output must not pass as a normal push.
    assert not runs("", "skipped")
    assert not runs("garbage", "success")
    # A cancelled workflow must not start the check.
    assert not runs("false", "skipped", cancelled=True)
    # `!cancelled()` disables implicit needs-gating, so upstream failures have to
    # be rejected explicitly or a red dependency would be silently tolerated.
    assert not runs("false", "skipped", detect="failure")
    assert not runs("true", "success", detect="failure")


def test_guard_defers_instead_of_polling() -> None:
    """The guard defers to release.yml's post-publish refresh; it never polls."""
    guard = _load_workflow()["jobs"]["guard-pending-release"]

    # A slow `gh` read must not hold the `release-please` slot that every other
    # push queues on.
    assert "concurrency" not in guard

    script = _step(guard, "check")["run"]
    # The old 45-minute label poll is gone; only GitHub read errors retry.
    assert "MAX_WAIT" not in script
    assert "POLL_INTERVAL" not in script
    retry = int(re.search(r"^\s*RETRY_INTERVAL=(\d+)", script, re.MULTILINE).group(1))
    fails = int(re.search(r"^\s*MAX_QUERY_FAILS=(\d+)", script, re.MULTILINE).group(1))
    # A hard timeout leaves `skip` unset, which blocks maintenance; the retry
    # budget has to finish well inside the job timeout.
    assert guard["timeout-minutes"] * 60 - retry * fails >= 300

    # A pending release PR defers this run (skip=true) to the refresh dispatch.
    defer = script[script.index('if [ -z "$PENDING" ]; then') :]
    assert 'echo "skip=true" >> "$GITHUB_OUTPUT"' in defer


def test_pending_check_reads_labels_directly() -> None:
    """Search-index lag must neither hide a fresh merge nor a cleared label."""
    script = PENDING_SCRIPT.read_text()
    # Candidates come from `gh pr list --label` (search index); each is then
    # re-read with `gh pr view`, which reflects a just-flipped label.
    assert "gh pr list" in script
    assert "gh pr view" in script
    # A release commit's own PR is resolved from the commit, not the index.
    assert "/commits/${INCLUDE_SHA}/pulls" in script

    guard = _load_workflow()["jobs"]["guard-pending-release"]
    step = _step(guard, "check")
    assert step["env"]["HEAD_SHA"] == "${{ github.sha }}"
    body = step["run"][step["run"].index("list_pending_json() {") :]
    assert 'INCLUDE_SHA="$HEAD_SHA" bash .github/scripts/release/list_pending_releases.sh' in body


def test_release_please_rechecks_inside_concurrency_group() -> None:
    """The guard's answer can go stale while this job waits in the queue."""
    job = _load_workflow()["jobs"]["release-please"]
    steps = job["steps"]
    recheck_index = next(i for i, step in enumerate(steps) if step.get("id") == "recheck")
    recheck = steps[recheck_index]
    assert "list_pending_releases.sh" in recheck["run"]
    assert 'echo "skip=true" >> "$GITHUB_OUTPUT"' in recheck["run"]
    # Nothing that touches release branches may run before or without it.
    assert all("uses" in step for step in steps[:recheck_index])
    for step in steps[recheck_index + 1 :]:
        assert step.get("if") == "steps.recheck.outputs.skip == 'false'", step.get("name")


def test_dispatched_runs_are_maintenance_only() -> None:
    """A dispatched refresh must never re-detect a release commit."""
    workflow = _load_workflow()
    # PyYAML parses the bare `on` key as boolean True.
    triggers = workflow[True]
    assert "workflow_dispatch" in triggers
    assert triggers["push"] == {"branches": ["main"]}

    detect = workflow["jobs"]["detect-release-commit"]
    step = _step(detect, "check-releases")
    assert step["env"]["EVENT_NAME"] == "${{ github.event_name }}"
    script = step["run"]
    # Up to the branch's `exit 0`; its body has a nested `if ... fi`.
    dispatch_branch = re.search(
        r'if \[ "\$EVENT_NAME" = "workflow_dispatch" \]; then(.*?\n\s*exit 0\n)',
        script,
        re.DOTALL,
    )
    assert dispatch_branch, "missing the workflow_dispatch short-circuit"
    assert 'echo "release-commit=false" >> "$GITHUB_OUTPUT"' in dispatch_branch.group(1)
    assert "exit 0" in dispatch_branch.group(1)
    # Only main's scripts may run maintenance with write access.
    assert step["env"]["GIT_REF"] == "${{ github.ref }}"
    assert '"$GIT_REF" != "refs/heads/main"' in dispatch_branch.group(1)
    assert "exit 1" in dispatch_branch.group(1)
    # It must come before any release detection writes an output.
    assert script.index("workflow_dispatch") < script.index("release-commit=true")

    # The empty-commit guard still runs first on dispatch, so a dispatch cannot
    # route release-please around a blocked empty commit.
    reject = next(s for s in detect["steps"] if s.get("name") == "Reject empty commits")
    assert "if" not in reject


def test_dispatched_refresh_defers_on_sibling_failure() -> None:
    """A refresh after a good publish must not go red over a sibling's failure."""
    guard = _load_workflow()["jobs"]["guard-pending-release"]
    step = _step(guard, "check")
    assert step["env"]["EVENT_NAME"] == "${{ github.event_name }}"
    script = step["run"]
    body = script[script.index("fail_failed_releases() {") :]
    # The refresh branch comes before the HEAD-subject hotfix branch.
    assert body.index('"$IS_REFRESH" = true') < body.index('"$IS_HOTFIX" = true')
    assert 'defer_reason="post-publish refresh"' in body


def test_release_workflow_dispatches_refresh_after_tagging() -> None:
    """release.yml closes the loop the guard's deferral relies on."""
    release = yaml.safe_load(RELEASE_WORKFLOW.read_text())
    job = release["jobs"]["refresh-release-prs"]

    # After the label flip, never before: see guard-pending-release.
    assert _needs(job) == {"mark-release"}
    condition = _condition(job)
    assert "needs.mark-release.result == 'success'" in condition
    assert "github.ref == 'refs/heads/main'" in condition
    assert job["permissions"] == {"actions": "write"}

    script = "\n".join(step.get("run", "") for step in job["steps"])
    assert "gh workflow run release-please.yml" in script
    assert "--ref main" in script


def test_release_comment_update_ignores_dispatched_refreshes() -> None:
    """Only the push-triggered release-commit run reports the outcome."""
    workflow = yaml.safe_load(COMMENT_UPDATE_WORKFLOW.read_text())
    job = workflow["jobs"]["comment"]
    condition = _condition(job)
    assert "github.event.workflow_run.event == 'push'" in condition
    assert "github.event_name == 'workflow_dispatch'" in condition

    # The release-commit run now finishes before its publish does, so the poll
    # has to cover a full publish and still end before the job timeout.
    script = "\n".join(step.get("run", "") for step in job["steps"])
    deadline = int(re.search(r"deadline=\$\(\(SECONDS \+ (\d+)\)\)", script).group(1))
    assert deadline >= 900
    assert job["timeout-minutes"] * 60 - deadline >= 120


def test_release_please_maintenance_stays_serialized() -> None:
    """Only the release-please action is serialized, and it is never cancelled."""
    release_please = _load_workflow()["jobs"]["release-please"]

    # Job-scoped, not workflow-scoped: workflow-level concurrency would let
    # GitHub coalesce away a pending release-commit run before it can dispatch.
    assert release_please["concurrency"] == {
        "group": "release-please",
        "cancel-in-progress": False,
    }


def test_every_component_is_wired_for_detection_and_dispatch() -> None:
    """A component missing from either list publishes nothing, silently.

    `trigger-releases` is gated on the per-package outputs, so an unwired
    component skips the dispatch — and since the guard requires a successful
    dispatch on release commits, maintenance skips too. Every job would be green
    while the release did not happen. The detector now fails loudly in that case;
    this keeps the two hand-maintained lists from drifting in the first place.
    """
    jobs = _load_workflow()["jobs"]
    detect_job = jobs["detect-release-commit"]
    detect = _step(detect_job, "check-releases")["run"]
    trigger_if = _condition(jobs["trigger-releases"])
    packages = json.loads(CONFIG.read_text())["packages"]

    for path, meta in packages.items():
        component = meta["component"]
        assert f"release\\({component}\\)" in detect, (
            f"detect-release-commit does not match release({component})"
        )
        changelog = f"^{path}/{meta['changelog-path']}$"
        assert changelog in detect, (
            f"detect-release-commit does not watch {changelog}"
        )

    flags = [name for name in detect_job["outputs"] if name.endswith("-release")]
    assert len(flags) == len(packages), (
        f"{len(flags)} per-package outputs for {len(packages)} configured packages"
    )
    for flag in flags:
        assert f"needs.detect-release-commit.outputs.{flag} == 'true'" in trigger_if, (
            f"{flag} does not gate trigger-releases; that package would never publish"
        )


def test_unmatched_release_commit_fails_loudly() -> None:
    """A release commit matching no package must not conclude green."""
    detect = _step(
        _load_workflow()["jobs"]["detect-release-commit"], "check-releases"
    )["run"]

    assert 'ANY_PACKAGE=false' in detect
    assert detect.count("ANY_PACKAGE=true") == len(
        json.loads(CONFIG.read_text())["packages"]
    )
    # YAML block scalars strip the common indentation, so the closing `fi` of a
    # top-level `if` sits at column 0 in the parsed script.
    guard = re.search(
        r'^if \[ -n "\$RELEASE_VERSION" \] && \[ "\$ANY_PACKAGE" = false \]; then\n'
        r"(.*?)^fi$",
        detect,
        re.DOTALL | re.MULTILINE,
    )
    assert guard, "missing the unmatched-release-commit guard"
    assert "::error::" in guard.group(1)
    assert "exit 1" in guard.group(1)
