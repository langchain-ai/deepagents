"""Tests for the release-please workflow."""

import json
import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[4]
WORKFLOW = ROOT / ".github/workflows/release-please.yml"
CONFIG = ROOT / "release-please-config.json"


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

    guard = jobs["guard-pending-release"]
    assert _needs(guard) == {
        "guard-empty-commit",
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
        "guard-empty-commit",
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
        "guard-empty-commit",
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
        empty_commit: str = "success",
        detect: str = "success",
        guard: str = "success",
        cancelled: bool = False,
    ) -> bool:
        return _evaluate(
            release_please_if,
            {
                "needs.guard-empty-commit.result": empty_commit,
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
    assert not runs("false", empty_commit="failure")
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
        empty_commit: str = "success",
        detect: str = "success",
        guard: str = "success",
        release_please: str = "success",
        cancelled: bool = False,
    ) -> bool:
        return _evaluate(
            update_lockfiles_if,
            {
                "needs.guard-empty-commit.result": empty_commit,
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
    assert not runs("[1]", empty_commit="failure")
    assert not runs("[1]", detect="failure")
    assert not runs("[1]", guard="failure")
    assert not runs("[1]", guard="skipped")
    assert not runs("[1]", release_please="failure")
    assert not runs("[1]", release_please="skipped")
    assert not runs("[1]", cancelled=True)


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
        empty_commit: str = "success",
        detect: str = "success",
        guard: str = "success",
        release_please: str = "success",
        cancelled: bool = False,
    ) -> bool:
        return _evaluate(
            dispatch_if,
            {
                "needs.guard-empty-commit.result": empty_commit,
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
    assert not runs("[1]", empty_commit="failure")
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
        empty_commit: str = "success",
        detect: str = "success",
        cancelled: bool = False,
    ) -> bool:
        return _evaluate(
            guard_if,
            {
                "needs.guard-empty-commit.result": empty_commit,
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
    # A cancelled workflow must not start a poll that runs for up to MAX_WAIT.
    assert not runs("false", "skipped", cancelled=True)
    # `!cancelled()` disables implicit needs-gating, so upstream failures have to
    # be rejected explicitly or a red dependency would be silently tolerated.
    assert not runs("false", "skipped", empty_commit="failure")
    assert not runs("false", "skipped", empty_commit="skipped")
    assert not runs("false", "skipped", detect="failure")
    assert not runs("true", "success", detect="failure")


def test_guard_wait_is_not_serialized_and_outlives_its_poll() -> None:
    """Two invariants the guard's comments call deliberate."""
    guard = _load_workflow()["jobs"]["guard-pending-release"]

    # A poll of up to MAX_WAIT must not hold the `release-please` slot that every
    # other push queues on.
    assert "concurrency" not in guard

    # The job timeout has to sit well above MAX_WAIT so the graceful skip=true
    # fallback wins the race. A hard timeout leaves `skip` unset, which blocks
    # maintenance until the next push instead of deferring cleanly.
    script = _step(guard, "check")["run"]
    max_wait = int(re.search(r"^\s*MAX_WAIT=(\d+)", script, re.MULTILINE).group(1))
    assert guard["timeout-minutes"] * 60 - max_wait >= 600


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
