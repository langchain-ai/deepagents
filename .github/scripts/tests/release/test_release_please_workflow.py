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

    Supports just the subset the workflow uses — `!cancelled()`, `&&`, `||`,
    parentheses, and `needs.<job>.result` / `needs.<job>.outputs.<name>` compared
    against single-quoted literals.
    """
    expr = condition
    for ref, value in values.items():
        expr = expr.replace(ref, repr(value))
    expr = expr.replace("!cancelled()", repr(not cancelled))
    expr = expr.replace("&&", " and ").replace("||", " or ")
    leftover = re.findall(r"needs\.[\w.-]+", expr)
    assert not leftover, f"unsubstituted references (update the test): {leftover}"
    return bool(eval(expr))  # noqa: S307 - workflow-derived, no external input


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
    assert "trigger-releases" not in _needs(release_please)
    # Exact match, so no status function sneaks in: without one, the implicit
    # success() gate on `needs` is retained and a red/skipped guard blocks
    # maintenance. Maintenance may run after release commits, so this must not
    # re-test `release-commit` either.
    assert _condition(release_please) == (
        "needs.guard-pending-release.outputs.skip == 'false'"
    )

    assert "release-please" in _needs(jobs["update-lockfiles"])


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
