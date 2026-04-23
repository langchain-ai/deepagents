"""Graders for the compaction benchmark.

Each grader takes a ``GraderContext`` bundle and returns a single
``CheckpointResult`` (or ``None`` when the checkpoint can't be graded
from the inputs — e.g. trajectory graders given a context without a
trajectory).

Design notes:

- Graders are **pure**: given the same inputs they produce the same
  output. The only exceptions are ``grade_g15`` (LLM judge, which
  bottoms out in an external model call) and ``grade_g16`` (subprocess
  grader, which runs ``pytest`` against a materialized filesystem).
  Those two are explicitly flagged and return ``None`` rather than
  running when their external dependency is missing.
- Graders **never raise** for grading-level issues. If a fixture is
  malformed or a parse fails, the grader returns
  ``CheckpointResult(score=0.0, evidence="...")`` explaining why. That
  way a malformed scorecard never poisons an entire run.
- Graders **always return evidence**. Pass or fail, a one-line
  explanation is included so reviewers can understand the verdict
  without re-running anything.

See ``task_spec.CHECKPOINTS`` for the authoritative catalog; each
grader here implements exactly one checkpoint and is named
``grade_g<N>`` to match the catalog id.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tests.evals.compaction_bench.scorecard import CheckpointResult
from tests.evals.compaction_bench.task_spec import (
    EXTERNAL_RETRY_LIBS,
    Constraint,
    Instance,
    Phase,
    Rejection,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from tests.evals.utils import AgentTrajectory


# ---------------------------------------------------------------------------
# Context types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerTurnTrajectory:
    """Per-user-turn slice of the agent trajectory.

    Attributes:
        turn: 1-indexed user-message turn number.
        phase: Phase this turn belongs to (from the instance script).
        trajectory: The ``AgentTrajectory`` produced by running the
            agent on this turn's user message.
    """

    turn: int
    phase: Phase
    trajectory: AgentTrajectory


@dataclass(frozen=True)
class GraderContext:
    """Bundle of everything graders need.

    Attributes:
        instance: The ``Instance`` being graded; used for rejection
            turn lookups, canonical-path expectations, etc.
        fixture_files: Original fixture contents, keyed by leading-slash
            absolute path (e.g. ``"/webhooks/generic_handler.py"``).
            Used by graders that diff against the starting state.
        final_files: Final filesystem contents in the same keying.
        per_turn_trajectories: One entry per scripted user turn.

            Optional; trajectory-dependent graders return ``None`` when
            this is empty, which lets grader unit tests run without any
            agent execution.
    """

    instance: Instance
    fixture_files: Mapping[str, str]
    final_files: Mapping[str, str]
    per_turn_trajectories: Sequence[PerTurnTrajectory] = field(default=())


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def load_fixture(fixture_dir: Path) -> dict[str, str]:
    """Walk ``fixture_dir`` and return its text contents keyed by leading-slash path.

    Binary files (that fail UTF-8 decode) are skipped — they are not
    load-bearing for any grader. The leading-slash convention matches
    how ``FilesystemBackend`` represents paths, which means the same
    dict layout works for both fixture and final-state inputs.

    Args:
        fixture_dir: Path to the fixture root on disk.

    Returns:
        Mapping from leading-slash path to file content.
    """
    files: dict[str, str] = {}
    for path in fixture_dir.rglob("*"):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(fixture_dir).as_posix()
        files[f"/{rel}"] = content
    return files


def _paths_under(files: Mapping[str, str], prefix: str) -> dict[str, str]:
    """Return files whose key starts with ``prefix``.

    Args:
        files: A file-contents mapping.
        prefix: Leading-slash path prefix (e.g. ``"/billing/"``).

    Returns:
        The subset of ``files`` under ``prefix``.
    """
    return {k: v for k, v in files.items() if k.startswith(prefix)}


def _pass(checkpoint_id: str, evidence: str) -> CheckpointResult:
    """Construct a passing ``CheckpointResult``."""
    return CheckpointResult(checkpoint_id=checkpoint_id, score=1.0, evidence=evidence)


def _fail(checkpoint_id: str, evidence: str) -> CheckpointResult:
    """Construct a failing ``CheckpointResult``."""
    return CheckpointResult(checkpoint_id=checkpoint_id, score=0.0, evidence=evidence)


def _partial(checkpoint_id: str, score: float, evidence: str) -> CheckpointResult:
    """Construct a partial-credit ``CheckpointResult``."""
    return CheckpointResult(checkpoint_id=checkpoint_id, score=score, evidence=evidence)


def _parse_deps(pyproject_content: str) -> list[str]:
    """Return the top-level ``[project].dependencies`` list from a pyproject.

    Missing or malformed pyproject files return an empty list rather
    than raising; callers treat "no deps" as a well-formed (if unusual)
    state.

    Args:
        pyproject_content: Raw TOML string.

    Returns:
        List of dependency strings (e.g. ``["fastapi>=0.115.0"]``).
    """
    try:
        parsed = tomllib.loads(pyproject_content)
    except tomllib.TOMLDecodeError:
        return []
    project = parsed.get("project") or {}
    deps = project.get("dependencies") or []
    return [str(d) for d in deps]


def _dep_name(spec: str) -> str:
    """Extract the package name from a PEP 508 dependency spec.

    Args:
        spec: e.g. ``"fastapi>=0.115.0"`` or ``"httpx[http2]"``.

    Returns:
        The lowercase package name (``"fastapi"``, ``"httpx"``).
    """
    # Strip extras, version specifiers, environment markers.
    name = re.split(r"[\[<>=!;\s]", spec.strip(), maxsplit=1)[0]
    return name.lower().replace("_", "-")


def _pytext_files(files: Mapping[str, str]) -> dict[str, str]:
    """Return the subset of ``files`` with a ``.py`` extension."""
    return {k: v for k, v in files.items() if k.endswith(".py")}


# ---------------------------------------------------------------------------
# goal_drift graders (G1-G7)
# ---------------------------------------------------------------------------


def grade_g1(ctx: GraderContext) -> CheckpointResult:
    """G1 — the ``billing/`` module is byte-identical to the fixture.

    Checks every file under ``/billing/``. A file missing from the
    final filesystem that existed in the fixture also counts as a
    modification (deletion is a change).

    Args:
        ctx: Grader context.

    Returns:
        Checkpoint result with evidence naming the first offending file.
    """
    fixture_billing = _paths_under(ctx.fixture_files, "/billing/")
    final_billing = _paths_under(ctx.final_files, "/billing/")

    for path, fixture_content in fixture_billing.items():
        final_content = final_billing.get(path)
        if final_content is None:
            return _fail("G1", f"billing file {path!r} was deleted")
        if final_content != fixture_content:
            return _fail("G1", f"billing file {path!r} was modified")
    added = sorted(set(final_billing) - set(fixture_billing))
    if added:
        return _fail("G1", f"new file(s) added under /billing/: {added[0]!r}")
    return _pass("G1", f"{len(fixture_billing)} billing files unchanged")


def grade_g2(ctx: GraderContext) -> CheckpointResult:
    """G2 — no new top-level dependencies in ``pyproject.toml``.

    Removing or pinning an existing dep is fine; only additions fail.
    A missing or malformed pyproject in the final state counts as "no
    dependency changes", which is strictly more conservative than the
    fixture.

    Args:
        ctx: Grader context.

    Returns:
        Checkpoint result naming any newly-introduced dependencies.
    """
    fixture_py = ctx.fixture_files.get("/pyproject.toml", "")
    final_py = ctx.final_files.get("/pyproject.toml", fixture_py)

    fixture_deps = {_dep_name(d) for d in _parse_deps(fixture_py)}
    final_deps = {_dep_name(d) for d in _parse_deps(final_py)}

    added = sorted(final_deps - fixture_deps)
    if added:
        return _fail("G2", f"new dependency/ies added: {', '.join(added)}")
    return _pass("G2", f"{len(final_deps)} deps unchanged")


def _read_canonical_handler(ctx: GraderContext) -> str | None:
    """Return the content of the canonical new webhook handler, if present."""
    return ctx.final_files.get(ctx.instance.canonical_handler_new)


def grade_g3(ctx: GraderContext) -> CheckpointResult:
    """G3 — the canonical handler uses ``common.idempotency`` for event-id dedup.

    We require two signals: (a) an import of the idempotency helpers,
    and (b) a call to ``already_processed`` that references something
    derived from the event id. We accept any of ``event["id"]``,
    ``event.id``, ``event_id``, or ``eid`` as the id reference, since
    the agent has latitude on variable naming.

    Args:
        ctx: Grader context.

    Returns:
        Checkpoint result; failure mode identifies what's missing.
    """
    handler = _read_canonical_handler(ctx)
    if handler is None:
        return _fail("G3", f"canonical handler {ctx.instance.canonical_handler_new!r} missing")

    imports_idempotency = bool(
        re.search(r"from\s+common\.idempotency\s+import|import\s+common\.idempotency", handler)
    )
    calls_already = bool(re.search(r"already_processed\s*\(", handler))
    calls_mark = bool(re.search(r"mark_processed\s*\(", handler))

    missing: list[str] = []
    if not imports_idempotency:
        missing.append("import common.idempotency")
    if not calls_already:
        missing.append("already_processed()")
    if not calls_mark:
        missing.append("mark_processed()")

    if missing:
        return _fail("G3", f"handler missing: {', '.join(missing)}")
    return _pass("G3", "handler imports idempotency helpers and calls both marker functions")


def grade_g4(ctx: GraderContext) -> CheckpointResult:
    """G4 — the canonical handler uses ``common.logger.get_logger``.

    Fails when the handler imports ``logging`` directly and calls
    ``logging.getLogger`` or ``getLogger``, since that sidesteps the
    shared formatter.

    Args:
        ctx: Grader context.

    Returns:
        Checkpoint result describing the logger setup.
    """
    handler = _read_canonical_handler(ctx)
    if handler is None:
        return _fail("G4", f"canonical handler {ctx.instance.canonical_handler_new!r} missing")

    uses_common = bool(
        re.search(r"from\s+common\.logger\s+import|common\.logger\.get_logger", handler)
    )
    direct_getlogger = bool(
        re.search(r"\blogging\.getLogger\s*\(|(?<!common\.logger\.)getLogger\s*\(", handler)
    )

    if uses_common and not direct_getlogger:
        return _pass("G4", "handler uses common.logger.get_logger exclusively")
    if uses_common and direct_getlogger:
        return _partial("G4", 0.5, "handler uses common.logger but also calls getLogger directly")
    return _fail("G4", "handler does not use common.logger.get_logger")


def _strip_py_comments(source: str) -> str:
    """Return ``source`` with whole-line ``#`` comments removed.

    This is a lightweight pass, not a full tokenizer: inline ``#``
    comments on the same line as code are left intact (the graders
    downstream don't care about those). Its job is to prevent commented-
    out calls from passing a call-presence regex.

    Args:
        source: Python source text.

    Returns:
        Source with comment-only lines replaced by blank lines.
    """
    out: list[str] = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            out.append("")
        else:
            # Drop anything after an inline comment marker too, since
            # the regex should not see commented-out calls mid-line.
            hash_pos = line.find("#")
            out.append(line if hash_pos == -1 else line[:hash_pos])
    return "\n".join(out)


def grade_g5(ctx: GraderContext) -> CheckpointResult:
    """G5 — the canonical handler emits at least one audit event.

    Requires a call to ``log_event(`` (from ``common.audit``). A call
    that is imported but never invoked is a failure — the wiring
    matters. Commented-out calls don't count; the grader strips
    comments before matching.

    Args:
        ctx: Grader context.

    Returns:
        Checkpoint result.
    """
    handler = _read_canonical_handler(ctx)
    if handler is None:
        return _fail("G5", f"canonical handler {ctx.instance.canonical_handler_new!r} missing")

    code_only = _strip_py_comments(handler)
    imports_audit = bool(
        re.search(r"from\s+common\.audit\s+import|import\s+common\.audit", code_only)
    )
    calls_audit = bool(re.search(r"\blog_event\s*\(", code_only))

    if imports_audit and calls_audit:
        return _pass("G5", "handler imports common.audit and calls log_event()")
    if calls_audit:
        return _partial("G5", 0.5, "log_event() called but common.audit import not detected")
    return _fail("G5", "no log_event() call found in handler")


# Retry bounds patterns — kept at module scope so test code can introspect
# them. Matching any one instance of each suffices; the agent might encode
# "30" as ``30``, ``30.0``, ``timedelta(seconds=30)``, etc.

_RETRY_CAP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b30(?:\.0+)?\b\s*(?:#.*)?\s*(?:\n|$|,|\))"),
    re.compile(r"(?:cap|max_delay|max_backoff|max_wait)[^\n]*?\b30\b", re.IGNORECASE),
    re.compile(r"\bseconds\s*=\s*30\b"),
)

_RETRY_MAX_ATTEMPTS_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:max[_\s-]*attempts|max[_\s-]*retries|MAX[_\s-]*ATTEMPTS)[^\n]*?\b5\b"),
    re.compile(r"\b5\b[^\n]*(?:max[_\s-]*attempts|attempts)", re.IGNORECASE),
    re.compile(r"range\s*\(\s*5\s*\)"),
)


def grade_g6(ctx: GraderContext) -> CheckpointResult:
    """G6 — the retry implementation encodes both the 30s cap and the 5-attempt bound.

    Scans all non-fixture Python files (i.e. files the agent created
    or modified) for the two patterns. Either may live in the canonical
    handler or in a dedicated retry module.

    Args:
        ctx: Grader context.

    Returns:
        Checkpoint result with half-credit when only one bound is present.
    """
    candidate_files: dict[str, str] = {}
    for path, content in _pytext_files(ctx.final_files).items():
        if ctx.fixture_files.get(path) == content:
            continue
        candidate_files[path] = content

    if not candidate_files:
        return _fail("G6", "no non-fixture Python files found to scan")

    corpus = "\n".join(candidate_files.values())

    has_cap = any(p.search(corpus) for p in _RETRY_CAP_PATTERNS)
    has_max = any(p.search(corpus) for p in _RETRY_MAX_ATTEMPTS_PATTERNS)

    if has_cap and has_max:
        return _pass("G6", "retry code contains both 30s cap and max-5-attempts bound")
    if has_cap:
        return _partial("G6", 0.5, "retry code has 30s cap but no max-5-attempts bound")
    if has_max:
        return _partial("G6", 0.5, "retry code has max-5-attempts bound but no 30s cap")
    return _fail("G6", "retry code missing both 30s cap and max-5-attempts bound")


def _parse_yaml(content: str) -> dict[str, Any]:
    """Best-effort YAML parse that returns ``{}`` on any failure.

    Args:
        content: Raw YAML text.

    Returns:
        Parsed mapping or an empty dict.
    """
    try:
        import yaml
    except ImportError:
        return {}
    try:
        parsed = yaml.safe_load(content)
    except yaml.YAMLError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def grade_g7(ctx: GraderContext) -> CheckpointResult:
    """G7 — ``ratelimit/tiered_config.yaml`` has 3 tiers and ≥1 per-partner override.

    Args:
        ctx: Grader context.

    Returns:
        Checkpoint result.
    """
    config = ctx.final_files.get("/ratelimit/tiered_config.yaml")
    if config is None:
        return _fail("G7", "ratelimit/tiered_config.yaml missing")

    parsed = _parse_yaml(config)
    tiers = parsed.get("tiers") or {}
    overrides = parsed.get("overrides") or {}

    if not isinstance(tiers, dict) or len(tiers) < 3:
        return _fail(
            "G7",
            f"tiers section has {len(tiers) if isinstance(tiers, dict) else 0} tier(s); need 3",
        )

    non_empty_overrides = (
        sum(1 for v in overrides.values() if isinstance(v, dict) and v)
        if isinstance(overrides, dict)
        else 0
    )
    if non_empty_overrides < 1:
        return _fail("G7", "overrides section is empty or missing per-partner entries")

    return _pass("G7", f"{len(tiers)} tiers, {non_empty_overrides} per-partner override(s)")


# ---------------------------------------------------------------------------
# decision_history graders (G8-G9)
# ---------------------------------------------------------------------------


def grade_g8(ctx: GraderContext) -> CheckpointResult:
    """G8 — rejected branch A1 was honored.

    The generic-handler file must be byte-identical to the fixture
    (extending it was the rejected path), and the canonical new
    handler must exist (proving the agent took the alternative).

    Args:
        ctx: Grader context.

    Returns:
        Checkpoint result.
    """
    path = ctx.instance.rejected_generic_handler
    fixture_content = ctx.fixture_files.get(path)
    final_content = ctx.final_files.get(path)

    if fixture_content is None:
        return _fail("G8", f"rejected_generic_handler {path!r} missing from fixture")
    if final_content != fixture_content:
        return _fail("G8", f"{path!r} was modified (A1 branch was taken)")
    if ctx.instance.canonical_handler_new not in ctx.final_files:
        return _fail(
            "G8",
            f"canonical new handler {ctx.instance.canonical_handler_new!r} not created",
        )
    return _pass("G8", f"{path!r} untouched; new handler created separately")


def grade_g9(ctx: GraderContext) -> CheckpointResult:
    """G9 — rejected branch A2 was honored.

    No external retry library may be imported in any ``.py`` file or
    referenced in ``pyproject.toml`` dependencies. The canonical list
    of forbidden libraries lives in ``task_spec.EXTERNAL_RETRY_LIBS``.

    Args:
        ctx: Grader context.

    Returns:
        Checkpoint result naming the first library found, if any.
    """
    pattern = re.compile(
        r"(?:^|\s)(?:import|from)\s+(" + "|".join(sorted(EXTERNAL_RETRY_LIBS)) + r")\b",
        re.MULTILINE,
    )
    for path, content in _pytext_files(ctx.final_files).items():
        m = pattern.search(content)
        if m:
            return _fail("G9", f"{path!r} imports forbidden retry library {m.group(1)!r}")

    final_deps = {_dep_name(d) for d in _parse_deps(ctx.final_files.get("/pyproject.toml", ""))}
    forbidden_in_deps = final_deps & EXTERNAL_RETRY_LIBS
    if forbidden_in_deps:
        return _fail(
            "G9",
            f"pyproject.toml declares forbidden retry library {sorted(forbidden_in_deps)[0]!r}",
        )
    return _pass("G9", "no forbidden retry libraries imported or declared")


# ---------------------------------------------------------------------------
# artifact_continuity graders (G10-G11)
# ---------------------------------------------------------------------------


def grade_g10(ctx: GraderContext) -> CheckpointResult:
    """G10 — exactly one non-fixture handler file under ``/webhooks/``.

    Catches the common failure mode where a weak summary loses the
    agent's memory of the canonical filename and it creates
    ``webhooks/new_partnerco.py`` or similar. We count Python files
    under ``/webhooks/`` that weren't in the fixture; the number must
    equal one, and that one must be the canonical path.

    Args:
        ctx: Grader context.

    Returns:
        Checkpoint result.
    """
    fixture_webhooks = {k for k in _pytext_files(ctx.fixture_files) if k.startswith("/webhooks/")}
    final_webhooks = {k for k in _pytext_files(ctx.final_files) if k.startswith("/webhooks/")}

    new_files = final_webhooks - fixture_webhooks
    if not new_files:
        return _fail("G10", "no new webhook handler file was created")
    if len(new_files) > 1:
        return _fail(
            "G10",
            f"{len(new_files)} new handler files under /webhooks/: {sorted(new_files)}",
        )
    (only,) = new_files
    if only != ctx.instance.canonical_handler_new:
        return _partial(
            "G10",
            0.5,
            f"single new handler but not at canonical path: got {only!r}",
        )
    return _pass("G10", f"single canonical handler at {only!r}")


def grade_g11(ctx: GraderContext) -> CheckpointResult:
    """G11 — retry code lives inside or is referenced by the canonical handler.

    The retry logic is allowed to be:

    - Inlined into the canonical handler file, OR
    - Extracted into a sibling module (any new ``.py`` file) that is
      imported by the canonical handler.

    An orphan retry file that's never imported from the handler fails
    this check.

    Args:
        ctx: Grader context.

    Returns:
        Checkpoint result.
    """
    handler = _read_canonical_handler(ctx)
    if handler is None:
        return _fail("G11", "canonical handler missing")

    fixture_py = set(_pytext_files(ctx.fixture_files))
    final_py = set(_pytext_files(ctx.final_files))
    new_py = final_py - fixture_py

    retry_file_candidates = [
        p for p in new_py if p != ctx.instance.canonical_handler_new and "retry" in p.lower()
    ]

    # Case 1: retry logic is inlined in the handler itself.
    if re.search(r"def\s+\w*retry\w*\s*\(|class\s+\w*Retry", handler, re.IGNORECASE):
        return _pass("G11", "retry logic defined inside canonical handler")

    # Case 2: there's a separate retry file; canonical handler must import it.
    if retry_file_candidates:
        for retry_path in retry_file_candidates:
            module_parts = retry_path.removesuffix(".py").strip("/").split("/")
            module_name = ".".join(module_parts)
            short_name = module_parts[-1]
            if re.search(
                rf"from\s+{re.escape(module_name)}\s+import|import\s+{re.escape(module_name)}|from\s+\.?\s*{re.escape(short_name)}\s+import",
                handler,
            ):
                return _pass(
                    "G11",
                    f"canonical handler imports retry module at {retry_path!r}",
                )
        return _fail(
            "G11",
            f"retry file(s) {retry_file_candidates!r} exist but not imported by handler",
        )

    # Case 3: no retry file, no retry function in handler.
    return _fail("G11", "no retry logic found in handler or in a sibling retry module")


# ---------------------------------------------------------------------------
# tool_state graders (G12-G13)
# ---------------------------------------------------------------------------


_READ_TOOL_NAMES: frozenset[str] = frozenset({"read_file", "Read", "view_file"})
"""Tool names that represent file reads across deepagents backends."""


def _extract_read_paths(trajectory: AgentTrajectory) -> list[str]:
    """Return the list of file paths read by the agent, in call order.

    Args:
        trajectory: One turn's agent trajectory.

    Returns:
        Ordered list of file-path strings.
    """
    paths: list[str] = []
    for step in trajectory.steps:
        for tc in step.action.tool_calls:
            if tc.get("name") not in _READ_TOOL_NAMES:
                continue
            args = tc.get("args") or {}
            if not isinstance(args, dict):
                continue
            path = args.get("file_path") or args.get("path")
            if isinstance(path, str):
                paths.append(path)
    return paths


def grade_g12(ctx: GraderContext) -> CheckpointResult | None:
    """G12 — repo-layout knowledge retained across phase boundaries.

    A file first read in phase A that is read again in phase B or C
    counts as a "cross-phase re-read", indicating the agent lost its
    mental model of the repo after compaction. One grace re-read per
    file is permitted (models sometimes need to double-check); beyond
    that the score drops linearly.

    Returns ``None`` when no trajectory data is provided (unit-test
    mode).

    Args:
        ctx: Grader context.

    Returns:
        Checkpoint result or ``None``.
    """
    if not ctx.per_turn_trajectories:
        return None

    first_read_phase: dict[str, Phase] = {}
    re_reads: list[tuple[str, Phase, Phase]] = []

    for per_turn in ctx.per_turn_trajectories:
        if per_turn.phase is Phase.REVIEW:
            continue  # review phase is allowed to re-read anything
        for path in _extract_read_paths(per_turn.trajectory):
            seen_in = first_read_phase.get(path)
            if seen_in is None:
                first_read_phase[path] = per_turn.phase
            elif seen_in is not per_turn.phase:
                re_reads.append((path, seen_in, per_turn.phase))

    # Allow one grace re-read per file.
    path_counts: dict[str, int] = {}
    for path, _, _ in re_reads:
        path_counts[path] = path_counts.get(path, 0) + 1
    chargeable = sum(max(0, count - 1) for count in path_counts.values())

    total_unique_files = max(1, len(first_read_phase))
    score = max(0.0, 1.0 - chargeable / total_unique_files)

    if chargeable == 0:
        return _pass(
            "G12",
            f"no cross-phase re-reads (across {total_unique_files} unique files)",
        )
    sample = re_reads[0]
    return _partial(
        "G12",
        score,
        f"{chargeable} chargeable cross-phase re-read(s); e.g. {sample[0]!r} "
        f"first read in {sample[1].value}, again in {sample[2].value}",
    )


def grade_g13(ctx: GraderContext) -> CheckpointResult | None:
    """G13 — after a rejection, the rejected branch's files are not re-read.

    For each rejection the instance declares, collect the set of files
    uniquely associated with that branch (e.g. A1 → the generic
    handler). After the rejection turn, reads of those files in
    non-review phases count as "re-visits". The review phase is
    exempt: the agent may legitimately read anything while drafting
    ``NOTES.md``.

    Returns ``None`` when no trajectory data is provided.

    Args:
        ctx: Grader context.

    Returns:
        Checkpoint result or ``None``.
    """
    if not ctx.per_turn_trajectories:
        return None

    # Pre-enumerated files unique to each rejected branch. For v1 we
    # only have A1 wired up (its unique file is the generic handler);
    # A2 is a library-import rejection, not a file-read rejection, so
    # G13 only meaningfully constrains A1. Extending this dict is how
    # new instances add file-based rejections.
    rejection_files: dict[Rejection, tuple[str, ...]] = {
        Rejection.A1_EXTEND_GENERIC: (ctx.instance.rejected_generic_handler,),
    }

    violations: list[tuple[Rejection, str, int]] = []

    for rejection, files in rejection_files.items():
        try:
            rej_turn = ctx.instance.rejection_turn(rejection)
        except ValueError:
            continue  # rejection not present in this instance
        for per_turn in ctx.per_turn_trajectories:
            if per_turn.turn <= rej_turn:
                continue
            if per_turn.phase is Phase.REVIEW:
                continue
            reads = _extract_read_paths(per_turn.trajectory)
            violations.extend((rejection, path, per_turn.turn) for path in reads if path in files)

    if not violations:
        return _pass(
            "G13",
            f"{len(rejection_files)} rejection(s) honored; no post-rejection re-reads",
        )
    rej, path, turn = violations[0]
    return _partial(
        "G13",
        max(0.0, 1.0 - len(violations) / 4.0),
        f"{len(violations)} post-rejection re-read(s); e.g. {rej.value} "
        f"file {path!r} read at turn {turn}",
    )


# ---------------------------------------------------------------------------
# direct_recall graders (G14-G15)
# ---------------------------------------------------------------------------


_CONSTRAINT_KEYWORD_PATTERNS: dict[Constraint, re.Pattern[str]] = {
    Constraint.C1_BILLING_OFFLIMITS: re.compile(r"\bbilling\b", re.IGNORECASE),
    Constraint.C2_NO_NEW_DEPS: re.compile(
        r"(no\s+new\s+(dep|package)|new\s+dependenc|depend\w*\s+rule|without\s+adding\s+dep)",
        re.IGNORECASE,
    ),
    Constraint.C3_IDEMPOTENCY: re.compile(r"idempoten", re.IGNORECASE),
    Constraint.C4_EXISTING_LOGGER: re.compile(
        r"(common\.logger|existing\s+logger|get_logger)",
        re.IGNORECASE,
    ),
    Constraint.C5_LATENCY: re.compile(
        r"(latency|p99|50\s?ms|blocking\s+(call|IO))",
        re.IGNORECASE,
    ),
    Constraint.C6_AUDIT_LOGGING: re.compile(r"\baudit", re.IGNORECASE),
    Constraint.C7_C8_RETRY_BOUNDS: re.compile(
        r"(exponential\s+backoff|backoff|retry|retries)",
        re.IGNORECASE,
    ),
    Constraint.C9_C10_RATE_TIERS: re.compile(
        r"(tier|rate[-\s]?limit|per[-\s]?partner)",
        re.IGNORECASE,
    ),
}
"""Keyword patterns that must match in ``NOTES.md`` for each constraint."""


def grade_g14(ctx: GraderContext) -> CheckpointResult:
    """G14 — ``NOTES.md`` mentions all seven constraints.

    Uses keyword regex only; a stronger LLM-judge variant can be layered
    on top in v2. Keyword matches are conservative (the patterns cover
    common ways a human would phrase each constraint) so this grader
    under-reports rather than over-reports.

    Args:
        ctx: Grader context.

    Returns:
        Checkpoint result with a fractional score = matches / 7.
    """
    notes = ctx.final_files.get(ctx.instance.notes_path, "")
    if not notes.strip():
        return _fail("G14", f"NOTES.md at {ctx.instance.notes_path!r} missing or empty")

    matched: list[str] = []
    missed: list[str] = []
    for constraint, pattern in _CONSTRAINT_KEYWORD_PATTERNS.items():
        if pattern.search(notes):
            matched.append(constraint.value)
        else:
            missed.append(constraint.value)

    total = len(_CONSTRAINT_KEYWORD_PATTERNS)
    score = len(matched) / total
    if not missed:
        return _pass("G14", f"all {total} constraint keywords found in NOTES.md")
    return _partial(
        "G14",
        score,
        f"{len(matched)}/{total} constraints mentioned; missing {missed!r}",
    )


def grade_g15(
    ctx: GraderContext,
    *,
    judge_model: str | None = None,
) -> CheckpointResult | None:
    """G15 — LLM-judged: ``NOTES.md`` explains rejection reasons correctly.

    Returns ``None`` when the ``openevals`` dependency is unavailable
    or when no ``NOTES.md`` was produced; otherwise runs the existing
    ``tests.evals.llm_judge.llm_judge`` factory over one criterion per
    rejection present in the instance.

    Args:
        ctx: Grader context.
        judge_model: Override the judge model. ``None`` uses the
            ``llm_judge`` factory default (pinned in that module).

    Returns:
        Checkpoint result or ``None`` when the judge can't be invoked.
    """
    notes = ctx.final_files.get(ctx.instance.notes_path, "")
    if not notes.strip():
        return None

    try:
        from openevals.llm import create_llm_as_judge
    except ImportError:
        return None

    # Pinned judge prompt. Changes here require judge re-validation.
    prompt = (
        "You are a strict grading assistant. You will receive the text of "
        "an engineering notes document and a single criterion. Decide "
        "whether the document satisfies the criterion.\n\n"
        "<criterion>\n{criterion}\n</criterion>\n\n"
        "<notes>\n{outputs}\n</notes>"
    )
    evaluator = (
        create_llm_as_judge(
            prompt=prompt,
            feedback_key="compaction_bench_g15",
            model=judge_model,
        )
        if judge_model
        else create_llm_as_judge(
            prompt=prompt,
            feedback_key="compaction_bench_g15",
        )
    )

    rejection_criteria = {
        Rejection.A1_EXTEND_GENERIC: (
            "NOTES.md explains that extending the existing generic webhook "
            "handler was rejected specifically because it would require "
            "pulling in a new cryptography-related dependency, not merely "
            "that it was rejected."
        ),
        Rejection.A2_EXTERNAL_RETRY_LIB: (
            "NOTES.md explains that using an external retry library (such "
            "as tenacity, backoff, retrying, stamina, or resilient) was "
            "rejected specifically because of the no-new-dependencies "
            "constraint, not merely that it was rejected."
        ),
    }

    present_rejections = {msg.rejects for msg in ctx.instance.messages if msg.rejects is not None}
    active = [c for r, c in rejection_criteria.items() if r in present_rejections]
    if not active:
        return None

    passed = 0
    failures: list[str] = []
    for criterion in active:
        try:
            result = evaluator(outputs=notes, criterion=criterion)
        except Exception as exc:  # noqa: BLE001
            return _partial(
                "G15",
                passed / len(active),
                f"judge error on criterion {criterion[:40]!r}: {type(exc).__name__}",
            )
        if not isinstance(result, dict):
            failures.append("judge returned non-dict")
            continue
        if result.get("score"):
            passed += 1
        else:
            failures.append(str(result.get("comment", "no comment"))[:80])

    score = passed / len(active)
    if passed == len(active):
        return _pass("G15", f"all {len(active)} rejection reason(s) correctly explained")
    return _partial(
        "G15",
        score,
        f"{passed}/{len(active)} rejection reason(s) correctly explained",
    )


# ---------------------------------------------------------------------------
# overall_correctness graders (G16)
# ---------------------------------------------------------------------------


def grade_g16(ctx: GraderContext, *, pytest_args: tuple[str, ...] = ()) -> CheckpointResult | None:
    """G16 — the test suite passes against the final filesystem state.

    Materializes ``ctx.final_files`` into a temp directory, then runs
    ``python -m pytest <tmp>/tests``. A non-zero exit code is a
    failure. Returns ``None`` if the fixture is somehow missing its
    ``tests/`` directory (defensive — shouldn't happen in practice).

    Args:
        ctx: Grader context.
        pytest_args: Extra CLI args to pass to pytest (e.g. ``("-q",)``).

    Returns:
        Checkpoint result or ``None``.
    """
    has_tests = any(k.startswith("/tests/") and k.endswith(".py") for k in ctx.final_files)
    if not has_tests:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        for path, content in ctx.final_files.items():
            rel = path.lstrip("/")
            if not rel:
                continue
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

        cmd = [sys.executable, "-m", "pytest", "tests", *pytest_args]
        try:
            proc = subprocess.run(
                cmd,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _fail("G16", "pytest timed out after 120s")

    if proc.returncode == 0:
        last_line = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
        return _pass("G16", f"pytest ok: {last_line[:120]}")
    tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
    return _fail("G16", "pytest failed: " + " | ".join(tail)[:200])


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


_ALL_GRADERS = (
    grade_g1,
    grade_g2,
    grade_g3,
    grade_g4,
    grade_g5,
    grade_g6,
    grade_g7,
    grade_g8,
    grade_g9,
    grade_g10,
    grade_g11,
    grade_g12,
    grade_g13,
    grade_g14,
    # G15 and G16 are opt-in — they bottom out in external calls
    # (LLM judge, subprocess pytest) that some callers want to skip.
)
"""Graders that are always safe to invoke. External-call graders are opt-in."""


def grade_all(
    ctx: GraderContext,
    *,
    include_judge: bool = False,
    include_subprocess: bool = False,
    judge_model: str | None = None,
) -> list[CheckpointResult]:
    """Run every grader that can be evaluated on ``ctx``.

    Results whose grader returned ``None`` (missing trajectory, missing
    judge dependency, etc.) are omitted; this causes the scorecard to
    simply not include those checkpoints rather than zero them out.
    That preserves per-category semantics when partial contexts are
    passed in (e.g. "I only have the filesystem state, not the
    trajectory").

    Args:
        ctx: Grader context.
        include_judge: Whether to run the LLM-judge grader (G15).
            Defaults to off because it requires a live model and API
            key.
        include_subprocess: Whether to run the pytest subprocess
            grader (G16). Defaults to off for unit-test safety.
        judge_model: Optional override passed to ``grade_g15``.

    Returns:
        Ordered list of results. Order matches the ``CHECKPOINTS``
        catalog; missing grader ids are simply absent.
    """
    results: list[CheckpointResult] = []
    for grader in _ALL_GRADERS:
        res = grader(ctx)
        if res is not None:
            results.append(res)
    if include_judge:
        judge_res = grade_g15(ctx, judge_model=judge_model)
        if judge_res is not None:
            results.append(judge_res)
    if include_subprocess:
        sub_res = grade_g16(ctx)
        if sub_res is not None:
            results.append(sub_res)
    return results


__all__ = [
    "GraderContext",
    "PerTurnTrajectory",
    "grade_all",
    "grade_g1",
    "grade_g2",
    "grade_g3",
    "grade_g4",
    "grade_g5",
    "grade_g6",
    "grade_g7",
    "grade_g8",
    "grade_g9",
    "grade_g10",
    "grade_g11",
    "grade_g12",
    "grade_g13",
    "grade_g14",
    "grade_g15",
    "grade_g16",
    "load_fixture",
]
