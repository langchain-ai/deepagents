"""Harbor runner: execute one harness version against a set of cases."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any

from better_harness.core import Case, Experiment, SplitResult, Trace, slug
from better_harness.traces import TRACE_ENV, load_trace


def run_split(
    *,
    experiment: Experiment,
    harness_path: Path,
    split: str,
    output_dir: Path,
    reuse_existing: bool = False,
) -> SplitResult:
    """Run all cases for one split and return results with traces."""
    cases = [c for c in experiment.cases if c.split == split]
    traces: list[Trace] = []
    for case in cases:
        trace = run_case(
            experiment=experiment,
            harness_path=harness_path,
            case=case,
            case_dir=output_dir / slug(case.id),
            reuse_existing=reuse_existing,
        )
        traces.append(trace)
    return SplitResult(split=split, traces=traces)


def run_case(
    *,
    experiment: Experiment,
    harness_path: Path,
    case: Case,
    case_dir: Path,
    reuse_existing: bool = False,
) -> Trace:
    """Run one Harbor case and return a Trace (score + execution record)."""
    trace_path = case_dir / "trace.json"
    result_path = case_dir / "result.json"

    if reuse_existing and trace_path.exists() and result_path.exists():
        score, failure = _read_score(result_path, experiment.runner_config)
        return load_trace(trace_path, case_id=case.id, split=case.split, score=score, failure=failure)

    case_dir.mkdir(parents=True, exist_ok=True)
    cfg = experiment.runner_config

    command = _build_command(
        cfg=cfg,
        harness_path=harness_path,
        case_id=case.id,
        jobs_dir=case_dir / "jobs",
        job_name=slug(case.id),
    )

    env = _build_env(harness_path=harness_path, trace_path=trace_path)
    (case_dir / "command.json").write_text(
        json.dumps({"argv": command, "shell": shlex.join(command)}, indent=2) + "\n"
    )

    completed = subprocess.run(command, env=env, capture_output=True, check=False, text=True)
    (case_dir / "stdout.log").write_text(completed.stdout)
    (case_dir / "stderr.log").write_text(completed.stderr)

    # Harbor writes result.json somewhere inside jobs_dir — find and copy it up.
    harbor_result = _find_result_json(case_dir / "jobs")
    if harbor_result and harbor_result != result_path:
        result_path.write_text(harbor_result.read_text())

    score, failure = _read_score(result_path, cfg)
    return load_trace(trace_path, case_id=case.id, split=case.split, score=score, failure=failure)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_command(
    *,
    cfg: dict[str, Any],
    harness_path: Path,
    case_id: str,
    jobs_dir: Path,
    job_name: str,
) -> list[str]:
    command = list(cfg["command"])
    command.extend([
        "run",
        "-p", str(cfg["tasks_root"]),
        "--task-name", case_id,
        "-l", "1",
        "-n", str(cfg.get("concurrency", 1)),
        "-o", str(jobs_dir),
        "--job-name", job_name,
    ])
    # Agent import path: explicit config wins; otherwise derive from harness filename.
    agent_path = cfg.get("agent_import_path") or f"{harness_path.stem}:HarborAgent"
    command.extend(["--agent-import-path", agent_path])
    command.extend(str(a) for a in cfg.get("extra_args", []))
    return command


def _build_env(*, harness_path: Path, trace_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    # Tell the harness fixed boundary where to write trace.json.
    env[TRACE_ENV] = str(trace_path)
    # Ensure the harness directory is importable.
    existing = env.get("PYTHONPATH", "")
    harness_dir = str(harness_path.parent)
    env["PYTHONPATH"] = harness_dir + (os.pathsep + existing if existing else "")
    return env


def _find_result_json(jobs_dir: Path) -> Path | None:
    """Find the first result.json Harbor wrote inside jobs_dir."""
    if not jobs_dir.exists():
        return None
    for path in sorted(jobs_dir.rglob("result.json")):
        return path
    return None


def _read_score(result_path: Path, cfg: dict[str, Any]) -> tuple[float, str | None]:
    """Return (score, failure_message) from a Harbor result.json."""
    threshold = float(cfg.get("pass_threshold", 1.0))
    if not result_path.exists():
        return 0.0, "no result.json found"
    try:
        data = json.loads(result_path.read_text())
        score = float(data.get("score", data.get("reward", 0.0)))
        failure = None if score >= threshold else str(data.get("message", "score below threshold"))
    except (json.JSONDecodeError, ValueError, KeyError):
        return 0.0, "could not parse result.json"
    else:
        return score, failure
