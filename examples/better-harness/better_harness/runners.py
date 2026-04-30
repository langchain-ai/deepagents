"""Pytest and Harbor eval runners."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from better_harness.core import (
    CaseOutcome,
    EvalCase,
    Experiment,
    RunLayout,
    SplitResult,
    Variant,
    extract_trace_refs,
    write_trace_refs,
)
from better_harness.patching import (
    VARIANT_ENV,
    ensure_sitecustomize,
    prepend_pythonpath,
    workspace_override_context,
)


class PytestRunner:
    """Run explicit pytest subsets against one variant."""

    def __init__(self, *, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parents[1]

    def collect_inventory(self, experiment: Experiment) -> list[str]:
        """Collect pytest nodeids."""
        project_root = Path(str(experiment.runner_config["project_root"]))
        command = self._base_command(experiment)
        command.extend(["--collect-only"])
        command.extend(str(arg) for arg in experiment.runner_config.get("pytest_args", ["-q"]))
        env = os.environ.copy()
        runtime_dir = ensure_sitecustomize(self.repo_root / ".runtime")
        env["PYTHONPATH"] = prepend_pythonpath(
            [runtime_dir, self.repo_root, experiment.workspace_root],
            env.get("PYTHONPATH"),
        )
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=env,
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode != 0:
            msg = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"pytest collection failed: {msg}")
        return [
            line.strip()
            for line in completed.stdout.splitlines()
            if "::" in line and line.strip()
        ]

    def run_split(
        self,
        *,
        experiment: Experiment,
        variant: Variant,
        split: str,
        layout: RunLayout,
        reuse_existing: bool = False,
    ) -> SplitResult:
        """Run one split and capture artifacts."""
        split_dir = layout.split_dir(variant_key=variant.key, split=split)
        result_path = split_dir / "result.json"
        if reuse_existing and result_path.exists():
            return SplitResult.load(result_path)

        split_dir.mkdir(parents=True, exist_ok=True)
        variant_path = layout.variant_path(variant.key)
        variant.save(variant_path)
        project_root = Path(str(experiment.runner_config["project_root"]))
        env = self._build_env(
            experiment=experiment,
            variant_path=variant_path,
            runtime_dir=ensure_sitecustomize(layout.runtime_dir),
        )

        outcomes: list[CaseOutcome] = []
        returncodes: list[int] = []
        split_stdout: list[str] = []
        split_stderr: list[str] = []

        with workspace_override_context(experiment.workspace_root, variant.file_overrides()):
            for case in experiment.cases_for_split(split):
                rendered = case.render(model=experiment.model)
                case_slug = safe_slug(rendered)
                case_dir = split_dir / "cases" / case_slug
                case_dir.mkdir(parents=True, exist_ok=True)
                summary_path = case_dir / "summary.json"
                junit_path = case_dir / "junit.xml"
                command = self._base_command(experiment)
                if summary_flag := experiment.runner_config.get("summary_flag", "--evals-report-file"):
                    command.extend([str(summary_flag), str(summary_path)])
                command.extend(["--junitxml", str(junit_path)])
                command.extend(str(arg) for arg in experiment.runner_config.get("pytest_args", ["-q"]))
                command.append(rendered)

                (case_dir / "command.json").write_text(
                    json.dumps(
                        {
                            "argv": command,
                            "shell": shlex.join(command),
                            "cwd": str(project_root),
                            "env": {
                                VARIANT_ENV: str(variant_path),
                                "PYTHONPATH": env["PYTHONPATH"],
                                "LANGSMITH_TEST_SUITE": env["LANGSMITH_TEST_SUITE"],
                            },
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )

                completed = subprocess.run(
                    command,
                    cwd=project_root,
                    env=env,
                    capture_output=True,
                    check=False,
                    text=True,
                )
                (case_dir / "stdout.log").write_text(completed.stdout)
                (case_dir / "stderr.log").write_text(completed.stderr)
                split_stdout.append(f"## {rendered}\n{completed.stdout}")
                split_stderr.append(f"## {rendered}\n{completed.stderr}")
                returncodes.append(completed.returncode)

                case_outcome = parse_pytest_outcomes(
                    junit_path=junit_path,
                    cases=[case],
                    model=experiment.model,
                    artifacts_dir=case_dir,
                )[0]

                if summary_path.exists():
                    summary_payload: dict[str, Any] | None = json.loads(summary_path.read_text())
                else:
                    summary_payload = {
                        "passed": 1 if case_outcome.passed else 0,
                        "total": 1,
                        "correctness": 1.0 if case_outcome.passed else 0.0,
                    }
                    summary_path.write_text(json.dumps(summary_payload, indent=2) + "\n")

                trace_refs = extract_trace_refs(
                    payload=summary_payload,
                    stdout=completed.stdout,
                    stderr=completed.stderr,
                )
                write_trace_refs(case_dir, trace_refs)
                case_outcome = CaseOutcome(
                    case_id=case_outcome.case_id,
                    split=case_outcome.split,
                    stratum=case_outcome.stratum,
                    status=case_outcome.status,
                    score=case_outcome.score,
                    duration_s=case_outcome.duration_s,
                    failure_message=case_outcome.failure_message,
                    artifacts_dir=str(case_dir),
                    trace_ref=trace_refs[0] if trace_refs else None,
                )
                outcomes.append(case_outcome)

        (split_dir / "stdout.log").write_text("\n\n".join(split_stdout))
        (split_dir / "stderr.log").write_text("\n\n".join(split_stderr))
        passed = sum(1 for outcome in outcomes if outcome.passed)
        total = len(outcomes)
        summary_payload = {
            "passed": passed,
            "failed": sum(1 for outcome in outcomes if outcome.status == "failed"),
            "skipped": sum(1 for outcome in outcomes if outcome.status == "skipped"),
            "total": total,
            "correctness": 0.0 if total == 0 else passed / total,
        }
        (split_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")

        result = SplitResult(
            split=split,
            variant=variant.key,
            model=experiment.model,
            passed=passed,
            total=total,
            score=float(passed),
            returncode=max(returncodes) if returncodes else 0,
            run_dir=str(split_dir),
            outcomes=tuple(outcomes),
        )
        result.save(result_path)
        return result

    def _build_env(self, *, experiment: Experiment, variant_path: Path, runtime_dir: Path) -> dict[str, str]:
        env = os.environ.copy()
        env[VARIANT_ENV] = str(variant_path)
        env["PYTHONPATH"] = prepend_pythonpath(
            [runtime_dir, self.repo_root, experiment.workspace_root],
            env.get("PYTHONPATH"),
        )
        env.setdefault("LANGSMITH_TEST_SUITE", f"better-harness-{experiment.name}")
        return env

    def _base_command(self, experiment: Experiment) -> list[str]:
        project_root = Path(str(experiment.runner_config["project_root"]))
        command = ["uv", "run", "--project", str(project_root)]
        if group := experiment.runner_config.get("uv_group", "test"):
            command.extend(["--group", str(group)])
        command.append("pytest")
        if model_flag := experiment.runner_config.get("model_flag"):
            command.extend([str(model_flag), experiment.model])
        command.extend(["-p", "better_harness_plugin"])
        return command


class HarborRunner:
    """Run Harbor tasks one case at a time."""

    def __init__(self, *, repo_root: Path | None = None) -> None:
        self.repo_root = repo_root or Path(__file__).resolve().parents[1]

    def collect_inventory(self, experiment: Experiment) -> list[str]:
        """Collect Harbor task names by scanning the tasks directory."""
        tasks_root = Path(str(experiment.runner_config["tasks_root"]))
        inventory = [
            str(path.parent.relative_to(tasks_root))
            for path in tasks_root.rglob("task.toml")
        ]
        return sorted(inventory)

    def run_split(
        self,
        *,
        experiment: Experiment,
        variant: Variant,
        split: str,
        layout: RunLayout,
        reuse_existing: bool = False,
    ) -> SplitResult:
        """Run one Harbor split."""
        split_dir = layout.split_dir(variant_key=variant.key, split=split)
        result_path = split_dir / "result.json"
        if reuse_existing and result_path.exists():
            return SplitResult.load(result_path)

        split_dir.mkdir(parents=True, exist_ok=True)
        variant_path = layout.variant_path(variant.key)
        variant.save(variant_path)
        cases = experiment.cases_for_split(split)
        outcomes: list[CaseOutcome] = []
        returncodes: list[int] = []

        for case in cases:
            rendered = case.render(model=experiment.model)
            case_slug = safe_slug(rendered)
            case_dir = split_dir / case_slug
            case_dir.mkdir(parents=True, exist_ok=True)
            jobs_dir = case_dir / "jobs"
            command = self._build_command(
                experiment=experiment,
                task_name=rendered,
                jobs_dir=jobs_dir,
                job_name=case_slug,
            )
            env = os.environ.copy()
            runtime_dir = ensure_sitecustomize(layout.runtime_dir)
            env[VARIANT_ENV] = str(variant_path)
            env["BETTER_HARNESS_WORKSPACE_ROOT"] = str(experiment.workspace_root)
            env["PYTHONPATH"] = prepend_pythonpath(
                [runtime_dir, self.repo_root, experiment.workspace_root],
                env.get("PYTHONPATH"),
            )
            (case_dir / "command.json").write_text(
                json.dumps(
                    {
                        "argv": command,
                        "shell": shlex.join(command),
                        "cwd": str(experiment.workspace_root),
                        "env": {
                            VARIANT_ENV: str(variant_path),
                            "BETTER_HARNESS_WORKSPACE_ROOT": str(experiment.workspace_root),
                            "PYTHONPATH": env["PYTHONPATH"],
                        },
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            with workspace_override_context(experiment.workspace_root, variant.file_overrides()):
                completed = subprocess.run(
                    command,
                    cwd=experiment.workspace_root,
                    env=env,
                    capture_output=True,
                    check=False,
                    text=True,
                )
            (case_dir / "stdout.log").write_text(completed.stdout)
            (case_dir / "stderr.log").write_text(completed.stderr)
            returncodes.append(completed.returncode)

            score, payload, failure_message = parse_harbor_case(
                jobs_dir=jobs_dir,
                pass_threshold=float(experiment.runner_config.get("pass_threshold", 1.0)),
            )
            trace_refs = extract_trace_refs(
                payload=payload,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
            write_trace_refs(case_dir, trace_refs)
            status = "passed" if score >= float(experiment.runner_config.get("pass_threshold", 1.0)) else "failed"
            outcomes.append(
                CaseOutcome(
                    case_id=rendered,
                    split=split,
                    stratum=case.stratum,
                    status=status,
                    score=score,
                    duration_s=0.0,
                    failure_message=failure_message,
                    artifacts_dir=str(case_dir),
                    trace_ref=trace_refs[0] if trace_refs else None,
                )
            )

        passed = sum(1 for outcome in outcomes if outcome.passed)
        result = SplitResult(
            split=split,
            variant=variant.key,
            model=experiment.model,
            passed=passed,
            total=len(outcomes),
            score=float(sum(outcome.score for outcome in outcomes)),
            returncode=max(returncodes, default=0),
            run_dir=str(split_dir),
            outcomes=tuple(outcomes),
        )
        result.save(result_path)
        summary_payload = {
            "passed": result.passed,
            "total": result.total,
            "correctness": result.correctness,
            "score": result.score,
        }
        (split_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")
        return result

    def _build_command(
        self,
        *,
        experiment: Experiment,
        task_name: str,
        jobs_dir: Path,
        job_name: str,
    ) -> list[str]:
        config = experiment.runner_config
        command = [str(item) for item in config["command"]]
        command.extend(
            [
                "run",
                "-p",
                str(config["tasks_root"]),
                "--task-name",
                task_name,
                "-l",
                "1",
                "-n",
                str(config.get("concurrency", 1)),
            ]
        )
        if agent_import_path := config.get("agent_import_path"):
            command.extend(["--agent-import-path", str(agent_import_path)])
        command.extend(["-o", str(jobs_dir), "--job-name", job_name])
        command.extend(str(item) for item in config.get("extra_args", []))
        return command


def build_runner(experiment: Experiment):
    """Build the configured runner."""
    if experiment.runner == "pytest":
        return PytestRunner()
    if experiment.runner == "harbor":
        return HarborRunner()
    msg = f"unknown runner {experiment.runner!r}"
    raise ValueError(msg)


def parse_pytest_outcomes(
    *,
    junit_path: Path,
    cases: list[EvalCase],
    model: str,
    artifacts_dir: Path,
) -> list[CaseOutcome]:
    """Parse JUnit results for configured cases."""
    root = ET.fromstring(junit_path.read_text())
    configured = {case.render(model=model): case for case in cases}
    outcomes: dict[str, CaseOutcome] = {}
    for testcase in root.iter("testcase"):
        case_id = rebuild_case_id(
            file_attr=testcase.attrib.get("file", ""),
            classname_attr=testcase.attrib.get("classname", ""),
            name_attr=testcase.attrib.get("name", ""),
        )
        case = configured.get(case_id)
        if case is None:
            continue
        status = "passed"
        failure_message = None
        failure = testcase.find("failure")
        if failure is None:
            failure = testcase.find("error")
        if failure is not None:
            status = "failed"
            failure_message = failure.text or failure.attrib.get("message")
        elif testcase.find("skipped") is not None:
            status = "skipped"
        outcomes[case_id] = CaseOutcome(
            case_id=case_id,
            split=case.split,
            stratum=case.stratum,
            status=status,
            score=1.0 if status == "passed" else 0.0,
            duration_s=float(testcase.attrib.get("time", "0") or "0"),
            failure_message=failure_message,
            artifacts_dir=str(artifacts_dir),
        )
    return [
        outcomes.get(
            case.render(model=model),
            CaseOutcome(
                case_id=case.render(model=model),
                split=case.split,
                stratum=case.stratum,
                status="missing",
                score=0.0,
                duration_s=0.0,
                failure_message="case missing from junit.xml",
                artifacts_dir=str(artifacts_dir),
            ),
        )
        for case in cases
    ]


def rebuild_case_id(*, file_attr: str, classname_attr: str, name_attr: str) -> str:
    """Best-effort reconstruction of a pytest nodeid from JUnit fields."""
    if file_attr:
        return f"{file_attr}::{name_attr}"
    if classname_attr.startswith("tests."):
        return f"{classname_attr.replace('.', '/')}.py::{name_attr}"
    return name_attr


def parse_harbor_case(*, jobs_dir: Path, pass_threshold: float) -> tuple[float, dict[str, object] | None, str | None]:
    """Parse one Harbor task result."""
    payload = None
    score = 0.0
    failure_message: str | None = None
    json_paths = sorted(jobs_dir.rglob("result.json"))
    if json_paths:
        payload = json.loads(json_paths[0].read_text())
        score = float(payload.get("score", payload.get("reward", 0.0)))
        failure_message = None if score >= pass_threshold else str(payload.get("message", "score below threshold"))
        return score, payload, failure_message

    reward_paths = sorted(jobs_dir.rglob("reward.txt"))
    if reward_paths:
        raw_score = reward_paths[0].read_text().strip()
        score = float(raw_score or "0")
        payload = {"score": score}
        failure_message = None if score >= pass_threshold else "score below threshold"
        return score, payload, failure_message

    return 0.0, None, "missing Harbor result files"


def safe_slug(value: str) -> str:
    """Return a filesystem-safe slug."""
    cleaned = [
        character if character.isalnum() else "-"
        for character in value
    ]
    slug = "".join(cleaned).strip("-")
    return slug or "case"
