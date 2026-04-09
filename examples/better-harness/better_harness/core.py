"""Core data model, config loading, run loop, history, and CLI."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VALID_SPLITS = ("train", "holdout", "scorecard")
SPLIT_ALIASES = {
    "acceptance": "scorecard",
    "final_eval": "scorecard",
}
VISIBLE_SPLITS = {"train"}
PRIVATE_SPLITS = {"holdout", "scorecard"}
VALID_SURFACE_KINDS = ("module_attr", "workspace_file")
VALID_RUNNERS = ("pytest", "harbor")
ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")
URL_PATTERN = re.compile(r"https?://[^\s\"'>]+")
UUID_PATTERN = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
)


@dataclass(frozen=True)
class Surface:
    """One editable harness surface."""

    name: str
    kind: str
    target: str
    base_value: str
    filename: str


@dataclass(frozen=True)
class EvalCase:
    """One explicit eval assignment."""

    case_id: str
    split: str
    stratum: str

    def render(self, *, model: str) -> str:
        """Render one case id for a concrete model."""
        return self.case_id.format(model=model)


@dataclass(frozen=True)
class Experiment:
    """Loaded experiment config."""

    path: Path
    name: str
    runner: str
    workspace_root: Path
    model: str
    max_iterations: int
    better_agent_model: str
    better_agent_max_turns: int
    better_agent_deepagents_root: Path | None
    better_agent_system_prompt: str | None
    runner_config: dict[str, Any]
    surfaces: dict[str, Surface]
    cases: tuple[EvalCase, ...]

    def cases_for_split(self, split: str) -> list[EvalCase]:
        """Return cases for one split."""
        return [case for case in self.cases if case.split == split]

    def rendered_case_ids(self, split: str) -> list[str]:
        """Return rendered case ids for one split."""
        return [case.render(model=self.model) for case in self.cases_for_split(split)]

    def strata_for_split(self, split: str) -> set[str]:
        """Return the stratum set for one split."""
        return {case.stratum for case in self.cases_for_split(split)}

    def has_split(self, split: str) -> bool:
        """Return whether the experiment defines one split."""
        return bool(self.cases_for_split(split))


@dataclass(frozen=True)
class Variant:
    """Materialized set of surface values."""

    label: str
    model: str
    changed_surfaces: tuple[str, ...]
    surfaces: dict[str, Surface]
    values: dict[str, str]

    @property
    def key(self) -> str:
        """Return a stable filesystem key."""
        return self.label

    def attr_overrides(self) -> dict[str, str]:
        """Return module-attr overrides keyed by target."""
        return {
            surface.target: self.values[name]
            for name, surface in self.surfaces.items()
            if surface.kind == "module_attr"
        }

    def file_overrides(self) -> dict[str, str]:
        """Return workspace-file overrides keyed by relative file path."""
        return {
            surface.target: self.values[name]
            for name, surface in self.surfaces.items()
            if surface.kind == "workspace_file"
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the variant."""
        return {
            "label": self.label,
            "model": self.model,
            "changed_surfaces": list(self.changed_surfaces),
            "surfaces": {
                name: asdict(surface)
                for name, surface in self.surfaces.items()
            },
            "values": self.values,
        }

    def save(self, path: Path) -> None:
        """Persist the variant."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> Variant:
        """Load one variant from disk."""
        payload = json.loads(path.read_text())
        surfaces = {
            name: Surface(**surface_payload)
            for name, surface_payload in payload["surfaces"].items()
        }
        return cls(
            label=str(payload["label"]),
            model=str(payload["model"]),
            changed_surfaces=tuple(str(item) for item in payload["changed_surfaces"]),
            surfaces=surfaces,
            values={name: str(value) for name, value in payload["values"].items()},
        )


@dataclass(frozen=True)
class CaseOutcome:
    """One case-level outcome."""

    case_id: str
    split: str
    stratum: str
    status: str
    score: float
    duration_s: float
    failure_message: str | None = None
    artifacts_dir: str | None = None
    trace_ref: str | None = None

    @property
    def passed(self) -> bool:
        """Return whether the outcome counts as passed."""
        return self.status == "passed"


@dataclass(frozen=True)
class SplitResult:
    """One split result."""

    split: str
    variant: str
    model: str
    passed: int
    total: int
    score: float
    returncode: int
    run_dir: str
    outcomes: tuple[CaseOutcome, ...]

    @property
    def correctness(self) -> float:
        """Return pass rate for the split."""
        return 0.0 if self.total == 0 else self.passed / self.total

    def passing_case_ids(self) -> set[str]:
        """Return the set of passed case ids."""
        return {
            outcome.case_id
            for outcome in self.outcomes
            if outcome.passed
        }

    def failing_outcomes(self) -> list[CaseOutcome]:
        """Return failed or missing outcomes."""
        return [
            outcome
            for outcome in self.outcomes
            if outcome.status != "passed"
        ]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result."""
        return {
            "split": self.split,
            "variant": self.variant,
            "model": self.model,
            "passed": self.passed,
            "total": self.total,
            "score": self.score,
            "correctness": self.correctness,
            "returncode": self.returncode,
            "run_dir": self.run_dir,
            "outcomes": [asdict(outcome) for outcome in self.outcomes],
        }

    def save(self, path: Path) -> None:
        """Persist result JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: Path) -> SplitResult:
        """Load one split result."""
        payload = json.loads(path.read_text())
        return cls(
            split=str(payload["split"]),
            variant=str(payload["variant"]),
            model=str(payload["model"]),
            passed=int(payload["passed"]),
            total=int(payload["total"]),
            score=float(payload["score"]),
            returncode=int(payload["returncode"]),
            run_dir=str(payload["run_dir"]),
            outcomes=tuple(CaseOutcome(**item) for item in payload["outcomes"]),
        )


@dataclass(frozen=True)
class Proposal:
    """One outer-loop Deep Agent proposal."""

    changed_surfaces: tuple[str, ...]
    workspace_dir: str
    summary: str
    final_message: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the proposal."""
        return asdict(self)


@dataclass(frozen=True)
class CandidateEvaluation:
    """One candidate evaluation."""

    variant: str
    proposal: Proposal
    train: SplitResult
    holdout: SplitResult
    accepted: bool
    reason: str

    def combined_passed(self) -> int:
        """Return the combined pass count."""
        return self.train.passed + self.holdout.passed


@dataclass(frozen=True)
class IterationRecord:
    """One optimization iteration."""

    iteration: int
    starting_variant: str
    candidate: CandidateEvaluation | None


@dataclass(frozen=True)
class RunReport:
    """Final run report."""

    created_at: str
    config_path: str
    model: str
    better_agent_model: str
    baseline: Variant
    final: Variant
    baseline_train: SplitResult
    baseline_holdout: SplitResult
    final_train: SplitResult
    final_holdout: SplitResult
    baseline_scorecard: SplitResult | None
    final_scorecard: SplitResult | None
    iterations: tuple[IterationRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report."""
        return {
            "created_at": self.created_at,
            "config_path": self.config_path,
            "model": self.model,
            "better_agent_model": self.better_agent_model,
            "baseline": self.baseline.to_dict(),
            "final": self.final.to_dict(),
            "baseline_train": self.baseline_train.to_dict(),
            "baseline_holdout": self.baseline_holdout.to_dict(),
            "final_train": self.final_train.to_dict(),
            "final_holdout": self.final_holdout.to_dict(),
            "baseline_scorecard": None if self.baseline_scorecard is None else self.baseline_scorecard.to_dict(),
            "final_scorecard": None if self.final_scorecard is None else self.final_scorecard.to_dict(),
            "iterations": [
                {
                    "iteration": iteration.iteration,
                    "starting_variant": iteration.starting_variant,
                    "candidate": None
                    if iteration.candidate is None
                    else {
                        "variant": iteration.candidate.variant,
                        "proposal": iteration.candidate.proposal.to_dict(),
                        "accepted": iteration.candidate.accepted,
                        "reason": iteration.candidate.reason,
                        "train": iteration.candidate.train.to_dict(),
                        "holdout": iteration.candidate.holdout.to_dict(),
                    },
                }
                for iteration in self.iterations
            ],
        }

    def to_markdown(self) -> str:
        """Render a concise Markdown report."""
        lines = [
            "# better-harness report",
            "",
            f"- Target model: `{self.model}`",
            f"- Better-agent model: `{self.better_agent_model}`",
            f"- Baseline changed surfaces: `{', '.join(self.baseline.changed_surfaces) or 'none'}`",
            f"- Final changed surfaces: `{', '.join(self.final.changed_surfaces) or 'none'}`",
            "",
            "| Split | Baseline | Final |",
            "| --- | --- | --- |",
            (
                f"| Train | `{self.baseline_train.passed}/{self.baseline_train.total}` | "
                f"`{self.final_train.passed}/{self.final_train.total}` |"
            ),
            (
                f"| Holdout | `{self.baseline_holdout.passed}/{self.baseline_holdout.total}` | "
                f"`{self.final_holdout.passed}/{self.final_holdout.total}` |"
            ),
        ]
        if self.baseline_scorecard is not None and self.final_scorecard is not None:
            lines.append(
                f"| Scorecard | `{self.baseline_scorecard.passed}/{self.baseline_scorecard.total}` | "
                f"`{self.final_scorecard.passed}/{self.final_scorecard.total}` |"
            )
        lines.extend(["", "## Iterations", ""])
        for iteration in self.iterations:
            if iteration.candidate is None:
                lines.append(f"- Iteration {iteration.iteration}: no candidate produced")
                continue
            candidate = iteration.candidate
            decision = "accepted" if candidate.accepted else "rejected"
            lines.extend(
                [
                    f"- Iteration {iteration.iteration}: {decision} `{candidate.variant}`",
                    f"  - Changed surfaces: `{', '.join(candidate.proposal.changed_surfaces) or 'none'}`",
                    f"  - Train: `{candidate.train.passed}/{candidate.train.total}`",
                    f"  - Holdout: `{candidate.holdout.passed}/{candidate.holdout.total}`",
                    f"  - Reason: {candidate.reason}",
                ]
            )
        lines.append("")
        return "\n".join(lines)

    def write(self, output_dir: Path) -> None:
        """Write JSON and Markdown reports."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "report.json").write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        (output_dir / "report.md").write_text(self.to_markdown())


class RunLayout:
    """Filesystem layout for one experiment run."""

    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def variants_dir(self) -> Path:
        return self.root / "variants"

    @property
    def visible_root(self) -> Path:
        return self.root / "history" / "visible"

    @property
    def private_root(self) -> Path:
        return self.root / "history" / "private"

    @property
    def visible_iterations_dir(self) -> Path:
        return self.visible_root / "iterations"

    def split_dir(self, *, variant_key: str, split: str) -> Path:
        base = self.visible_root if split in VISIBLE_SPLITS else self.private_root
        return base / split / variant_key

    def variant_path(self, variant_key: str) -> Path:
        return self.variants_dir / f"{variant_key}.json"

    @property
    def runtime_dir(self) -> Path:
        return self.root / "_runtime"

    def proposer_workspace_dir(self, iteration: int) -> Path:
        return self.visible_iterations_dir / f"{iteration:03d}" / "proposer_workspace"

    def iteration_dir(self, iteration: int) -> Path:
        return self.visible_iterations_dir / f"{iteration:03d}"

    def write_manifest(self, experiment: Experiment) -> None:
        """Write experiment metadata and split manifests."""
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": experiment.name,
            "runner": experiment.runner,
            "workspace_root": str(experiment.workspace_root),
            "model": experiment.model,
            "max_iterations": experiment.max_iterations,
            "better_agent_model": experiment.better_agent_model,
            "better_agent_max_turns": experiment.better_agent_max_turns,
            "better_agent_deepagents_root": None
            if experiment.better_agent_deepagents_root is None
            else str(experiment.better_agent_deepagents_root),
        }
        (self.root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
        write_split_manifest(experiment, self.root)

    def write_iteration_decision(
        self,
        *,
        iteration: int,
        starting_variant: str,
        proposal: Proposal,
        candidate: CandidateEvaluation,
    ) -> None:
        """Persist one iteration summary."""
        iteration_dir = self.iteration_dir(iteration)
        iteration_dir.mkdir(parents=True, exist_ok=True)
        decision = "accepted" if candidate.accepted else "rejected"
        payload = {
            "iteration": iteration,
            "starting_variant": starting_variant,
            "candidate_variant": candidate.variant,
            "decision": decision,
            "reason": candidate.reason,
            "changed_surfaces": list(proposal.changed_surfaces),
            "train_passed": candidate.train.passed,
            "train_total": candidate.train.total,
            "summary": proposal.summary,
            "final_message": proposal.final_message,
        }
        (iteration_dir / "decision.json").write_text(json.dumps(payload, indent=2) + "\n")
        lines = [
            f"# Iteration {iteration}",
            "",
            f"- Starting variant: `{starting_variant}`",
            f"- Candidate variant: `{candidate.variant}`",
            f"- Decision: `{decision}`",
            f"- Train: `{candidate.train.passed}/{candidate.train.total}`",
            f"- Changed surfaces: `{', '.join(proposal.changed_surfaces) or 'none'}`",
            f"- Reason: {candidate.reason}",
            "",
            "## Proposal Summary",
            "",
            proposal.summary or "_No proposal summary written._",
            "",
        ]
        (iteration_dir / "decision.md").write_text("\n".join(lines))

    def write_report(self, report: RunReport) -> None:
        """Write the final report."""
        report.write(self.root)


def expand_env(value: str) -> str:
    """Expand ${ENV_VAR} references."""
    return ENV_PATTERN.sub(lambda match: os.environ[match.group(1)], value)


def normalize_split(value: str) -> str:
    """Normalize one split name and apply aliases."""
    return SPLIT_ALIASES.get(value, value)


def _resolve_path(config_path: Path, raw: str) -> Path:
    path = Path(expand_env(raw)).expanduser()
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def _resolve_command_tokens(config_path: Path, tokens: list[str]) -> list[str]:
    resolved: list[str] = []
    for token in tokens:
        expanded = expand_env(token)
        candidate = config_path.parent / expanded
        if ("/" in expanded or expanded.endswith(".py")) and candidate.exists():
            resolved.append(str(candidate.resolve()))
        else:
            resolved.append(expanded)
    return resolved


def _surface_filename(
    *,
    name: str,
    target: str,
    base_suffix: str | None,
    kind: str,
    payload: dict,
) -> str:
    if "filename" in payload:
        return str(payload["filename"])
    if kind == "workspace_file":
        return Path(target).name
    suffix = base_suffix or ".txt"
    return f"{name}{suffix}"


def load_experiment(path: str | Path, *, model_override: str | None = None) -> Experiment:
    """Load one experiment config."""
    config_path = Path(path).resolve()
    raw = tomllib.loads(config_path.read_text())
    experiment = raw.get("experiment", {})

    runner = str(experiment.get("runner", "pytest"))
    runner_config = dict(raw.get("runner", {}).get(runner, {}))

    if runner == "pytest" and "evals_project" in experiment and "project_root" not in runner_config:
        runner_config["project_root"] = str(experiment["evals_project"])

    if runner == "pytest":
        runner_config.setdefault("project_root", "libs/evals")
        runner_config.setdefault("pytest_args", ["-q"])
    elif runner == "harbor":
        runner_config.setdefault("tasks_root", "tasks")
        runner_config.setdefault("command", ["harbor"])
        runner_config.setdefault("extra_args", [])
        runner_config.setdefault("pass_threshold", 1.0)

    if "command" in runner_config:
        runner_config["command"] = _resolve_command_tokens(
            config_path,
            [str(item) for item in runner_config["command"]],
        )

    for key in ("project_root", "tasks_root"):
        if key in runner_config:
            runner_config[key] = str(_resolve_path(config_path, str(runner_config[key])))

    name = str(experiment["name"])
    workspace_root = _resolve_path(config_path, str(experiment["workspace_root"]))
    model = model_override or str(experiment.get("model", "default-model"))
    max_iterations = int(experiment.get("max_iterations", 3))

    better_agent = raw.get("better_agent", {})
    better_agent_model = str(better_agent.get("model", model))
    better_agent_max_turns = int(better_agent.get("max_turns", 11000))
    better_agent_deepagents_root = None
    if raw_root := better_agent.get("deepagents_root"):
        better_agent_deepagents_root = _resolve_path(config_path, str(raw_root))
    elif env_root := os.environ.get("DEEPAGENTS_ROOT"):
        better_agent_deepagents_root = Path(env_root).expanduser().resolve()

    better_agent_system_prompt = None
    if raw_prompt := better_agent.get("system_prompt_file"):
        better_agent_system_prompt = _resolve_path(config_path, str(raw_prompt)).read_text().strip()

    surfaces: dict[str, Surface] = {}
    for surface_name, payload in raw.get("surfaces", {}).items():
        kind = str(payload["kind"])
        target = str(payload["target"])
        has_base_file = "base_file" in payload
        has_base_value = "base_value" in payload
        if has_base_file == has_base_value:
            msg = (
                f"surface '{surface_name}' must define exactly one of "
                "'base_file' or 'base_value'"
            )
            raise ValueError(msg)
        if has_base_file:
            base_file = _resolve_path(config_path, str(payload["base_file"]))
            base_value = base_file.read_text().strip()
            base_suffix = base_file.suffix or ".txt"
        else:
            base_value = str(payload["base_value"]).strip()
            base_suffix = None
        surfaces[surface_name] = Surface(
            name=surface_name,
            kind=kind,
            target=target,
            base_value=base_value,
            filename=_surface_filename(
                name=surface_name,
                target=target,
                base_suffix=base_suffix,
                kind=kind,
                payload=payload,
            ),
        )

    cases = tuple(
        EvalCase(
            case_id=str(item.get("case_id", item.get("nodeid"))),
            split=normalize_split(str(item["split"])),
            stratum=str(item["stratum"]),
        )
        for item in raw.get("cases", [])
    )

    loaded = Experiment(
        path=config_path,
        name=name,
        runner=runner,
        workspace_root=workspace_root,
        model=model,
        max_iterations=max_iterations,
        better_agent_model=better_agent_model,
        better_agent_max_turns=better_agent_max_turns,
        better_agent_deepagents_root=better_agent_deepagents_root,
        better_agent_system_prompt=better_agent_system_prompt,
        runner_config=runner_config,
        surfaces=surfaces,
        cases=cases,
    )
    validate_experiment(loaded)
    return loaded


def validate_experiment(experiment: Experiment) -> None:
    """Validate one experiment config."""
    if experiment.runner not in VALID_RUNNERS:
        msg = f"invalid runner {experiment.runner!r}"
        raise ValueError(msg)
    if not experiment.surfaces:
        msg = "config must define at least one surface"
        raise ValueError(msg)
    if experiment.max_iterations < 1:
        msg = "max_iterations must be at least 1"
        raise ValueError(msg)
    if experiment.better_agent_max_turns < 1:
        msg = "better_agent.max_turns must be at least 1"
        raise ValueError(msg)

    for surface in experiment.surfaces.values():
        if surface.kind not in VALID_SURFACE_KINDS:
            msg = f"invalid surface kind {surface.kind!r}"
            raise ValueError(msg)

    splits = {case.split for case in experiment.cases}
    unknown_splits = splits - set(VALID_SPLITS)
    if unknown_splits:
        msg = f"unknown split names: {sorted(unknown_splits)}"
        raise ValueError(msg)

    for split in ("train", "holdout"):
        if not experiment.cases_for_split(split):
            msg = f"split {split!r} must include at least one case"
            raise ValueError(msg)

    rendered = [case.render(model=experiment.model) for case in experiment.cases]
    if len(rendered) != len(set(rendered)):
        msg = "rendered case ids must be unique across all splits"
        raise ValueError(msg)

    if experiment.strata_for_split("train") != experiment.strata_for_split("holdout"):
        msg = (
            "train and holdout must cover the same strata; "
            f"got train={sorted(experiment.strata_for_split('train'))} "
            f"holdout={sorted(experiment.strata_for_split('holdout'))}"
        )
        raise ValueError(msg)

    if experiment.runner == "pytest" and not experiment.runner_config.get("project_root"):
        msg = "pytest runner requires runner.pytest.project_root"
        raise ValueError(msg)
    if experiment.runner == "harbor":
        if not experiment.runner_config.get("tasks_root"):
            msg = "harbor runner requires runner.harbor.tasks_root"
            raise ValueError(msg)
        if not experiment.runner_config.get("command"):
            msg = "harbor runner requires runner.harbor.command"
            raise ValueError(msg)


def write_split_manifest(experiment: Experiment, output_dir: Path) -> None:
    """Write split manifests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        split: [
            {
                "case_id": case.render(model=experiment.model),
                "stratum": case.stratum,
            }
            for case in experiment.cases_for_split(split)
        ]
        for split in VALID_SPLITS
        if experiment.cases_for_split(split)
    }
    (output_dir / "split.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = ["# Split Manifest", ""]
    for split, items in payload.items():
        lines.extend([f"## {split.title()}", ""])
        lines.extend(f"- `{item['stratum']}`: `{item['case_id']}`" for item in items)
        lines.append("")
    (output_dir / "split.md").write_text("\n".join(lines))


def extract_trace_refs(*, payload: Any | None, stdout: str, stderr: str) -> list[str]:
    """Extract URL references from structured payloads and raw logs."""
    urls: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
            return
        if isinstance(value, list):
            for child in value:
                walk(child)
            return
        if isinstance(value, str):
            urls.update(URL_PATTERN.findall(value))

    if payload is not None:
        walk(payload)
    urls.update(URL_PATTERN.findall(stdout))
    urls.update(URL_PATTERN.findall(stderr))
    return sorted(urls)


def write_trace_refs(split_dir: Path, refs: list[str]) -> None:
    """Persist trace references if any were found."""
    if not refs:
        return
    payload = {
        "provider": "langsmith" if any("smith.langchain" in ref for ref in refs) else "generic",
        "urls": refs,
    }
    (split_dir / "trace_refs.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Trace References", ""]
    lines.extend(f"- {ref}" for ref in refs)
    lines.append("")
    (split_dir / "trace_refs.md").write_text("\n".join(lines))
    write_trace_payloads(split_dir, refs)


def _resolve_session_run_ids(
    *,
    endpoint: str,
    api_key: str,
    session_id: str,
) -> list[str]:
    """Query LangSmith for root run IDs belonging to an experiment session."""
    url = f"{endpoint}/runs/query"
    if not url.startswith(("https://", "http://")):
        return []
    body = json.dumps(
        {"session": [session_id], "is_root": True, "limit": 50, "select": ["id"]},
    ).encode()
    request = urllib.request.Request(  # noqa: S310
        url,
        data=body,
        headers={
            "X-API-Key": api_key,
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            data = json.loads(response.read().decode("utf-8"))
        return [str(run["id"]) for run in data.get("runs", [])]
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError):
        return []


def write_trace_payloads(split_dir: Path, refs: list[str]) -> None:
    """Fetch and persist local copies of LangSmith traces when possible."""
    api_key = os.environ.get("LANGSMITH_API_KEY")
    if not api_key:
        return

    endpoint = (
        os.environ.get("LANGSMITH_ENDPOINT")
        or os.environ.get("LANGCHAIN_ENDPOINT")
        or "https://api.smith.langchain.com"
    ).rstrip("/")
    traces_dir = split_dir / "traces" / "langsmith"
    payloads: list[dict[str, Any]] = []

    for ref in refs:
        trace_id = extract_langsmith_trace_id(ref)
        if trace_id is None:
            continue
        trace_path = traces_dir / f"{trace_id}.json"
        error_text: str | None = None
        if not trace_path.exists():
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                payload = fetch_langsmith_trace(
                    endpoint=endpoint,
                    api_key=api_key,
                    trace_id=trace_id,
                )
            except RuntimeError:
                # The ID may be an experiment session rather than a run
                # (e.g. from dataset comparison URLs). Resolve actual
                # run IDs and fetch those instead.
                for run_id in _resolve_session_run_ids(
                    endpoint=endpoint,
                    api_key=api_key,
                    session_id=trace_id,
                ):
                    run_path = traces_dir / f"{run_id}.json"
                    try:
                        run_payload = fetch_langsmith_trace(
                            endpoint=endpoint,
                            api_key=api_key,
                            trace_id=run_id,
                        )
                    except RuntimeError as exc:
                        error_text = str(exc)
                    else:
                        run_path.parent.mkdir(parents=True, exist_ok=True)
                        run_path.write_text(json.dumps(run_payload, indent=2) + "\n")
                    payloads.append(
                        {
                            "url": ref,
                            "trace_id": run_id,
                            "path": str(run_path) if run_path.exists() else None,
                            "error": error_text,
                        }
                    )
                continue
            else:
                trace_path.write_text(json.dumps(payload, indent=2) + "\n")
        payloads.append(
            {
                "url": ref,
                "trace_id": trace_id,
                "path": str(trace_path) if trace_path.exists() else None,
                "error": error_text,
            }
        )

    if payloads:
        (split_dir / "trace_payloads.json").write_text(json.dumps(payloads, indent=2) + "\n")


def extract_langsmith_trace_id(url: str) -> str | None:
    """Extract one likely LangSmith run id from a URL."""
    if "smith.langchain" not in url:
        return None
    matches = UUID_PATTERN.findall(url)
    if not matches:
        return None
    return matches[-1]


def fetch_langsmith_trace(*, endpoint: str, api_key: str, trace_id: str) -> dict[str, Any]:
    """Fetch one LangSmith run with messages included."""
    base_url = f"{endpoint}/runs/{trace_id}?include_messages=true"
    if not base_url.startswith(("https://", "http://")):
        msg = f"Unsupported LangSmith endpoint: {endpoint}"
        raise RuntimeError(msg)
    request = urllib.request.Request(  # noqa: S310
        base_url,
        headers={
            "X-API-Key": api_key,
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LangSmith fetch failed for {trace_id}: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LangSmith fetch failed for {trace_id}: {exc}") from exc


_MAX_TRACE_CONTENT_LEN = 2000


@dataclass(frozen=True)
class TraceStep:
    """One step in a summarized agent trace."""

    type: str
    content: str
    name: str | None = None
    status: str | None = None
    tool_calls: tuple[dict[str, Any], ...] | None = None


@dataclass(frozen=True)
class TraceSummary:
    """Summarized LangSmith trace for the proposer workspace."""

    status: str | None
    error: str | None
    trace_ref: str | None
    steps: tuple[TraceStep, ...]


def summarize_langsmith_trace(
    trace_data: dict[str, Any],
    *,
    trace_ref: str | None = None,
) -> TraceSummary:
    """Summarize a LangSmith trace into a compact structure.

    Supports both LangChain constructor format (`kwargs.type`) and
    OpenAI chat format (`role`/`content`) as returned by the LangSmith
    `include_messages=true` API.

    Args:
        trace_data: Raw JSON dict from `fetch_langsmith_trace`.
        trace_ref: Optional LangSmith URL for linking back.

    Returns:
        A `TraceSummary` with extracted steps. Returns a summary with
        empty steps if the trace cannot be parsed.

    """
    messages = trace_data.get("messages") or []
    steps: list[TraceStep] = []
    for msg in messages:
        if "kwargs" in msg:
            kwargs = msg["kwargs"]
            msg_type = kwargs.get("type", "unknown")
            content = kwargs.get("content")
            raw_tool_calls = kwargs.get("tool_calls")
            name = kwargs.get("name")
            status = kwargs.get("status")
        else:
            role = msg.get("role", "unknown")
            # user/assistant/tool/system all pass through; normalize
            # the two that differ from LC naming.
            msg_type = {"user": "human", "assistant": "ai"}.get(role, role)
            content = msg.get("content")
            raw_tool_calls = msg.get("tool_calls")
            name = msg.get("name")
            status = msg.get("status")

        if content is None:
            content = ""
        elif isinstance(content, list):
            text_parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            content = "\n".join(text_parts)
        elif not isinstance(content, str):
            content = str(content)

        if len(content) > _MAX_TRACE_CONTENT_LEN:
            content = content[:_MAX_TRACE_CONTENT_LEN] + "... [truncated]"

        tool_calls = None
        if raw_tool_calls:
            parsed: list[dict[str, Any]] = []
            for tc in raw_tool_calls:
                if "function" in tc:
                    fn = tc["function"]
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        with contextlib.suppress(json.JSONDecodeError, ValueError):
                            args = json.loads(args)
                    parsed.append({"name": fn.get("name"), "args": args})
                else:
                    parsed.append({"name": tc.get("name"), "args": tc.get("args")})
            tool_calls = tuple(parsed)

        steps.append(
            TraceStep(
                type=msg_type,
                content=content,
                name=name,
                status=status,
                tool_calls=tool_calls,
            )
        )

    return TraceSummary(
        status=trace_data.get("status"),
        error=trace_data.get("error"),
        trace_ref=trace_ref,
        steps=tuple(steps),
    )


def collect_trace_refs(run_dir: Path) -> list[dict[str, Any]]:
    """Collect all saved trace reference files under one run."""
    collected: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("trace_refs.json")):
        payload = json.loads(path.read_text())
        collected.append(
            {
                "path": str(path),
                "provider": payload.get("provider", "generic"),
                "urls": payload.get("urls", []),
            }
        )
    return collected


def run_experiment(
    experiment: Experiment,
    *,
    output_dir: Path,
    max_iterations: int | None = None,
    reuse_existing: bool = False,
) -> RunReport:
    """Run the better-harness optimization loop."""
    from better_harness.agent import propose_variant
    from better_harness.patching import build_baseline_variant
    from better_harness.runners import build_runner

    runner = build_runner(experiment)
    layout = RunLayout(output_dir.resolve())
    layout.write_manifest(experiment)

    iteration_limit = experiment.max_iterations if max_iterations is None else max_iterations
    baseline = build_baseline_variant(experiment)
    current = baseline
    baseline_train = runner.run_split(
        experiment=experiment,
        variant=baseline,
        split="train",
        layout=layout,
        reuse_existing=reuse_existing,
    )
    baseline_holdout = runner.run_split(
        experiment=experiment,
        variant=baseline,
        split="holdout",
        layout=layout,
        reuse_existing=reuse_existing,
    )
    current_train = baseline_train
    current_holdout = baseline_holdout

    iterations: list[IterationRecord] = []
    for index in range(1, iteration_limit + 1):
        if current_train.passed == current_train.total and current_holdout.passed == current_holdout.total:
            break
        proposal, candidate_variant = propose_variant(
            experiment=experiment,
            current=current,
            train_result=current_train,
            layout=layout,
            iteration=index,
        )
        if not proposal.changed_surfaces:
            iterations.append(
                IterationRecord(
                    iteration=index,
                    starting_variant=current.key,
                    candidate=None,
                )
            )
            break

        train = runner.run_split(
            experiment=experiment,
            variant=candidate_variant,
            split="train",
            layout=layout,
            reuse_existing=reuse_existing,
        )
        holdout = runner.run_split(
            experiment=experiment,
            variant=candidate_variant,
            split="holdout",
            layout=layout,
            reuse_existing=reuse_existing,
        )
        current_combined = current_train.passed + current_holdout.passed
        candidate_combined = train.passed + holdout.passed
        accepted = candidate_combined > current_combined
        reason = (
            "improved combined train + holdout pass count"
            if accepted
            else "did not improve combined train + holdout pass count"
        )
        candidate = CandidateEvaluation(
            variant=candidate_variant.key,
            proposal=proposal,
            train=train,
            holdout=holdout,
            accepted=accepted,
            reason=reason,
        )
        layout.write_iteration_decision(
            iteration=index,
            starting_variant=current.key,
            proposal=proposal,
            candidate=candidate,
        )
        iterations.append(
            IterationRecord(
                iteration=index,
                starting_variant=current.key,
                candidate=candidate,
            )
        )
        if accepted:
            current = candidate_variant
            current_train = train
            current_holdout = holdout

    baseline_scorecard = _run_optional_scorecard(
        experiment=experiment,
        runner=runner,
        variant=baseline,
        layout=layout,
        reuse_existing=reuse_existing,
    )
    final_scorecard = _run_optional_scorecard(
        experiment=experiment,
        runner=runner,
        variant=current,
        layout=layout,
        reuse_existing=reuse_existing,
    )

    report = RunReport(
        created_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        config_path=str(experiment.path),
        model=experiment.model,
        better_agent_model=experiment.better_agent_model,
        baseline=baseline,
        final=current,
        baseline_train=baseline_train,
        baseline_holdout=baseline_holdout,
        final_train=current_train,
        final_holdout=current_holdout,
        baseline_scorecard=baseline_scorecard,
        final_scorecard=final_scorecard,
        iterations=tuple(iterations),
    )
    layout.write_report(report)
    return report


def _run_optional_scorecard(
    *,
    experiment: Experiment,
    runner,
    variant,
    layout: RunLayout,
    reuse_existing: bool,
) -> SplitResult | None:
    if not experiment.has_split("scorecard"):
        return None
    return runner.run_split(
        experiment=experiment,
        variant=variant,
        split="scorecard",
        layout=layout,
        reuse_existing=reuse_existing,
    )


def inventory_payload(experiment: Experiment) -> dict[str, object]:
    """Build one JSON-serializable inventory payload."""
    from better_harness.runners import build_runner

    runner = build_runner(experiment)
    return {
        "name": experiment.name,
        "runner": experiment.runner,
        "workspace_root": str(experiment.workspace_root),
        "model": experiment.model,
        "cases": runner.collect_inventory(experiment),
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Improve an agent harness with a Deep Agent outer loop")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate one experiment config")
    validate_parser.add_argument("config", type=Path)
    validate_parser.add_argument("--model")

    inventory_parser = subparsers.add_parser("inventory", help="List available eval units")
    inventory_parser.add_argument("config", type=Path)
    inventory_parser.add_argument("--model")
    inventory_parser.add_argument("--output", type=Path)

    split_parser = subparsers.add_parser("split", help="Write the configured split manifest")
    split_parser.add_argument("config", type=Path)
    split_parser.add_argument("--model")
    split_parser.add_argument("--output-dir", type=Path, default=Path("split"))

    run_parser = subparsers.add_parser("run", help="Run the outer Deep Agent optimization loop")
    run_parser.add_argument("config", type=Path)
    run_parser.add_argument("--model")
    run_parser.add_argument("--max-iterations", type=int)
    run_parser.add_argument("--reuse-existing", action="store_true")
    run_parser.add_argument("--output-dir", type=Path)

    inspect_parser = subparsers.add_parser("inspect", help="Summarize one run directory")
    inspect_parser.add_argument("run_dir", type=Path)

    traces_parser = subparsers.add_parser("traces", help="List saved local and LangSmith trace refs")
    traces_parser.add_argument("run_dir", type=Path)

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect":
        report_path = args.run_dir / "report.json"
        payload = json.loads(report_path.read_text())
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "traces":
        refs = collect_trace_refs(args.run_dir)
        print(json.dumps({"count": len(refs), "items": refs}, indent=2))
        return 0

    experiment = load_experiment(args.config, model_override=getattr(args, "model", None))

    if args.command == "validate":
        print(f"Config valid: {experiment.path}")
        print(f"Runner: {experiment.runner}")
        print(f"Workspace: {experiment.workspace_root}")
        print(f"Model: {experiment.model}")
        print(f"Better-agent model: {experiment.better_agent_model}")
        print(f"Surfaces: {', '.join(experiment.surfaces)}")
        print(f"Train: {len(experiment.cases_for_split('train'))} cases")
        print(f"Holdout: {len(experiment.cases_for_split('holdout'))} cases")
        print(f"Scorecard: {len(experiment.cases_for_split('scorecard'))} cases")
        return 0

    if args.command == "inventory":
        payload = inventory_payload(experiment)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n")
            print(args.output)
        else:
            print(json.dumps(payload, indent=2))
        return 0

    if args.command == "split":
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        write_split_manifest(experiment, output_dir)
        print(output_dir / "split.json")
        return 0

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("runs") / f"{experiment.name}-{timestamp}"
    report = run_experiment(
        experiment,
        output_dir=output_dir,
        max_iterations=args.max_iterations,
        reuse_existing=args.reuse_existing,
    )
    print(report.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
