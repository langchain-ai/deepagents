"""Core data models and config loading for better-harness."""
from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Case:
    """One eval case with a split assignment."""

    id: str
    split: str  # "train" | "holdout"


@dataclass
class ToolCall:
    """One tool call within an agent turn."""

    tool: str
    input: dict[str, Any]
    output: str | None = None
    error: str | None = None


@dataclass
class Turn:
    """One agent response, including any tool calls it made."""

    agent: str  # what the agent said / its reasoning text
    calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Trace:
    """Full execution record for one eval case."""

    case_id: str
    split: str
    score: float
    task: str
    turns: list[Turn]
    final_output: str
    failure: str | None = None
    total_turns: int = 0  # number of agent turns (0 = not captured)

    def passed(self) -> bool:
        return self.score >= 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "split": self.split,
            "score": self.score,
            "task": self.task,
            "total_turns": self.total_turns,
            "turns": [
                {
                    "agent": t.agent,
                    "calls": [
                        {
                            "tool": c.tool,
                            "input": c.input,
                            "output": c.output,
                            "error": c.error,
                        }
                        for c in t.calls
                    ],
                }
                for t in self.turns
            ],
            "final_output": self.final_output,
            "failure": self.failure,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> Trace:
        data = json.loads(path.read_text())
        turns = [
            Turn(
                agent=t.get("agent", ""),
                calls=[
                    ToolCall(
                        tool=c["tool"],
                        input=c.get("input", {}),
                        output=c.get("output"),
                        error=c.get("error"),
                    )
                    for c in t.get("calls", [])
                ],
            )
            for t in data.get("turns", [])
        ]
        return cls(
            case_id=data["case_id"],
            split=data["split"],
            score=float(data.get("score", 0.0)),
            task=data.get("task", ""),
            turns=turns,
            final_output=data.get("final_output", ""),
            failure=data.get("failure"),
            total_turns=int(data.get("total_turns", len(turns))),
        )


@dataclass
class SplitResult:
    """Results for one split (train or holdout)."""

    split: str
    traces: list[Trace]

    @property
    def passed(self) -> int:
        return sum(1 for t in self.traces if t.passed())

    @property
    def total(self) -> int:
        return len(self.traces)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def failing(self) -> list[Trace]:
        return [t for t in self.traces if not t.passed()]

    def summary(self) -> str:
        return f"{self.passed}/{self.total}"


@dataclass
class IterResult:
    """Outcome of one optimization iteration."""

    iteration: int
    accepted: bool
    train: SplitResult
    holdout: SplitResult
    proposal: str = ""
    prior_train: SplitResult | None = None   # train result before this iteration (for delta)


@dataclass
class Experiment:
    """Loaded experiment config."""

    name: str
    harness_path: Path
    cases: list[Case]
    runner_config: dict[str, Any]
    max_iterations: int = 5
    min_replays: int = 1   # evals per candidate; mean used for acceptance when > 1
    better_agent_model: str = "claude-sonnet-4-6"
    better_agent_max_turns: int = 10000
    better_agent_deepagents_root: Path | None = None

    def train_cases(self) -> list[Case]:
        return [c for c in self.cases if c.split == "train"]

    def holdout_cases(self) -> list[Case]:
        return [c for c in self.cases if c.split == "holdout"]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_experiment(
    config_path: str | Path,
    *,
    model_override: str | None = None,
) -> Experiment:
    """Load and validate an experiment from a TOML config file."""
    path = Path(config_path).resolve()
    raw = tomllib.loads(path.read_text())

    exp = raw.get("experiment", {})
    name = str(exp["name"])
    harness_path = _resolve(path, str(exp["harness"]))

    runner_cfg = dict(raw.get("runner", {}))
    runner_cfg.setdefault("command", ["harbor"])
    runner_cfg.setdefault("tasks_root", "tasks")
    runner_cfg.setdefault("pass_threshold", 1.0)
    if "tasks_root" in runner_cfg:
        runner_cfg["tasks_root"] = str(_resolve(path, str(runner_cfg["tasks_root"])))
    runner_cfg["command"] = [str(t) for t in runner_cfg.get("command", ["harbor"])]

    ba = raw.get("better_agent", {})
    better_agent_model = model_override or str(ba.get("model", "claude-sonnet-4-6"))
    better_agent_max_turns = int(ba.get("max_turns", 10000))

    min_replays = max(1, int(exp.get("min_replays", 1)))
    better_agent_deepagents_root: Path | None = None
    if raw_root := ba.get("deepagents_root"):
        better_agent_deepagents_root = _resolve(path, str(raw_root))

    cases = [
        Case(id=str(c["id"]), split=str(c["split"]))
        for c in raw.get("cases", [])
    ]

    experiment = Experiment(
        name=name,
        harness_path=harness_path,
        cases=cases,
        runner_config=runner_cfg,
        max_iterations=int(exp.get("max_iterations", 5)),
        min_replays=min_replays,
        better_agent_model=better_agent_model,
        better_agent_max_turns=better_agent_max_turns,
        better_agent_deepagents_root=better_agent_deepagents_root,
    )
    _validate(experiment)
    return experiment


def _validate(experiment: Experiment) -> None:
    if not experiment.harness_path.exists():
        msg = f"harness not found: {experiment.harness_path}"
        raise ValueError(msg)
    if not experiment.train_cases():
        msg = "experiment must have at least one train case"
        raise ValueError(msg)
    if not experiment.holdout_cases():
        msg = "experiment must have at least one holdout case"
        raise ValueError(msg)
    ids = [c.id for c in experiment.cases]
    if len(ids) != len(set(ids)):
        msg = "case ids must be unique"
        raise ValueError(msg)


def _resolve(config_path: Path, raw: str) -> Path:
    p = Path(raw).expanduser()
    if not p.is_absolute():
        p = (config_path.parent / p).resolve()
    return p


def slug(value: str) -> str:
    """Return a filesystem-safe slug from a string."""
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-") or "case"
