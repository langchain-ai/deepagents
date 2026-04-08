"""Focused baseline-vs-improved harness comparison helpers."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalSlice:
    """One focused eval slice used in the harness comparison."""

    key: str
    label: str
    test_path: str


@dataclass(frozen=True)
class SliceResult:
    """One eval-slice result loaded from a pytest JSON report."""

    passed: int
    total: int
    correctness: float
    artifact_path: str

    @property
    def summary(self) -> str:
        """Return a compact pass-count summary."""
        return f"{self.passed}/{self.total}"


@dataclass(frozen=True)
class PromptVariantResult:
    """Aggregated focused-slice results for one prompt variant."""

    prompt_variant: str
    tool_use: SliceResult
    conversation: SliceResult

    @property
    def combined_passed(self) -> int:
        """Return combined passed-case count across focused slices."""
        return self.tool_use.passed + self.conversation.passed

    @property
    def combined_total(self) -> int:
        """Return combined case count across focused slices."""
        return self.tool_use.total + self.conversation.total

    @property
    def combined_correctness(self) -> float:
        """Return combined correctness across focused slices."""
        return self.combined_passed / self.combined_total


@dataclass(frozen=True)
class ModelComparison:
    """Baseline-vs-improved focused comparison for a model."""

    model_name: str
    baseline: PromptVariantResult
    improved: PromptVariantResult

    def to_dict(self) -> dict[str, Any]:
        """Serialize the model comparison."""
        return {
            "model_name": self.model_name,
            "baseline": variant_to_dict(self.baseline),
            "improved": variant_to_dict(self.improved),
        }


FOCUSED_SLICES = (
    EvalSlice(
        key="tool_use",
        label="Tool Use",
        test_path="tests/evals/test_tool_selection.py",
    ),
    EvalSlice(
        key="conversation",
        label="Conversation",
        test_path="tests/evals/test_followup_quality.py",
    ),
)


def slugify_model_name(model_name: str) -> str:
    """Return a filesystem-safe model slug."""
    return re.sub(r"[^a-z0-9]+", "-", model_name.lower()).strip("-")


def load_slice_result(artifact_path: Path) -> SliceResult:
    """Load one pytest JSON report artifact."""
    payload = json.loads(artifact_path.read_text())
    return SliceResult(
        passed=int(payload["passed"]),
        total=int(payload["total"]),
        correctness=float(payload["correctness"]),
        artifact_path=str(artifact_path),
    )


def variant_to_dict(variant: PromptVariantResult) -> dict[str, Any]:
    """Serialize one prompt-variant comparison result."""
    return {
        "prompt_variant": variant.prompt_variant,
        "tool_use": asdict(variant.tool_use),
        "conversation": asdict(variant.conversation),
        "combined": {
            "passed": variant.combined_passed,
            "total": variant.combined_total,
            "correctness": round(variant.combined_correctness, 4),
        },
    }


def run_slice(
    *,
    evals_root: Path,
    model_name: str,
    prompt_variant: str,
    slice_config: EvalSlice,
    artifact_path: Path,
) -> SliceResult:
    """Run one focused pytest slice and return the parsed report."""
    env = os.environ.copy()
    if prompt_variant == "legacy":
        env["DEEPAGENTS_BASE_PROMPT_VARIANT"] = "legacy"
    else:
        env.pop("DEEPAGENTS_BASE_PROMPT_VARIANT", None)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        slice_config.test_path,
        "--model",
        model_name,
        "--evals-report-file",
        str(artifact_path),
    ]
    subprocess.run(cmd, cwd=evals_root, env=env, check=True)  # noqa: S603
    return load_slice_result(artifact_path)


def run_variant(
    *,
    evals_root: Path,
    model_name: str,
    prompt_variant: str,
    output_dir: Path,
    reuse_existing: bool,
) -> PromptVariantResult:
    """Run or reuse the focused slices for one prompt variant."""
    model_dir = output_dir / slugify_model_name(model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, SliceResult] = {}

    for slice_config in FOCUSED_SLICES:
        artifact_path = model_dir / f"{prompt_variant}-{slice_config.key}.json"
        if reuse_existing and artifact_path.exists():
            results[slice_config.key] = load_slice_result(artifact_path)
            continue

        results[slice_config.key] = run_slice(
            evals_root=evals_root,
            model_name=model_name,
            prompt_variant=prompt_variant,
            slice_config=slice_config,
            artifact_path=artifact_path,
        )

    return PromptVariantResult(
        prompt_variant=prompt_variant,
        tool_use=results["tool_use"],
        conversation=results["conversation"],
    )


def to_markdown(created_at: str, comparisons: list[ModelComparison]) -> str:
    """Render the comparison report as Markdown."""
    lines = [
        "# Focused Harness Comparison",
        "",
        f"- Created: {created_at}",
        "",
        "| Model | Prompt | Tool Use | Conversation | Combined |",
        "| --- | --- | --- | --- | --- |",
    ]

    for comparison in comparisons:
        for variant in (comparison.baseline, comparison.improved):
            label = "Baseline" if variant.prompt_variant == "legacy" else "Improved"
            lines.append(
                f"| `{comparison.model_name}` | {label} | "
                f"`{variant.tool_use.summary}` | "
                f"`{variant.conversation.summary}` | "
                f"`{variant.combined_passed}/{variant.combined_total}` |"
            )

    lines.extend(["", "## Artifacts", ""])
    for comparison in comparisons:
        lines.append(f"### `{comparison.model_name}`")
        lines.append("")
        lines.append(
            f"- Baseline tool-use: `{comparison.baseline.tool_use.artifact_path}`"
        )
        lines.append(
            f"- Baseline conversation: `{comparison.baseline.conversation.artifact_path}`"
        )
        lines.append(
            f"- Improved tool-use: `{comparison.improved.tool_use.artifact_path}`"
        )
        lines.append(
            f"- Improved conversation: `{comparison.improved.conversation.artifact_path}`"
        )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_focused_comparison(
    *,
    models: list[str] | None = None,
    output_dir: Path = Path("artifacts/focused-comparison"),
    reuse_existing: bool = False,
    evals_root: Path | None = None,
) -> dict[str, Any]:
    """Run the focused legacy-vs-improved comparison and write artifacts."""
    evals_root = evals_root or Path(__file__).resolve().parents[2]
    models = models or ["claude-sonnet-4-6", "baseten:zai-org/GLM-5"]
    created_at = datetime.now(tz=UTC).isoformat(timespec="seconds")
    comparisons: list[ModelComparison] = []

    for model_name in models:
        baseline = run_variant(
            evals_root=evals_root,
            model_name=model_name,
            prompt_variant="legacy",
            output_dir=output_dir,
            reuse_existing=reuse_existing,
        )
        improved = run_variant(
            evals_root=evals_root,
            model_name=model_name,
            prompt_variant="improved",
            output_dir=output_dir,
            reuse_existing=reuse_existing,
        )
        comparisons.append(
            ModelComparison(
                model_name=model_name,
                baseline=baseline,
                improved=improved,
            )
        )

    payload = {
        "created_at": created_at,
        "comparisons": [comparison.to_dict() for comparison in comparisons],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparison.json"
    markdown_path = output_dir / "comparison.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(to_markdown(created_at, comparisons))
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the focused comparison runner."""
    parser = argparse.ArgumentParser(
        description="Run focused baseline-vs-improved harness comparisons",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model name to compare. Repeat to run multiple models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/focused-comparison"),
        help="Directory for per-slice reports and aggregate comparison output.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing per-slice JSON reports in --output-dir instead of rerunning them.",
    )
    return parser


def main() -> None:
    """Parse arguments and run focused comparisons."""
    args = build_parser().parse_args()
    run_focused_comparison(
        models=args.models,
        output_dir=args.output_dir,
        reuse_existing=args.reuse_existing,
    )
