"""Shareable end-to-end better-harness example workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from deepagents_evals.better_harness.benchmarks import (
    BetterHarnessCase,
    build_default_benchmark,
)
from deepagents_evals.better_harness.focused_comparison import (
    FOCUSED_SLICES,
    EvalSlice,
    PromptVariantResult,
    SliceResult,
    load_slice_result,
    slugify_model_name,
)
from deepagents_evals.better_harness.optimizer import (
    OptimizationResult,
    hill_climb_prompt_modules,
)
from deepagents_evals.better_harness.prompt_modules import (
    DEFAULT_PROMPT_MODULE_ORDER,
    PROMPT_MODULES,
    PromptModule,
)
from deepagents_evals.better_harness.variants import (
    BETTER_HARNESS_VARIANT_FILE_ENV,
    EXAMPLE_BASE_AGENT_PROMPT,
    PromptHarnessVariant,
    build_prompt_variant,
)


@dataclass(frozen=True)
class ExampleWorkflowResult:
    """Materialized result of an end-to-end better-harness example run."""

    created_at: str
    model_name: str
    baseline_variant: PromptHarnessVariant
    optimized_variant: PromptHarnessVariant
    optimization_result: OptimizationResult
    acceptance_baseline: PromptVariantResult | None
    acceptance_optimized: PromptVariantResult | None
    output_dir: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the workflow result."""
        return {
            "created_at": self.created_at,
            "model_name": self.model_name,
            "baseline_variant": self.baseline_variant.to_dict(),
            "optimized_variant": self.optimized_variant.to_dict(),
            "optimization_result": self.optimization_result.to_dict(),
            "acceptance_baseline": None if self.acceptance_baseline is None else _variant_to_dict(self.acceptance_baseline),
            "acceptance_optimized": None if self.acceptance_optimized is None else _variant_to_dict(self.acceptance_optimized),
            "output_dir": self.output_dir,
        }

    def to_markdown(self, *, benchmark_cases: list[BetterHarnessCase], acceptance_slices: tuple[EvalSlice, ...]) -> str:
        """Render a concise workflow report."""
        lines = [
            "# Better Harness Example",
            "",
            f"- Created: {self.created_at}",
            f"- Model: `{self.model_name}`",
            f"- Output dir: `{self.output_dir}`",
            "",
            "## Split",
            "",
            "| Group | Cases |",
            "| --- | --- |",
        ]
        grouped = _group_case_ids(benchmark_cases)
        lines.append(f"| Optimization | `{', '.join(grouped['optimization'])}` |")
        lines.append(f"| Holdout | `{', '.join(grouped['holdout'])}` |")
        lines.append(
            f"| Acceptance | `{', '.join(slice_config.test_path for slice_config in acceptance_slices)}` |"
        )
        lines.extend(
            [
                "",
                "## Hill Climb",
                "",
                "| Split | Baseline | Final |",
                "| --- | --- | --- |",
                (
                    f"| Optimization | `{self.optimization_result.optimization_suite_baseline.passed_count}/"
                    f"{self.optimization_result.optimization_suite_baseline.total}` | "
                    f"`{self.optimization_result.optimization_suite_final.passed_count}/"
                    f"{self.optimization_result.optimization_suite_final.total}` |"
                ),
                (
                    f"| Holdout | `{self.optimization_result.holdout_suite_baseline.passed_count}/"
                    f"{self.optimization_result.holdout_suite_baseline.total}` | "
                    f"`{self.optimization_result.holdout_suite_final.passed_count}/"
                    f"{self.optimization_result.holdout_suite_final.total}` |"
                ),
                "",
                f"- Selected modules: `{', '.join(self.optimization_result.selected_modules) or 'baseline'}`",
            ]
        )

        if self.acceptance_baseline is not None and self.acceptance_optimized is not None:
            lines.extend(
                [
                    "",
                    "## Acceptance",
                    "",
                    "| Variant | Tool Use | Conversation | Combined |",
                    "| --- | --- | --- | --- |",
                    (
                        f"| Baseline | `{self.acceptance_baseline.tool_use.summary}` | "
                        f"`{self.acceptance_baseline.conversation.summary}` | "
                        f"`{self.acceptance_baseline.combined_passed}/{self.acceptance_baseline.combined_total}` |"
                    ),
                    (
                        f"| Optimized | `{self.acceptance_optimized.tool_use.summary}` | "
                        f"`{self.acceptance_optimized.conversation.summary}` | "
                        f"`{self.acceptance_optimized.combined_passed}/{self.acceptance_optimized.combined_total}` |"
                    ),
                ]
            )

        return "\n".join(lines) + "\n"

    def write(
        self,
        *,
        output_dir: Path,
        benchmark_cases: list[BetterHarnessCase],
        acceptance_slices: tuple[EvalSlice, ...],
    ) -> None:
        """Write JSON and Markdown workflow artifacts."""
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "workflow.json").write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        (output_dir / "workflow.md").write_text(
            self.to_markdown(
                benchmark_cases=benchmark_cases,
                acceptance_slices=acceptance_slices,
            )
        )


def write_split_manifest(
    *,
    output_dir: Path,
    benchmark_cases: list[BetterHarnessCase],
    acceptance_slices: tuple[EvalSlice, ...] = FOCUSED_SLICES,
) -> None:
    """Write the representative-sample and acceptance split definitions."""
    grouped = _group_case_ids(benchmark_cases)
    payload = {
        "optimization": grouped["optimization"],
        "holdout": grouped["holdout"],
        "acceptance": [asdict(slice_config) for slice_config in acceptance_slices],
    }
    (output_dir / "split.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Better Harness Split",
        "",
        "## Optimization",
        "",
        *[f"- `{case_id}`" for case_id in grouped["optimization"]],
        "",
        "## Holdout",
        "",
        *[f"- `{case_id}`" for case_id in grouped["holdout"]],
        "",
        "## Acceptance",
        "",
        *[f"- `{slice_config.test_path}`" for slice_config in acceptance_slices],
        "",
    ]
    (output_dir / "split.md").write_text("\n".join(lines))


def run_acceptance_variant(
    *,
    evals_root: Path,
    model_name: str,
    variant: PromptHarnessVariant,
    output_dir: Path,
    acceptance_slices: tuple[EvalSlice, ...] = FOCUSED_SLICES,
    reuse_existing: bool = False,
) -> PromptVariantResult:
    """Run or reuse the focused acceptance eval slices for one variant."""
    output_dir.mkdir(parents=True, exist_ok=True)
    variant_path = output_dir / f"{variant.label}-variant.json"
    variant.save(variant_path)

    results: dict[str, SliceResult] = {}
    for slice_config in acceptance_slices:
        artifact_path = output_dir / f"{variant.label}-{slice_config.key}.json"
        if reuse_existing and artifact_path.exists():
            results[slice_config.key] = load_slice_result(artifact_path)
            continue
        _run_acceptance_slice(
            evals_root=evals_root,
            model_name=model_name,
            slice_config=slice_config,
            artifact_path=artifact_path,
            variant_path=variant_path,
        )
        results[slice_config.key] = load_slice_result(artifact_path)

    return PromptVariantResult(
        prompt_variant=variant.label,
        tool_use=results["tool_use"],
        conversation=results["conversation"],
    )


def run_example_workflow(
    *,
    model_name: str,
    output_dir: Path = Path("artifacts/better-harness-example"),
    max_iterations: int = 4,
    reuse_existing: bool = False,
    benchmark_cases: list[BetterHarnessCase] | None = None,
    prompt_modules: dict[str, PromptModule] | None = None,
    module_order: tuple[str, ...] | None = None,
    base_prompt: str = EXAMPLE_BASE_AGENT_PROMPT,
    acceptance_slices: tuple[EvalSlice, ...] = FOCUSED_SLICES,
    evals_root: Path | None = None,
    skip_acceptance: bool = False,
) -> ExampleWorkflowResult:
    """Run the full split -> hill-climb -> holdout -> acceptance workflow."""
    benchmark_cases = benchmark_cases or build_default_benchmark()
    prompt_modules = prompt_modules or PROMPT_MODULES
    module_order = module_order or DEFAULT_PROMPT_MODULE_ORDER
    evals_root = evals_root or Path(__file__).resolve().parents[2]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_split_manifest(
        output_dir=output_dir,
        benchmark_cases=benchmark_cases,
        acceptance_slices=acceptance_slices,
    )

    optimization_result = hill_climb_prompt_modules(
        model_name=model_name,
        benchmark_cases=benchmark_cases,
        prompt_modules=prompt_modules,
        module_order=module_order,
        max_iterations=max_iterations,
        base_prompt=base_prompt,
    )
    optimization_dir = output_dir / "optimization"
    optimization_result.write(
        json_path=optimization_dir / "report.json",
        markdown_path=optimization_dir / "report.md",
    )

    baseline_variant = build_prompt_variant(
        label="baseline",
        base_prompt=base_prompt,
        variant=optimization_result.optimization_suite_baseline.variant,
        module_prompts={name: module.prompt for name, module in prompt_modules.items()},
    )
    optimized_variant = build_prompt_variant(
        label="optimized",
        base_prompt=base_prompt,
        variant=optimization_result.optimization_suite_final.variant,
        module_prompts={name: module.prompt for name, module in prompt_modules.items()},
    )

    variants_dir = output_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)
    baseline_variant.save(variants_dir / "baseline.json")
    optimized_variant.save(variants_dir / "optimized.json")

    acceptance_dir = output_dir / "acceptance" / slugify_model_name(model_name)
    acceptance_baseline: PromptVariantResult | None = None
    acceptance_optimized: PromptVariantResult | None = None
    if not skip_acceptance:
        acceptance_baseline = run_acceptance_variant(
            evals_root=evals_root,
            model_name=model_name,
            variant=baseline_variant,
            output_dir=acceptance_dir,
            acceptance_slices=acceptance_slices,
            reuse_existing=reuse_existing,
        )
        acceptance_optimized = run_acceptance_variant(
            evals_root=evals_root,
            model_name=model_name,
            variant=optimized_variant,
            output_dir=acceptance_dir,
            acceptance_slices=acceptance_slices,
            reuse_existing=reuse_existing,
        )
        _write_acceptance_summary(
            output_dir=output_dir / "acceptance",
            created_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
            model_name=model_name,
            baseline=acceptance_baseline,
            optimized=acceptance_optimized,
        )

    result = ExampleWorkflowResult(
        created_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        model_name=model_name,
        baseline_variant=baseline_variant,
        optimized_variant=optimized_variant,
        optimization_result=optimization_result,
        acceptance_baseline=acceptance_baseline,
        acceptance_optimized=acceptance_optimized,
        output_dir=str(output_dir),
    )
    result.write(
        output_dir=output_dir,
        benchmark_cases=benchmark_cases,
        acceptance_slices=acceptance_slices,
    )
    return result


def build_smoke_selection(model_name: str) -> tuple[list[BetterHarnessCase], tuple[EvalSlice, ...]]:
    """Return a cheap end-to-end smoke selection for one model."""
    selected_ids = {
        "tool_indirect_email_report",
        "followup_vague_send_report",
        "tool_direct_slack_dm",
    }
    benchmark_cases = [
        case for case in build_default_benchmark() if case.case_id in selected_ids
    ]
    acceptance_slices = (
        EvalSlice(
            key="tool_use",
            label="Tool Use",
            test_path="tests/evals/test_tool_selection.py::test_direct_request_slack_dm",
        ),
        EvalSlice(
            key="conversation",
            label="Conversation",
            test_path=(
                "tests/evals/test_followup_quality.py::"
                f"test_followup_question_quality[{model_name}-vague_send_report]"
            ),
        ),
    )
    return benchmark_cases, acceptance_slices


def build_example_parser() -> argparse.ArgumentParser:
    """Return the CLI parser for the full better-harness example workflow."""
    parser = argparse.ArgumentParser(
        description="Run the full better-harness split/holdout/eval example",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Model name to optimize and validate.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=2,
        help="Maximum hill-climbing iterations.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/better-harness-example"),
        help="Directory to write the example artifacts.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing acceptance JSON reports in --output-dir when present.",
    )
    parser.add_argument(
        "--skip-acceptance",
        action="store_true",
        help="Run only the representative-sample optimization and holdout benchmark.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a small built-in smoke selection instead of the full representative sample and acceptance slice.",
    )
    return parser


def main() -> str:
    """Parse CLI arguments and run the example workflow."""
    args = build_example_parser().parse_args()
    benchmark_cases = None
    acceptance_slices = FOCUSED_SLICES
    if args.smoke:
        benchmark_cases, acceptance_slices = build_smoke_selection(args.model)
    result = run_example_workflow(
        model_name=args.model,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        reuse_existing=args.reuse_existing,
        skip_acceptance=args.skip_acceptance,
        benchmark_cases=benchmark_cases,
        acceptance_slices=acceptance_slices,
    )
    return result.to_markdown(
        benchmark_cases=benchmark_cases or build_default_benchmark(),
        acceptance_slices=acceptance_slices,
    )


def _group_case_ids(benchmark_cases: list[BetterHarnessCase]) -> dict[str, list[str]]:
    return {
        "optimization": [case.case_id for case in benchmark_cases if case.split == "optimization"],
        "holdout": [case.case_id for case in benchmark_cases if case.split == "holdout"],
    }


def _run_acceptance_slice(
    *,
    evals_root: Path,
    model_name: str,
    slice_config: EvalSlice,
    artifact_path: Path,
    variant_path: Path,
) -> None:
    env = os.environ.copy()
    env[BETTER_HARNESS_VARIANT_FILE_ENV] = str(variant_path)
    env.setdefault("LANGSMITH_TEST_SUITE", "better-harness-example")
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        slice_config.test_path,
        "--model",
        model_name,
        "-p",
        "deepagents_evals.better_harness.pytest_plugin",
        "--evals-report-file",
        str(artifact_path),
    ]
    subprocess.run(cmd, cwd=evals_root, env=env, check=True)  # noqa: S603


def _variant_to_dict(variant: PromptVariantResult) -> dict[str, Any]:
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


def _write_acceptance_summary(
    *,
    output_dir: Path,
    created_at: str,
    model_name: str,
    baseline: PromptVariantResult,
    optimized: PromptVariantResult,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": created_at,
        "model_name": model_name,
        "baseline": _variant_to_dict(baseline),
        "optimized": _variant_to_dict(optimized),
    }
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Acceptance Comparison",
        "",
        f"- Created: {created_at}",
        f"- Model: `{model_name}`",
        "",
        "| Variant | Tool Use | Conversation | Combined |",
        "| --- | --- | --- | --- |",
        (
            f"| Baseline | `{baseline.tool_use.summary}` | `{baseline.conversation.summary}` | "
            f"`{baseline.combined_passed}/{baseline.combined_total}` |"
        ),
        (
            f"| Optimized | `{optimized.tool_use.summary}` | `{optimized.conversation.summary}` | "
            f"`{optimized.combined_passed}/{optimized.combined_total}` |"
        ),
        "",
    ]
    (output_dir / "comparison.md").write_text("\n".join(lines))
