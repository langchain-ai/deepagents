"""Hill-climbing optimizer for Deep Agents prompt modules."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from deepagents_evals.better_harness.benchmarks import (
    BetterHarnessCase,
    HarnessVariant,
    SuiteResult,
    build_default_benchmark,
    evaluate_variant,
)
from deepagents_evals.better_harness.prompt_modules import (
    DEFAULT_PROMPT_MODULE_ORDER,
    PROMPT_MODULES,
    PromptModule,
)
from deepagents_evals.better_harness.variants import EXAMPLE_BASE_AGENT_PROMPT

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class SuiteEvaluator(Protocol):
    """Protocol for evaluating one harness variant against a suite."""

    def __call__(
        self,
        variant: HarnessVariant,
        *,
        cases: list[BetterHarnessCase],
        model_name: str,
        module_prompts: dict[str, str],
    ) -> SuiteResult:
        """Run the variant against the provided cases."""


@dataclass
class CandidateEvaluation:
    """One candidate module scored during an optimization iteration."""

    module_name: str
    variant: HarnessVariant
    suite: SuiteResult
    accepted: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the candidate evaluation."""
        payload = asdict(self)
        payload["suite"] = self.suite.to_dict()
        payload["variant"] = self.variant.key
        return payload


@dataclass
class IterationResult:
    """The results of a single hill-climbing iteration."""

    iteration: int
    baseline_suite: SuiteResult
    candidates: list[CandidateEvaluation]
    accepted_module: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the iteration result."""
        return {
            "iteration": self.iteration,
            "baseline_suite": self.baseline_suite.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "accepted_module": self.accepted_module,
        }


@dataclass
class OptimizationResult:
    """Full optimization output including baseline and holdout evaluation."""

    created_at: str
    model_name: str
    optimization_suite_baseline: SuiteResult
    holdout_suite_baseline: SuiteResult
    optimization_suite_final: SuiteResult
    holdout_suite_final: SuiteResult
    iterations: list[IterationResult]
    selected_modules: tuple[str, ...]
    final_prompt: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the optimization result."""
        return {
            "created_at": self.created_at,
            "model_name": self.model_name,
            "optimization_suite_baseline": self.optimization_suite_baseline.to_dict(),
            "holdout_suite_baseline": self.holdout_suite_baseline.to_dict(),
            "optimization_suite_final": self.optimization_suite_final.to_dict(),
            "holdout_suite_final": self.holdout_suite_final.to_dict(),
            "iterations": [iteration.to_dict() for iteration in self.iterations],
            "selected_modules": list(self.selected_modules),
            "final_prompt": self.final_prompt,
        }

    def to_markdown(self) -> str:
        """Render a concise human-readable report."""
        lines = [
            "# Better Harness Report",
            "",
            f"- Created: {self.created_at}",
            f"- Model: `{self.model_name}`",
            f"- Selected modules: `{', '.join(self.selected_modules) or 'baseline'}`",
            "",
            "## Score Summary",
            "",
            "| Split | Baseline | Final |",
            "| --- | --- | --- |",
            (
                f"| Optimization | {self.optimization_suite_baseline.passed_count}/"
                f"{self.optimization_suite_baseline.total} | "
                f"{self.optimization_suite_final.passed_count}/"
                f"{self.optimization_suite_final.total} |"
            ),
            (
                f"| Holdout | {self.holdout_suite_baseline.passed_count}/"
                f"{self.holdout_suite_baseline.total} | "
                f"{self.holdout_suite_final.passed_count}/"
                f"{self.holdout_suite_final.total} |"
            ),
            "",
            "## Iterations",
            "",
        ]

        for iteration in self.iterations:
            accepted = iteration.accepted_module or "none"
            lines.append(
                f"- Iteration {iteration.iteration}: accepted `{accepted}` "
                f"from baseline {iteration.baseline_suite.passed_count}/{iteration.baseline_suite.total}"
            )
            lines.extend(
                [
                    f"  - `{candidate.module_name}` -> {candidate.suite.passed_count}/"
                    f"{candidate.suite.total} ({candidate.reason})"
                    for candidate in iteration.candidates
                ]
            )

        lines.extend(
            [
                "",
                "## Final Holdout Failures",
                "",
            ]
        )
        holdout_failures = self.holdout_suite_final.failures_by_case()
        if not holdout_failures:
            lines.append("- None")
        else:
            for case_id, failure in holdout_failures.items():
                lines.append(f"- `{case_id}`: {failure}")

        return "\n".join(lines) + "\n"

    def write(self, *, json_path: Path, markdown_path: Path | None = None) -> None:
        """Write JSON and optional Markdown artifacts to disk."""
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        if markdown_path is not None:
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(self.to_markdown())


def _module_prompts(modules: dict[str, PromptModule]) -> dict[str, str]:
    return {name: module.prompt for name, module in modules.items()}


def _is_non_regressing(previous: SuiteResult, candidate: SuiteResult) -> bool:
    """Return True if the candidate preserved every previous pass."""
    return previous.passed_case_ids().issubset(candidate.passed_case_ids())


def _candidate_sort_key(candidate: CandidateEvaluation) -> tuple[int, int, float]:
    """Return a stable score key for ranking candidates."""
    return (
        1 if candidate.accepted else 0,
        candidate.suite.passed_count,
        -candidate.suite.mean_duration_s,
    )


def hill_climb_prompt_modules(
    *,
    model_name: str,
    benchmark_cases: list[BetterHarnessCase] | None = None,
    prompt_modules: dict[str, PromptModule] | None = None,
    module_order: tuple[str, ...] | None = None,
    max_iterations: int = 4,
    evaluator: SuiteEvaluator = evaluate_variant,
    base_prompt: str = EXAMPLE_BASE_AGENT_PROMPT,
) -> OptimizationResult:
    """Run a simple non-regressing hill climb over prompt modules."""
    cases = benchmark_cases or build_default_benchmark()
    modules = prompt_modules or PROMPT_MODULES
    ordered_module_names = module_order or DEFAULT_PROMPT_MODULE_ORDER

    optimization_cases = [case for case in cases if case.split == "optimization"]
    holdout_cases = [case for case in cases if case.split == "holdout"]
    if not optimization_cases:
        msg = "benchmark must include at least one optimization case"
        raise ValueError(msg)
    if not holdout_cases:
        msg = "benchmark must include at least one holdout case"
        raise ValueError(msg)

    prompt_map = _module_prompts(modules)
    current_variant = HarnessVariant()

    def score_variant(
        variant: HarnessVariant,
        *,
        cases: list[BetterHarnessCase],
    ) -> SuiteResult:
        if evaluator is evaluate_variant:
            return evaluate_variant(
                variant,
                cases=cases,
                model_name=model_name,
                module_prompts=prompt_map,
                base_prompt=base_prompt,
            )
        return evaluator(
            variant,
            cases=cases,
            model_name=model_name,
            module_prompts=prompt_map,
        )

    optimization_suite_baseline = score_variant(
        current_variant,
        cases=optimization_cases,
    )
    holdout_suite_baseline = score_variant(
        current_variant,
        cases=holdout_cases,
    )

    current_suite = optimization_suite_baseline
    iterations: list[IterationResult] = []
    remaining_modules = [name for name in ordered_module_names if name in modules]

    for iteration_index in range(1, max_iterations + 1):
        logger.info(
            "Starting iteration=%d current_variant=%s optimization_score=%d/%d",
            iteration_index,
            current_variant.key,
            current_suite.passed_count,
            current_suite.total,
        )
        iteration_baseline = current_suite
        candidate_evaluations: list[CandidateEvaluation] = []
        best_candidate: CandidateEvaluation | None = None

        for module_name in remaining_modules:
            variant = current_variant.add_module(module_name)
            suite = score_variant(
                variant,
                cases=optimization_cases,
            )
            logger.info(
                "Scored candidate module=%s variant=%s optimization_score=%d/%d",
                module_name,
                variant.key,
                suite.passed_count,
                suite.total,
            )

            if not _is_non_regressing(current_suite, suite):
                candidate_evaluations.append(
                    CandidateEvaluation(
                        module_name=module_name,
                        variant=variant,
                        suite=suite,
                        accepted=False,
                        reason="regressed existing optimization pass",
                    )
                )
                continue

            if suite.passed_count <= current_suite.passed_count:
                candidate_evaluations.append(
                    CandidateEvaluation(
                        module_name=module_name,
                        variant=variant,
                        suite=suite,
                        accepted=False,
                        reason="did not improve optimization score",
                    )
                )
                continue

            candidate = CandidateEvaluation(
                module_name=module_name,
                variant=variant,
                suite=suite,
                accepted=False,
                reason="improved optimization score",
            )
            candidate_evaluations.append(candidate)

            if best_candidate is None or suite.passed_count > best_candidate.suite.passed_count:
                best_candidate = CandidateEvaluation(
                    module_name=module_name,
                    variant=variant,
                    suite=suite,
                    accepted=True,
                    reason="best non-regressing improvement",
                )

        if best_candidate is not None:
            candidate_evaluations = [
                best_candidate if candidate.module_name == best_candidate.module_name else candidate
                for candidate in candidate_evaluations
            ]
            candidate_evaluations.sort(key=_candidate_sort_key, reverse=True)
            current_variant = best_candidate.variant
            current_suite = best_candidate.suite
            remaining_modules.remove(best_candidate.module_name)
            accepted_module = best_candidate.module_name
            logger.info(
                "Accepted module=%s new_variant=%s optimization_score=%d/%d",
                best_candidate.module_name,
                current_variant.key,
                current_suite.passed_count,
                current_suite.total,
            )
        else:
            candidate_evaluations.sort(key=_candidate_sort_key, reverse=True)
            accepted_module = None
            logger.info("No improving candidate found in iteration=%d", iteration_index)

        iterations.append(
            IterationResult(
                iteration=iteration_index,
                baseline_suite=iteration_baseline,
                candidates=candidate_evaluations,
                accepted_module=accepted_module,
            )
        )

        if accepted_module is None:
            break

    optimization_suite_final = current_suite
    holdout_suite_final = score_variant(
        current_variant,
        cases=holdout_cases,
    )

    final_prompt = current_variant.render_prompt(prompt_map)
    return OptimizationResult(
        created_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        model_name=model_name,
        optimization_suite_baseline=optimization_suite_baseline,
        holdout_suite_baseline=holdout_suite_baseline,
        optimization_suite_final=optimization_suite_final,
        holdout_suite_final=holdout_suite_final,
        iterations=iterations,
        selected_modules=current_variant.module_names,
        final_prompt=final_prompt,
    )
