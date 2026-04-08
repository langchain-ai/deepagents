from __future__ import annotations

from deepagents_evals.better_harness.assertions import TrajectoryScorer
from deepagents_evals.better_harness.benchmarks import (
    BetterHarnessCase,
    CaseResult,
    HarnessVariant,
    SuiteResult,
)
from deepagents_evals.better_harness.optimizer import hill_climb_prompt_modules
from deepagents_evals.better_harness.prompt_modules import PromptModule


def _case(case_id: str, split: str) -> BetterHarnessCase:
    return BetterHarnessCase(
        case_id=case_id,
        category="test",
        split=split,  # type: ignore[arg-type]
        query=case_id,
        scorer=TrajectoryScorer(),
    )


def _suite(
    variant: HarnessVariant,
    cases: list[BetterHarnessCase],
    passed_case_ids: set[str],
) -> SuiteResult:
    return SuiteResult(
        variant=variant,
        split=cases[0].split,
        cases=[
            CaseResult(
                case_id=case.case_id,
                category=case.category,
                split=case.split,
                passed=case.case_id in passed_case_ids,
                failure=None if case.case_id in passed_case_ids else "failed",
                duration_s=1.0,
                step_count=1,
                tool_call_count=0,
                final_text="ok",
                trajectory="step 1",
            )
            for case in cases
        ],
    )


def test_harness_variant_renders_joined_prompt() -> None:
    variant = HarnessVariant(module_names=("a", "b"))

    rendered = variant.render_prompt(
        {
            "a": "first block",
            "b": "second block",
        }
    )

    assert rendered == "first block\n\nsecond block"
    assert HarnessVariant().render_prompt({"a": "unused"}) is None


def test_hill_climber_accepts_best_non_regressing_sequence() -> None:
    optimization_cases = [_case("opt_1", "optimization"), _case("opt_2", "optimization"), _case("opt_3", "optimization")]
    holdout_cases = [_case("hold_1", "holdout"), _case("hold_2", "holdout")]
    all_cases = [*optimization_cases, *holdout_cases]

    pass_map = {
        "baseline": {"opt_1", "hold_1"},
        "a": {"opt_1", "opt_2", "hold_1"},
        "b": {"opt_2", "opt_3", "hold_1"},
        "c": {"opt_1", "hold_1"},
        "a+b": {"opt_1", "opt_2", "hold_1"},
        "a+c": {"opt_1", "opt_2", "opt_3", "hold_1", "hold_2"},
    }

    def fake_evaluator(
        variant: HarnessVariant,
        *,
        cases: list[BetterHarnessCase],
        model_name: str,
        module_prompts: dict[str, str],
    ) -> SuiteResult:
        _ = model_name, module_prompts
        passed_case_ids = {
            case_id
            for case_id in pass_map.get(variant.key, set())
            if case_id in {case.case_id for case in cases}
        }
        return _suite(variant, cases, passed_case_ids)

    result = hill_climb_prompt_modules(
        model_name="test-model",
        benchmark_cases=all_cases,
        prompt_modules={
            "a": PromptModule("a", "first", "prompt a"),
            "b": PromptModule("b", "second", "prompt b"),
            "c": PromptModule("c", "third", "prompt c"),
        },
        module_order=("a", "b", "c"),
        max_iterations=4,
        evaluator=fake_evaluator,
    )

    assert result.optimization_suite_baseline.passed_count == 1
    assert result.optimization_suite_final.passed_count == 3
    assert result.holdout_suite_baseline.passed_count == 1
    assert result.holdout_suite_final.passed_count == 2
    assert result.selected_modules == ("a", "c")

    assert len(result.iterations) == 3
    assert result.iterations[0].baseline_suite.passed_count == 1
    assert result.iterations[0].accepted_module == "a"
    assert result.iterations[1].baseline_suite.passed_count == 2
    assert result.iterations[1].accepted_module == "c"
    assert result.iterations[2].accepted_module is None
