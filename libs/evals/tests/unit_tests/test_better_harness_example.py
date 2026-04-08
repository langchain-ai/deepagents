from __future__ import annotations

import json
from typing import TYPE_CHECKING

import deepagents.graph as graph_module

from deepagents_evals.better_harness.benchmarks import (
    BetterHarnessCase,
    CaseResult,
    HarnessVariant,
    SuiteResult,
)
from deepagents_evals.better_harness.example_workflow import (
    build_smoke_selection,
    run_example_workflow,
    write_split_manifest,
)
from deepagents_evals.better_harness.optimizer import OptimizationResult
from deepagents_evals.better_harness.prompt_modules import PromptModule
from deepagents_evals.better_harness.variants import (
    EXAMPLE_BASE_AGENT_PROMPT,
    PromptHarnessVariant,
    build_prompt_variant,
    patched_base_agent_prompt,
)

if TYPE_CHECKING:
    from pathlib import Path


def _case(case_id: str, split: str) -> BetterHarnessCase:
    from deepagents_evals.better_harness.assertions import TrajectoryScorer

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
                duration_s=0.1,
                step_count=1,
                tool_call_count=0,
                final_text="ok",
                trajectory="step 1",
            )
            for case in cases
        ],
    )


def _write_report(path: Path, *, passed: int, total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "passed": passed,
                "total": total,
                "correctness": passed / total,
            }
        )
    )


def test_patched_base_agent_prompt_restores_original_value() -> None:
    original = graph_module.BASE_AGENT_PROMPT

    with patched_base_agent_prompt("temporary prompt"):
        assert graph_module.BASE_AGENT_PROMPT == "temporary prompt"

    assert original == graph_module.BASE_AGENT_PROMPT


def test_build_prompt_variant_appends_selected_modules() -> None:
    variant = build_prompt_variant(
        label="optimized",
        base_prompt="base prompt",
        variant=HarnessVariant(module_names=("a", "b")),
        module_prompts={"a": "first block", "b": "second block"},
    )

    assert variant.selected_modules == ("a", "b")
    assert variant.base_agent_prompt == "base prompt\n\nfirst block\n\nsecond block"


def test_write_split_manifest_materializes_groups(tmp_path: Path) -> None:
    write_split_manifest(
        output_dir=tmp_path,
        benchmark_cases=[_case("opt_1", "optimization"), _case("hold_1", "holdout")],
    )

    payload = json.loads((tmp_path / "split.json").read_text())
    assert payload["optimization"] == ["opt_1"]
    assert payload["holdout"] == ["hold_1"]
    assert (tmp_path / "split.md").exists()


def test_build_smoke_selection_targets_small_representative_slice() -> None:
    benchmark_cases, acceptance_slices = build_smoke_selection("claude-sonnet-4-6")

    assert {case.case_id for case in benchmark_cases} == {
        "tool_indirect_email_report",
        "followup_vague_send_report",
        "tool_direct_slack_dm",
    }
    assert acceptance_slices[0].test_path == "tests/evals/test_tool_selection.py::test_direct_request_slack_dm"
    assert acceptance_slices[1].test_path.endswith(
        "[claude-sonnet-4-6-vague_send_report]"
    )


def test_run_example_workflow_writes_reusable_artifacts(tmp_path: Path, monkeypatch) -> None:
    optimization_cases = [_case("opt_1", "optimization"), _case("opt_2", "optimization")]
    holdout_cases = [_case("hold_1", "holdout")]
    benchmark_cases = [*optimization_cases, *holdout_cases]

    optimization_result = OptimizationResult(
        created_at="2026-04-07T00:00:00+00:00",
        model_name="claude-sonnet-4-6",
        optimization_suite_baseline=_suite(HarnessVariant(), optimization_cases, {"opt_1"}),
        holdout_suite_baseline=_suite(HarnessVariant(), holdout_cases, set()),
        optimization_suite_final=_suite(
            HarnessVariant(module_names=("use_defaults",)),
            optimization_cases,
            {"opt_1", "opt_2"},
        ),
        holdout_suite_final=_suite(
            HarnessVariant(module_names=("use_defaults",)),
            holdout_cases,
            {"hold_1"},
        ),
        iterations=[],
        selected_modules=("use_defaults",),
        final_prompt="## Defaults\n- Use reasonable defaults.",
    )

    monkeypatch.setattr(
        "deepagents_evals.better_harness.example_workflow.hill_climb_prompt_modules",
        lambda **_: optimization_result,
    )

    acceptance_dir = tmp_path / "acceptance" / "claude-sonnet-4-6"
    _write_report(acceptance_dir / "baseline-tool_use.json", passed=7, total=8)
    _write_report(acceptance_dir / "baseline-conversation.json", passed=2, total=6)
    _write_report(acceptance_dir / "optimized-tool_use.json", passed=8, total=8)
    _write_report(acceptance_dir / "optimized-conversation.json", passed=6, total=6)

    result = run_example_workflow(
        model_name="claude-sonnet-4-6",
        output_dir=tmp_path,
        max_iterations=2,
        reuse_existing=True,
        benchmark_cases=benchmark_cases,
        prompt_modules={
            "use_defaults": PromptModule(
                name="use_defaults",
                description="Use defaults",
                prompt="## Defaults\n- Use reasonable defaults.",
            )
        },
        module_order=("use_defaults",),
        base_prompt=EXAMPLE_BASE_AGENT_PROMPT,
        evals_root=tmp_path,
    )

    assert result.optimization_result.selected_modules == ("use_defaults",)
    assert result.acceptance_baseline is not None
    assert result.acceptance_baseline.combined_passed == 9
    assert result.acceptance_optimized is not None
    assert result.acceptance_optimized.combined_passed == 14

    assert (tmp_path / "optimization" / "report.json").exists()
    assert (tmp_path / "variants" / "baseline.json").exists()
    assert (tmp_path / "variants" / "optimized.json").exists()
    assert (tmp_path / "acceptance" / "comparison.md").exists()
    assert (tmp_path / "workflow.md").exists()

    workflow_payload = json.loads((tmp_path / "workflow.json").read_text())
    assert workflow_payload["model_name"] == "claude-sonnet-4-6"

    baseline_variant = PromptHarnessVariant.load(tmp_path / "variants" / "baseline.json")
    optimized_variant = PromptHarnessVariant.load(tmp_path / "variants" / "optimized.json")
    assert baseline_variant.selected_modules == ()
    assert optimized_variant.selected_modules == ("use_defaults",)
