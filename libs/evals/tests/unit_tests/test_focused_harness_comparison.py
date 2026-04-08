from __future__ import annotations

import json
from typing import TYPE_CHECKING

from deepagents_evals.better_harness.focused_comparison import (
    ModelComparison,
    PromptVariantResult,
    SliceResult,
    run_focused_comparison,
    run_variant,
    slugify_model_name,
    to_markdown,
)

if TYPE_CHECKING:
    from pathlib import Path


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


def test_slugify_model_name() -> None:
    assert slugify_model_name("baseten:zai-org/GLM-5") == "baseten-zai-org-glm-5"


def test_run_variant_reuses_existing_artifacts(tmp_path: Path) -> None:
    model_name = "claude-sonnet-4-6"
    model_dir = tmp_path / slugify_model_name(model_name)
    _write_report(model_dir / "legacy-tool_use.json", passed=7, total=8)
    _write_report(model_dir / "legacy-conversation.json", passed=2, total=6)

    result = run_variant(
        evals_root=tmp_path,
        model_name=model_name,
        prompt_variant="legacy",
        output_dir=tmp_path,
        reuse_existing=True,
    )

    assert result.prompt_variant == "legacy"
    assert result.tool_use.summary == "7/8"
    assert result.conversation.summary == "2/6"
    assert result.combined_passed == 9
    assert result.combined_total == 14


def test_run_focused_comparison_writes_report_with_reused_artifacts(tmp_path: Path) -> None:
    sonnet_dir = tmp_path / "claude-sonnet-4-6"
    glm_dir = tmp_path / "baseten-zai-org-glm-5"

    _write_report(sonnet_dir / "legacy-tool_use.json", passed=7, total=8)
    _write_report(sonnet_dir / "legacy-conversation.json", passed=2, total=6)
    _write_report(sonnet_dir / "improved-tool_use.json", passed=8, total=8)
    _write_report(sonnet_dir / "improved-conversation.json", passed=6, total=6)

    _write_report(glm_dir / "legacy-tool_use.json", passed=6, total=8)
    _write_report(glm_dir / "legacy-conversation.json", passed=1, total=6)
    _write_report(glm_dir / "improved-tool_use.json", passed=8, total=8)
    _write_report(glm_dir / "improved-conversation.json", passed=6, total=6)

    payload = run_focused_comparison(
        models=["claude-sonnet-4-6", "baseten:zai-org/GLM-5"],
        output_dir=tmp_path,
        reuse_existing=True,
        evals_root=tmp_path,
    )

    assert len(payload["comparisons"]) == 2
    assert (tmp_path / "comparison.json").exists()
    assert (tmp_path / "comparison.md").exists()
    markdown = (tmp_path / "comparison.md").read_text()
    assert "`claude-sonnet-4-6` | Baseline | `7/8` | `2/6` | `9/14`" in markdown
    assert "`baseten:zai-org/GLM-5` | Improved | `8/8` | `6/6` | `14/14`" in markdown


def test_to_markdown_renders_both_variants() -> None:
    comparison = ModelComparison(
        model_name="claude-sonnet-4-6",
        baseline=PromptVariantResult(
            prompt_variant="legacy",
            tool_use=SliceResult(7, 8, 0.875, "/tmp/legacy-tool.json"),
            conversation=SliceResult(2, 6, 0.3333, "/tmp/legacy-conv.json"),
        ),
        improved=PromptVariantResult(
            prompt_variant="improved",
            tool_use=SliceResult(8, 8, 1.0, "/tmp/improved-tool.json"),
            conversation=SliceResult(6, 6, 1.0, "/tmp/improved-conv.json"),
        ),
    )

    rendered = to_markdown("2026-04-07T00:00:00+00:00", [comparison])

    assert "# Focused Harness Comparison" in rendered
    assert "| `claude-sonnet-4-6` | Baseline | `7/8` | `2/6` | `9/14` |" in rendered
    assert "| `claude-sonnet-4-6` | Improved | `8/8` | `6/6` | `14/14` |" in rendered
    assert "- Improved conversation: `/tmp/improved-conv.json`" in rendered
