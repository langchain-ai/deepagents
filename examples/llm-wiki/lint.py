"""Lint-specific workflow for the LLM wiki."""

from __future__ import annotations

from pathlib import Path

from models import CliDeps, RunnerConfig
import helpers


def build_lint_prompt(topic: str, note: str | None) -> str:
    """Build the single-pass lint prompt for wiki health checks."""
    note_text = note or "(none)"
    return (
        f"Run a single-pass lint reconciliation for the '{topic}' wiki under `/wiki/`.\n\n"
        "Execution mode:\n"
        "- Read recent `/log.md` entries first (latest ~10 `## [` headings) to account for recent work.\n"
        "- Apply updates immediately in this run (no review/confirm phase).\n"
        "- Update wiki pages in place; do not create a separate lint report directory.\n"
        "- You may create new canonical wiki pages when required for reconciliation.\n"
        "- Do not edit `/log.md`; the runner appends structured lint timeline entries.\n"
        "- Never write to `/raw/`.\n\n"
        "Required health checks and fixes:\n"
        "- Reconcile contradictions across wiki pages and preserve explicit uncertainty when unresolved.\n"
        "- Identify stale claims superseded by newer evidence and update or qualify those claims.\n"
        "- Detect orphan pages with no inbound links and add/repair cross-references or merge them.\n"
        "- Add missing cross-references between related pages and concepts.\n"
        "- When an important concept lacks a dedicated page, create a canonical page and link it.\n"
        "- Identify data gaps and missing evidence that block confidence.\n"
        "- Suggest high-value follow-up questions and source leads for unresolved gaps.\n\n"
        "External verification policy:\n"
        "- Use model-native web browsing/search only if available in this model/runtime.\n"
        "- If web access is unavailable, do not fabricate findings; mark gaps as unresolved and list what to verify next.\n\n"
        "After edits, return a concise markdown report with exactly these sections:\n"
        "## Reconciled Changes\n"
        "## Remaining Gaps\n"
        "## Suggested Next Questions and Sources\n\n"
        f"Operator note: {note_text}\n"
    )


def run_lint_workspace(config: RunnerConfig, workspace_dir: Path, deps: CliDeps) -> str:
    """Run lint mode as a single-pass apply and return the lint summary."""
    prompt = build_lint_prompt(config.topic, config.note)
    lint_summary = deps.run_agent_mode(workspace_dir, config.topic, prompt, config.model)

    helpers._refresh_index(config.topic, workspace_dir)
    lint_metadata: dict[str, object] = {}
    if config.note:
        lint_metadata["note"] = config.note
    helpers._append_log_entry(
        workspace_dir,
        "lint.apply",
        "applied",
        metadata=lint_metadata,
        summary=lint_summary or "Lint applied without model summary.",
    )

    summary = lint_summary.strip()
    if summary:
        return summary
    return "## Reconciled Changes\n- Lint applied.\n\n## Remaining Gaps\n- None reported.\n\n## Suggested Next Questions and Sources\n- None reported."
