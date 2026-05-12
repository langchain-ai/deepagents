"""Lint-specific workflow for the wiki runner."""

from __future__ import annotations

import re
from pathlib import Path

from models import CliDeps, RunnerConfig
import wiki_helpers as helpers


def build_lint_prompt(topic: str, note: str | None) -> str:
    """Build the single-pass lint prompt for wiki health checks."""
    note_text = note or "(none)"
    return (
        f"Run a single-pass lint reconciliation for the '{topic}' wiki under `/wiki/`.\n\n"
        "Execution mode:\n"
        "- Apply updates immediately in this run (no review/confirm phase).\n"
        "- Update wiki pages in place; do not create a separate lint report directory.\n"
        "- You may create new canonical wiki pages when required for reconciliation.\n"
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


def _lint_log_detail(summary: str) -> str:
    """Build a concise lint log detail line from the model summary."""
    collapsed = re.sub(r"\s+", " ", summary.strip()).strip()
    if not collapsed:
        return "summary=lint applied without model summary"
    if len(collapsed) > 180:
        collapsed = f"{collapsed[:177].rstrip()}..."
    return f"summary={collapsed}"


def run_lint_workspace(config: RunnerConfig, workspace_dir: Path, deps: CliDeps) -> str:
    """Run lint mode as a single-pass apply and return the lint summary."""
    prompt = build_lint_prompt(config.topic, config.note)
    lint_summary = deps.run_agent_mode(workspace_dir, config.topic, prompt, config.model)

    helpers._refresh_index(config.topic, workspace_dir)
    helpers._append_log_entry(workspace_dir, "lint", _lint_log_detail(lint_summary))

    summary = lint_summary.strip()
    if summary:
        return summary
    return "## Reconciled Changes\n- Lint applied.\n\n## Remaining Gaps\n- None reported.\n\n## Suggested Next Questions and Sources\n- None reported."
