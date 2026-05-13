"""Ingest-specific workflow for the LLM wiki."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Sequence

from models import CliDeps, RunnerConfig
import helpers

@dataclass(frozen=True)
class IngestResult:
    """Result from one ingest workspace pass."""

    answer: str | None
    should_push: bool


def _ingest_source_hint(staged_paths: Sequence[Path]) -> str:
    """Create a compact source hint for structured log metadata."""
    names = [path.name for path in staged_paths]
    if not names:
        return "none"
    if len(names) <= 2:
        return ", ".join(names)
    return f"{', '.join(names[:2])}, +{len(names) - 2} more"


def collect_directory_sources(directory: Path) -> list[Path]:
    """Collect allowed file paths from a source directory recursively."""
    collected: list[Path] = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            msg = f"Symlink sources are not supported for ingest: {path}"
            raise helpers.WikiError(msg)
        if path.is_file():
            collected.append(path.resolve())
    return collected


def expand_sources(sources: Sequence[Path]) -> list[Path]:
    """Expand source arguments into a deterministic list of file paths."""
    expanded: list[Path] = []
    seen: set[Path] = set()

    for source in sources:
        if source.is_symlink():
            msg = f"Symlink sources are not supported for ingest: {source}"
            raise helpers.WikiError(msg)
        if not source.exists():
            msg = f"Source path not found: {source}"
            raise helpers.WikiError(msg)

        if source.is_file():
            resolved = source.resolve()
            if resolved not in seen:
                expanded.append(resolved)
                seen.add(resolved)
            continue

        if source.is_dir():
            directory_files = collect_directory_sources(source)
            if not directory_files:
                msg = f"Source directory is empty: {source}"
                raise helpers.WikiError(msg)
            for file_path in directory_files:
                if file_path not in seen:
                    expanded.append(file_path)
                    seen.add(file_path)
            continue

        msg = f"Unsupported source path type: {source}"
        raise helpers.WikiError(msg)

    return expanded


def build_ingest_review_prompt(
    topic: str, staged_paths: Sequence[Path], note: str | None
) -> str:
    """Build the ingest review prompt for staged source material."""
    memory_paths = [f"/raw/{path.name}" for path in staged_paths]
    source_block = "\n".join(f"- {path}" for path in memory_paths)
    note_block = note or "(none)"
    return (
        f"Review the staged sources for topic '{topic}' and prepare a deep ingest plan.\n\n"
        "Phase constraint: review-only. Do not create, edit, move, or delete files yet.\n\n"
        "Analysis standards:\n"
        "- Read every staged source before proposing wiki edits.\n"
        "- Distinguish direct evidence from inference.\n"
        "- Prefer canonical page updates over creating fragmented pages.\n"
        "- Preserve uncertainty; do not invent unsupported claims.\n"
        "- Use source filename citations for non-trivial claims.\n\n"
        "Required output format (markdown):\n"
        "## 1) Source-by-source extraction\n"
        "- For each source: purpose, key claims, useful details, confidence/risk notes.\n\n"
        "## 2) Proposed wiki change set\n"
        "- Enumerate concrete file actions under `/wiki/`.\n"
        "- For each file: create/update, why it changes, and core additions.\n"
        "- Prefer canonical concept/entity/theme pages over per-source summary files.\n"
        "- Prefer a flat `/wiki/` layout by default; create subdirectories only when they clearly improve organization at current scale.\n\n"
        "## 3) Cross-source synthesis and structure\n"
        "- Shared themes, entity relationships, timelines, and important backlinks.\n"
        "- Mention candidate canonical pages if current structure is fragmented.\n\n"
        "## 4) Contradictions and unresolved claims\n"
        "- Conflicts between sources, what is unresolved, and how pages should reflect this.\n\n"
        "## 5) Index updates and recency notes\n"
        "- Exact updates needed for `/wiki/index.md`.\n"
        "- Optional recency note candidates for runner-managed `/log.md` timeline summaries.\n\n"
        "## 6) Gaps and follow-up questions\n"
        "- Missing evidence, suggested next sources, and optional future page candidates.\n\n"
        "Expect broad wiki impact where warranted; one source can update many pages.\n"
        "Never write to `/raw/`.\n\n"
        f"Staged sources:\n{source_block}\n\n"
        f"Operator note: {note_block}\n"
    )


def build_ingest_apply_prompt(
    topic: str,
    staged_paths: Sequence[Path],
    review_summary: str,
    note: str | None,
) -> str:
    """Build the ingest apply prompt."""
    memory_paths = [f"/raw/{path.name}" for path in staged_paths]
    source_block = "\n".join(f"- {path}" for path in memory_paths)
    note_block = note or "(none)"
    return (
        f"Apply an approved ingest update for topic '{topic}'.\n\n"
        "Required workflow:\n"
        "1) Read all staged files in `/raw/` before editing wiki content.\n"
        "2) Update canonical concept/entity/theme pages with high-signal evidence.\n"
        "3) Integrate cross-source synthesis, not just per-source summaries.\n"
        "4) Mark contradictions explicitly and preserve unresolved uncertainty.\n"
        "5) Update `/wiki/index.md`.\n"
        "6) Do not edit `/log.md`; the runner appends structured timeline entries.\n"
        "7) Never write to `/raw/`.\n"
        "8) Prefer files directly under `/wiki/`; only create subdirectories when they are clearly needed for organization.\n\n"
        "Writing standards:\n"
        "- Keep pages scannable with clear headings and concise prose.\n"
        "- Use source filename citations for non-trivial claims.\n"
        "- Avoid duplicative pages; merge into canonical pages when possible.\n"
        "- If evidence is weak or conflicting, state that directly.\n\n"
        "Return a concise apply report after edits:\n"
        "A) Files created\n"
        "B) Files updated\n"
        "C) Key synthesis changes\n"
        "D) Remaining uncertainties and suggested next ingest targets\n\n"
        f"Approved review plan:\n{review_summary}\n\n"
        f"Staged sources:\n{source_block}\n\n"
        f"Operator note: {note_block}\n"
    )


def confirm_ingest_apply(review: str, ask_user: Callable[[str], str]) -> bool:
    """Ask operator to approve ingest apply after the review phase."""
    review_block = review.strip() or "(no review summary returned by model)"
    prompt = (
        "Ingest review summary:\n\n"
        f"{review_block}\n\n"
        "Apply these wiki updates now? [y/N]: "
    )
    try:
        response = ask_user(prompt)
    except EOFError as exc:
        msg = "Ingest review requires an interactive confirmation response."
        raise helpers.WikiError(msg) from exc
    return response.strip().lower() in {"y", "yes"}


def run_ingest_workspace(
    config: RunnerConfig, workspace_dir: Path, deps: CliDeps
) -> IngestResult:
    """Run ingest mode against a pulled workspace directory."""
    expanded_sources = expand_sources(config.sources)
    staged = helpers._stage_sources(expanded_sources, workspace_dir)
    source_count = len(staged)
    source_hint = _ingest_source_hint(staged)
    if config.review:
        review_prompt = build_ingest_review_prompt(config.topic, staged, config.note)
        review_summary = deps.run_agent_review_mode(
            workspace_dir, config.topic, review_prompt, config.model
        )
        helpers._append_log_entry(
            workspace_dir,
            "ingest.review",
            "completed",
            metadata={"source_count": source_count, "source_hint": source_hint},
            summary=review_summary,
        )

        approved = confirm_ingest_apply(review_summary, deps.ask_user)
        if not approved:
            cancel_summary = "Operator declined apply after ingest review."
            helpers._append_log_entry(
                workspace_dir,
                "ingest.apply",
                "canceled",
                metadata={"source_count": source_count, "source_hint": source_hint},
                summary=cancel_summary,
            )
            return IngestResult(
                answer="Ingest canceled after review. No wiki changes were applied.",
                should_push=True,
            )

        apply_prompt = build_ingest_apply_prompt(
            config.topic,
            staged,
            review_summary,
            config.note,
        )
    else:
        apply_prompt = build_ingest_apply_prompt(
            config.topic,
            staged,
            (
                "No explicit review phase was run. First perform review-quality "
                "analysis (source extraction, change planning, contradiction checks), "
                "then apply updates directly."
            ),
            config.note,
        )

    apply_answer = deps.run_agent_mode(
        workspace_dir, config.topic, apply_prompt, config.model
    )

    helpers._refresh_index(config.topic, workspace_dir)
    apply_metadata: dict[str, object] = {
        "source_count": source_count,
        "source_hint": source_hint,
    }
    if config.note:
        apply_metadata["note"] = config.note
    helpers._append_log_entry(
        workspace_dir,
        "ingest.apply",
        "applied",
        metadata=apply_metadata,
        summary=apply_answer or "Ingest applied.",
    )

    return IngestResult(answer=apply_answer or "Ingest applied.", should_push=True)
