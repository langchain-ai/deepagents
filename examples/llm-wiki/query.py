"""Query-specific workflow for the LLM wiki."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from models import CliDeps, RunnerConfig
import helpers

@dataclass(frozen=True)
class QueryResult:
    """Result from one query workspace pass."""

    answer: str
    should_push: bool
    filed_path: str | None


@dataclass(frozen=True)
class QueryDecision:
    """Parsed decision output from query analysis."""

    answer: str
    should_file: bool
    reason: str


_QUERY_DECISION_PATTERN = re.compile(
    r"^FILING_DECISION:\s*(file|skip)\s*$", re.IGNORECASE | re.MULTILINE
)
_QUERY_REASON_PATTERN = re.compile(
    r"^FILING_REASON:\s*(.+)$", re.IGNORECASE | re.MULTILINE
)


def query_slug(question: str) -> str:
    """Create a stable slug for query filing pages."""
    slug = helpers._slugify_topic(question)
    shortened = slug[:80].rstrip("-")
    return shortened or "query"


def query_target_path(question: str) -> str:
    """Return the canonical wiki path for a filed query answer."""
    return f"/wiki/query/{query_slug(question)}.md"


def build_query_prompt(topic: str, question: str) -> str:
    """Build the read-only query prompt with filing decision output."""
    return (
        f"Answer this question about '{topic}': {question}\n\n"
        "This is analysis-only. Do not create, edit, move, or delete files.\n\n"
        "Required workflow:\n"
        "1) Read `/wiki/index.md` first and use its categorized summaries/metadata to "
        "choose candidate pages.\n"
        "2) Read recent `/log.md` entries (latest ~10 `## [` headings) to understand what "
        "was ingested, queried, or linted recently.\n"
        "3) Prefer checking relevant prior `/wiki/query/*.md` pages first as a discovery step.\n"
        "4) Use those query pages to identify likely canonical `/wiki/*.md` pages and topics.\n"
        "5) Read the canonical wiki pages before final synthesis.\n"
        "6) Provide a grounded answer with wiki file path citations.\n"
        "7) Decide whether this answer should be filed as a durable wiki page.\n\n"
        "Evidence policy:\n"
        "- Treat `/log.md` as operational recency context, not primary factual evidence.\n"
        "- Treat `/wiki/query/*.md` pages as routing hints, not primary evidence.\n"
        "- Cite canonical wiki pages for final claims whenever possible.\n"
        "- If a claim is only supported by query pages, explicitly note uncertainty and missing canonical grounding.\n\n"
        "Filing policy:\n"
        "- Choose `file` when the answer has durable reuse value for future research.\n"
        "- Choose `skip` for ad-hoc or low-reuse answers.\n\n"
        "Output format (exact keys):\n"
        "ANSWER:\n"
        "<markdown answer with citations>\n\n"
        "FILING_DECISION: file|skip\n"
        "FILING_REASON: <one sentence>\n"
    )


def parse_query_decision(raw_response: str) -> QueryDecision:
    """Parse query decision markers from model output."""
    response = raw_response.strip()

    decision_match = _QUERY_DECISION_PATTERN.search(response)
    reason_match = _QUERY_REASON_PATTERN.search(response)

    should_file = (
        decision_match is not None and decision_match.group(1).lower() == "file"
    )
    reason = (
        reason_match.group(1).strip()
        if reason_match is not None
        else "Decision marker missing; defaulted to skip."
    )

    answer_text = response
    if decision_match is not None:
        answer_text = response[: decision_match.start()].strip()
    if answer_text.upper().startswith("ANSWER:"):
        answer_text = answer_text[len("ANSWER:") :].strip()
    if not answer_text:
        answer_text = response or "No answer returned."

    return QueryDecision(answer=answer_text, should_file=should_file, reason=reason)


def build_query_apply_prompt(
    topic: str,
    question: str,
    answer_draft: str,
    filing_reason: str,
    target_path: str,
) -> str:
    """Build the query filing prompt for durable wiki page updates."""
    return (
        f"File a durable query answer for topic '{topic}'.\n\n"
        f"Create or overwrite exactly: `{target_path}`\n\n"
        "Requirements:\n"
        "1) Write a clean, scannable markdown page at the target path.\n"
        "2) Preserve grounded claims and include wiki file path citations.\n"
        "3) Include these sections: `Question`, `Answer`, and `Sources`.\n"
        "4) Keep the answer focused and useful for future reuse.\n"
        "5) Never write to `/raw/`.\n\n"
        f"Filing reason: {filing_reason}\n\n"
        f"Question: {question}\n\n"
        f"Answer draft:\n{answer_draft}\n"
    )


def run_query_workspace(
    config: RunnerConfig, workspace_dir: Path, deps: CliDeps
) -> QueryResult:
    """Run query mode and optionally file durable answers into the wiki."""
    question = config.question or ""
    review_prompt = build_query_prompt(config.topic, question)
    review_response = deps.run_agent_review_mode(
        workspace_dir, config.topic, review_prompt, config.model
    )
    decision = parse_query_decision(review_response)
    review_outcome = "file" if decision.should_file else "skip"
    helpers._append_log_entry(
        workspace_dir,
        "query.review",
        review_outcome,
        metadata={
            "question": question,
            "decision": review_outcome,
        },
        summary=decision.answer,
    )

    if not decision.should_file:
        return QueryResult(answer=decision.answer, should_push=True, filed_path=None)

    target_path = query_target_path(question)
    apply_prompt = build_query_apply_prompt(
        config.topic,
        question,
        decision.answer,
        decision.reason,
        target_path,
    )
    deps.run_agent_mode(workspace_dir, config.topic, apply_prompt, config.model)

    helpers._refresh_index(config.topic, workspace_dir)
    helpers._append_log_entry(
        workspace_dir,
        "query.apply",
        "filed",
        metadata={"question": question, "path": target_path},
        summary=decision.answer,
    )
    return QueryResult(answer=decision.answer, should_push=True, filed_path=target_path)
