"""Index-specific helpers for wiki content catalog generation."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

_INDEX_CATEGORY_ORDER = (
    "Entities",
    "Concepts",
    "Sources",
    "Timelines",
    "Queries",
    "Syntheses",
    "Other Pages",
)
_INDEX_DIRECTORY_CATEGORIES = {
    "entity": "Entities",
    "entities": "Entities",
    "concept": "Concepts",
    "concepts": "Concepts",
    "source": "Sources",
    "sources": "Sources",
    "timeline": "Timelines",
    "timelines": "Timelines",
    "query": "Queries",
    "queries": "Queries",
    "synthesis": "Syntheses",
    "syntheses": "Syntheses",
}
_INDEX_DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_INDEX_SOURCE_REF_PATTERN = re.compile(r"/raw/([A-Za-z0-9._/\-]+)")


def empty_index_text(topic: str) -> str:
    """Build default index markdown for empty wikis."""
    lines = [
        f"# {topic} Wiki",
        "",
        "Content catalog for wiki navigation and retrieval.",
        "Read this page first during query workflows.",
        "",
        "## Other Pages",
        "",
        "- _No pages yet._",
    ]
    return "\n".join(lines) + "\n"


def _index_fallback_title(path: Path) -> str:
    """Create a human-readable title from a markdown file path."""
    return path.stem.replace("-", " ").replace("_", " ").strip().title()


def _strip_markdown_inline(text: str) -> str:
    """Strip basic inline markdown to plain text for index snippets."""
    stripped = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    stripped = re.sub(r"`([^`]+)`", r"\1", stripped)
    stripped = re.sub(r"[*_~]", "", stripped)
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped.strip(" -:")


def _page_title_for_index(relative_path: Path, content: str) -> str:
    """Extract a display title for an index entry."""
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        heading = _strip_markdown_inline(stripped.lstrip("#").strip())
        if heading:
            return heading
    return _index_fallback_title(relative_path)


def _page_summary_for_index(content: str) -> str:
    """Extract a one-line summary for an index entry from page content."""
    in_code_block = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or not stripped or stripped.startswith("#"):
            continue
        candidate = _strip_markdown_inline(stripped.lstrip("-*+ ").strip())
        if not candidate:
            continue
        if len(candidate) > 150:
            return f"{candidate[:147].rstrip()}..."
        return candidate
    return "No summary available."


def _page_metadata_for_index(content: str) -> tuple[str, ...]:
    """Extract compact optional metadata for an index entry."""
    metadata: list[str] = []
    dates = sorted(set(_INDEX_DATE_PATTERN.findall(content)))
    if dates:
        metadata.append(f"date: {dates[-1]}")
    source_refs = {
        match.group(1).rstrip(".,;:)}")
        for match in _INDEX_SOURCE_REF_PATTERN.finditer(content)
    }
    if source_refs:
        metadata.append(f"sources: {len(source_refs)}")
    return tuple(metadata)


def _index_category_for_page(relative_path: Path) -> str:
    """Determine an index category label for a wiki page path."""
    parts = relative_path.parts
    if len(parts) <= 1:
        return "Other Pages"
    directory_category = _INDEX_DIRECTORY_CATEGORIES.get(parts[0].lower())
    if directory_category:
        return directory_category
    return "Other Pages"


def _build_index_text(topic: str, wiki_dir: Path) -> str:
    """Construct index markdown for current wiki pages."""
    pages = [
        path for path in sorted(wiki_dir.rglob("*.md")) if path.name != "index.md"
    ]
    if not pages:
        return empty_index_text(topic)

    section_lines: dict[str, list[str]] = {
        category: [] for category in _INDEX_CATEGORY_ORDER
    }
    for page in pages:
        relative_path = page.relative_to(wiki_dir)
        relative = relative_path.as_posix()
        content = page.read_text(encoding="utf-8")
        title = _page_title_for_index(relative_path, content)
        summary = _page_summary_for_index(content)
        metadata = _page_metadata_for_index(content)
        entry = f"- [{title}]({relative}) - {summary}"
        if metadata:
            entry = f"{entry} _({'; '.join(metadata)})_"
        section_lines[_index_category_for_page(relative_path)].append(entry)

    lines = [
        f"# {topic} Wiki",
        "",
        "Content catalog for wiki navigation and retrieval.",
        "Read this page first during query workflows.",
        "",
    ]
    for category in _INDEX_CATEGORY_ORDER:
        entries = section_lines[category]
        if not entries:
            continue
        lines.extend([f"## {category}", ""])
        lines.extend(entries)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def refresh_index(
    topic: str,
    workspace_dir: Path,
    *,
    write_text: Callable[[Path, str], None] | None = None,
) -> None:
    """Rebuild `/wiki/index.md` from current markdown pages."""
    wiki_dir = workspace_dir / "wiki"
    wiki_dir.mkdir(parents=True, exist_ok=True)
    index_path = wiki_dir / "index.md"
    content = _build_index_text(topic, wiki_dir)
    if write_text is None:
        index_path.write_text(content, encoding="utf-8")
        return
    write_text(index_path, content)


__all__ = ["empty_index_text", "refresh_index"]
