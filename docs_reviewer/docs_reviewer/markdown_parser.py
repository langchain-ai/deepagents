"""Markdown parsing utilities for extracting code snippets."""

import re
from pathlib import Path
from typing import TypedDict


class CodeSnippet(TypedDict):
    """Represents a code snippet extracted from markdown."""

    language: str
    code: str
    start_line: int
    end_line: int
    metadata: dict[str, str]  # Additional metadata from code fence


def extract_code_snippets(markdown_file: Path) -> list[CodeSnippet]:
    """
    Extract all code snippets from a markdown file.

    Args:
        markdown_file: Path to the markdown file

    Returns:
        List of code snippets with their metadata
    """
    with open(markdown_file, encoding="utf-8") as f:
        content = f.read()

    snippets: list[CodeSnippet] = []
    lines = content.split("\n")

    # Pattern to match code fence openings with optional metadata
    # Examples: ```python, ```python {title="example.py"}, ```py
    fence_pattern = re.compile(r"^```(\w+)?\s*(\{[^}]*\})?\s*$")

    i = 0
    while i < len(lines):
        line = lines[i]
        match = fence_pattern.match(line)

        if match:
            language = match.group(1) or "text"
            metadata_str = match.group(2) or "{}"

            # Parse metadata if present
            metadata: dict[str, str] = {}
            if metadata_str != "{}":
                # Simple parsing of {key="value"} style metadata
                metadata_pattern = re.compile(r'(\w+)="([^"]*)"')
                metadata = dict(metadata_pattern.findall(metadata_str))

            start_line = i + 1
            code_lines = []

            # Find the closing fence
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1

            end_line = i

            code = "\n".join(code_lines)

            snippets.append(
                CodeSnippet(
                    language=language,
                    code=code,
                    start_line=start_line,
                    end_line=end_line,
                    metadata=metadata,
                )
            )

        i += 1

    return snippets


def filter_executable_snippets(snippets: list[CodeSnippet]) -> list[CodeSnippet]:
    """
    Filter snippets to only include executable code.

    Excludes snippets that are:
    - Plain text or markdown
    - Shell output examples
    - Snippets marked as non-executable in metadata

    Args:
        snippets: List of all code snippets

    Returns:
        List of executable code snippets
    """
    executable_languages = {
        "python",
        "py",
        "javascript",
        "js",
        "typescript",
        "ts",
        "bash",
        "sh",
        "shell",
    }

    filtered = []
    for snippet in snippets:
        # Skip if language is not executable
        if snippet["language"].lower() not in executable_languages:
            continue

        # Skip if metadata indicates non-executable
        if snippet["metadata"].get("exec") == "false":
            continue

        # Skip if metadata indicates it's output
        if snippet["metadata"].get("output") == "true":
            continue

        filtered.append(snippet)

    return filtered


def categorize_snippets(snippets: list[CodeSnippet]) -> dict[str, list[CodeSnippet]]:
    """
    Categorize snippets by language.

    Args:
        snippets: List of code snippets

    Returns:
        Dictionary mapping language to list of snippets
    """
    categories: dict[str, list[CodeSnippet]] = {}

    for snippet in snippets:
        lang = snippet["language"].lower()
        if lang not in categories:
            categories[lang] = []
        categories[lang].append(snippet)

    return categories
