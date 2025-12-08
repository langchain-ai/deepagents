"""Markdown writing utilities for creating corrected documentation."""

from pathlib import Path
from typing import Any
from datetime import datetime
import difflib

from docs_reviewer.markdown_parser import CodeSnippet


def write_corrected_markdown(
    original_file: Path,
    output_file: Path,
    snippets: list[CodeSnippet],
    results: list[dict[str, Any]],
    inline: bool = True,
) -> dict[str, Any]:
    """
    Write a corrected markdown file with updated code snippets.

    Args:
        original_file: Path to the original markdown file
        output_file: Path to write the corrected markdown (ignored if inline=True)
        snippets: List of original code snippets
        results: List of review results for each snippet
        inline: If True, edits the original file in place. If False, creates a new file.

    Returns:
        Dictionary with diff information
    """
    # Read the original file
    with open(original_file, encoding="utf-8") as f:
        original_lines = f.readlines()

    # Create a mapping of snippet indices to results
    result_map = {r["snippet_index"]: r for r in results}

    # Track which lines need to be replaced
    replacements: dict[int, list[str]] = {}
    has_changes = False

    for i, snippet in enumerate(snippets):
        if i not in result_map:
            continue

        result = result_map[i]

        # Only replace if there were corrections
        if result["corrected_code"] != result["original_code"] or not result["success"]:
            has_changes = True
            start_line = snippet["start_line"] - 1  # Convert to 0-indexed
            end_line = snippet["end_line"]

            # Build the replacement content
            replacement_lines = []

            # Add a comment about the correction if snippet failed
            if not result["success"]:
                replacement_lines.append(f"<!-- REVIEW NOTE: This snippet had issues -->\n")
                replacement_lines.append(f"<!-- {_escape_html_comment(result['analysis'][:200])} -->\n")
                replacement_lines.append("\n")

            # Add the code fence
            replacement_lines.append(f"```{snippet['language']}\n")
            replacement_lines.append(result["corrected_code"])
            if not result["corrected_code"].endswith("\n"):
                replacement_lines.append("\n")
            replacement_lines.append("```\n")

            replacements[start_line] = replacement_lines

    # Build the corrected content
    corrected_lines = []
    i = 0
    while i < len(original_lines):
        if i in replacements:
            # Write the replacement
            for line in replacements[i]:
                corrected_lines.append(line)

            # Skip the original code block
            # Find the closing ```
            i += 1
            while i < len(original_lines) and not original_lines[i].strip().startswith("```"):
                i += 1
            i += 1  # Skip the closing ```
        else:
            corrected_lines.append(original_lines[i])
            i += 1

    # Determine output path
    if inline:
        target_file = original_file
    else:
        target_file = output_file

    # Write the corrected file
    with open(target_file, "w", encoding="utf-8") as f:
        f.writelines(corrected_lines)

    # Generate a minimalistic diff showing only the changes
    diff_lines = _generate_minimalistic_diff(original_lines, corrected_lines, original_file, target_file)

    return {
        "has_changes": has_changes,
        "diff": '\n'.join(diff_lines),
        "target_file": str(target_file),
    }


def _generate_minimalistic_diff(original_lines: list[str], corrected_lines: list[str],
                                  original_file: Path, target_file: Path) -> list[str]:
    """Generate a clean, minimalistic diff focusing on actual changes."""
    import difflib

    # Get the diff blocks
    matcher = difflib.SequenceMatcher(None, original_lines, corrected_lines)
    diff_output = []

    diff_output.append(f"--- {original_file}")
    diff_output.append(f"+++ {target_file}")

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue

        # Add location marker
        diff_output.append(f"@@ -{i1+1},{i2-i1} +{j1+1},{j2-j1} @@")

        # Show removed lines (non-empty only, or keep one empty to show the change)
        removed_lines = original_lines[i1:i2]
        non_empty_removed = [l for l in removed_lines if l.strip()]
        if non_empty_removed or (removed_lines and not corrected_lines[j1:j2]):
            for line in removed_lines:
                diff_output.append(f"-{line.rstrip()}")

        # Show added lines (non-empty only, or keep one empty to show the change)
        added_lines = corrected_lines[j1:j2]
        non_empty_added = [l for l in added_lines if l.strip()]
        if non_empty_added or (added_lines and not removed_lines):
            for line in added_lines:
                diff_output.append(f"+{line.rstrip()}")

    return diff_output


def write_review_report(
    output_file: Path,
    markdown_file: Path,
    snippets: list[CodeSnippet],
    results: list[dict[str, Any]],
) -> None:
    """
    Write a detailed review report.

    Args:
        output_file: Path to write the report
        markdown_file: Path to the reviewed markdown file
        snippets: List of code snippets
        results: List of review results
    """
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# Docs Review Report\n\n")
        f.write(f"**File:** {markdown_file}\n")
        f.write(f"**Date:** {datetime.now().isoformat()}\n\n")

        # Summary statistics
        total = len(results)
        successful = sum(1 for r in results if r["success"])
        failed = total - successful

        f.write(f"## Summary\n\n")
        f.write(f"- Total snippets reviewed: {total}\n")
        f.write(f"- Successful: {successful}\n")
        f.write(f"- Failed: {failed}\n")
        f.write(f"- Success rate: {(successful / total * 100):.1f}%\n\n")

        # Detailed results
        f.write(f"## Detailed Results\n\n")

        for i, result in enumerate(results, 1):
            snippet = snippets[result["snippet_index"]]

            status = "✅ Success" if result["success"] else "❌ Failed"
            f.write(f"### Snippet {i}: {status}\n\n")
            f.write(f"**Language:** {result['language']}\n")
            f.write(f"**Lines:** {result['start_line']}-{result['end_line']}\n\n")

            f.write(f"**Analysis:**\n")
            f.write(f"{result['analysis']}\n\n")

            if result["corrected_code"] != result["original_code"]:
                f.write(f"**Original Code:**\n")
                f.write(f"```{result['language']}\n")
                f.write(result["original_code"])
                f.write(f"\n```\n\n")

                f.write(f"**Corrected Code:**\n")
                f.write(f"```{result['language']}\n")
                f.write(result["corrected_code"])
                f.write(f"\n```\n\n")

            f.write("---\n\n")


def _escape_html_comment(text: str) -> str:
    """Escape text for safe inclusion in HTML comments."""
    return text.replace("--", "- -").replace("<!--", "< !--").replace("-->", "-- >")
