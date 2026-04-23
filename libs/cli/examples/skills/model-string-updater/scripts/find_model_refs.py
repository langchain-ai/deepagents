#!/usr/bin/env python3
"""Find LLM model identifier references across the deepagents monorepo.

Run from the repo root. Scans docstrings, Markdown, example configs, CI files,
and test files for model strings (OpenAI, Anthropic, Google/Vertex Gemini),
and prints hits grouped by directory so the user can plan edits by category.

Usage:
    scripts/find_model_refs.py                        # default pattern, full repo
    scripts/find_model_refs.py --exclude-tests        # skip tests/
    scripts/find_model_refs.py --paths libs/deepagents
    scripts/find_model_refs.py --pattern 'openai:gpt-5'
    scripts/find_model_refs.py --json                 # machine-readable

Exits non-zero only on bad arguments; an empty result is a valid outcome.
"""

from __future__ import annotations

import argparse
import json
import operator
import re
import sys
from pathlib import Path

DEFAULT_PATTERN = (
    r"(?:openai:|anthropic:|google_genai:|google_vertexai:|baseten:)?"
    r"(?:"
    r"gpt-[0-9][\w.\-]*"
    r"|o[134](?:-[\w.\-]+)?"
    r"|codex-[\w.\-]+"
    r"|chatgpt[\w.\-]*"
    r"|claude[\w.\-]*"
    r"|gemini[\w.\-]*"
    r")"
)

DEFAULT_EXTENSIONS = {
    ".py",
    ".md",
    ".mdx",
    ".rst",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ipynb",
}

SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
}


def _iter_files(roots: list[Path], exclude_tests: bool) -> list[Path]:
    """Return candidate files under ``roots``, filtered by extension and dir skips.

    Returns:
        Sorted list of file paths to scan.
    """
    out: list[Path] = []
    for root in roots:
        if root.is_file():
            out.append(root)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if path.suffix not in DEFAULT_EXTENSIONS:
                continue
            if exclude_tests and (
                "tests" in path.parts or path.name.startswith("test_")
            ):
                continue
            out.append(path)
    return sorted(set(out))


def _scan_file(path: Path, regex: re.Pattern[str]) -> list[tuple[int, str]]:
    """Return ``(line_number, line)`` hits found in ``path``.

    Returns:
        A list of (line_number, line_text) tuples for every matching line.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if regex.search(line):
            hits.append((lineno, line.rstrip()))
    return hits


def _category_for(path: Path) -> str:
    """Bucket a path into one of the skill's 6 categories (best effort).

    Returns:
        One of ``"2-examples"``, ``"3-docstrings-or-runtime"``, ``"4-ci-evals"``,
        ``"5-tests"``, or ``"other"``.
    """
    parts = path.parts
    if "tests" in parts or path.name.startswith("test_"):
        return "5-tests"
    if path.match(".github/**/*") or ".github" in parts:
        return "4-ci-evals"
    if "MODEL_GROUPS.md" in path.name:
        return "4-ci-evals"
    if "examples" in parts:
        return "2-examples"
    if path.suffix == ".py":
        return "3-docstrings-or-runtime"
    if path.suffix in {".md", ".mdx", ".rst"}:
        return "3-docstrings-or-runtime"
    return "other"


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Returns:
        Process exit code (``0`` on success, ``2`` on argument errors).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=(
            "Regex (Python re syntax) to match model strings. "
            "Default matches common OpenAI/Anthropic/Gemini names."
        ),
    )
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["."],
        help="Paths to scan (files or directories). Default: repo root.",
    )
    parser.add_argument(
        "--exclude-tests",
        action="store_true",
        help="Skip any path under 'tests/' or named 'test_*.py'.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of grouped text output.",
    )
    args = parser.parse_args(argv)

    try:
        regex = re.compile(args.pattern)
    except re.error as exc:
        print(f"error: invalid --pattern: {exc}", file=sys.stderr)
        return 2

    roots = [Path(p) for p in args.paths]
    missing = [r for r in roots if not r.exists()]
    if missing:
        for r in missing:
            print(f"error: path not found: {r}", file=sys.stderr)
        return 2

    files = _iter_files(roots, exclude_tests=args.exclude_tests)

    results: dict[str, list[dict[str, object]]] = {}
    for f in files:
        hits = _scan_file(f, regex)
        if not hits:
            continue
        cat = _category_for(f)
        results.setdefault(cat, []).append(
            {
                "path": str(f),
                "hits": [{"line": ln, "text": txt} for ln, txt in hits],
            }
        )

    if args.json:
        json.dump(results, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0

    if not results:
        print("No model references found.")
        return 0

    total = 0
    for cat in sorted(results):
        print(f"\n=== {cat} ===")
        for entry in sorted(results[cat], key=operator.itemgetter("path")):
            path = entry["path"]
            for hit in entry["hits"]:  # type: ignore[union-attr]
                total += 1
                print(f"{path}:{hit['line']}: {hit['text']}")
    file_count = sum(len(v) for v in results.values())
    print(f"\n{total} match(es) across {file_count} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
