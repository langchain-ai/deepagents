#!/usr/bin/env python3
"""Quick lint check helper for the code-review skill.

Scans Python files for common issues that a full linter might miss or that
are worth flagging during code review:

- Files missing a module docstring
- Functions longer than 50 lines
- Bare ``except:`` clauses

Usage::

    python /skills/code-review/lint_check.py [path ...]

If no paths are given, scans the current directory recursively.
"""

import ast
import sys
from pathlib import Path


def check_file(path: Path) -> list[str]:
    """Return a list of warnings for a single Python file."""
    warnings: list[str] = []
    try:
        source = path.read_text(encoding="utf-8")
    except Exception as exc:
        return [f"{path}: could not read ({exc})"]

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{path}:{exc.lineno}: syntax error: {exc.msg}"]

    # Check for missing module docstring
    if not ast.get_docstring(tree):
        warnings.append(f"{path}:1: missing module docstring")

    for node in ast.walk(tree):
        # Long functions
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            length = (node.end_lineno or node.lineno) - node.lineno + 1
            if length > 50:
                warnings.append(
                    f"{path}:{node.lineno}: function '{node.name}' is {length} lines long (>50)"
                )

        # Bare except
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            warnings.append(f"{path}:{node.lineno}: bare 'except:' clause")

    return warnings


def main(paths: list[str]) -> int:
    targets = [Path(p) for p in paths] if paths else [Path(".")]
    all_warnings: list[str] = []

    for target in targets:
        if target.is_file() and target.suffix == ".py":
            all_warnings.extend(check_file(target))
        elif target.is_dir():
            for py_file in sorted(target.rglob("*.py")):
                all_warnings.extend(check_file(py_file))

    for w in all_warnings:
        print(w)

    if all_warnings:
        print(f"\n{len(all_warnings)} warning(s) found.")
        return 1

    print("No warnings found.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
