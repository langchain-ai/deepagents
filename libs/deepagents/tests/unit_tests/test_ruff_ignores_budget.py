from __future__ import annotations

from pathlib import Path

import tomli as tomllib


def test_do_not_increase_number_of_allowed_ignores() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    ruff_tool = pyproject.get("tool", {}).get("ruff", {})

    lint_ignore = ruff_tool.get("lint", {}).get("ignore", [])
    per_file_ignores = ruff_tool.get("lint", {}).get("per-file-ignores", {})

    ignore_count = 0
    ignore_count += len(lint_ignore)

    for ignores in per_file_ignores.values():
        ignore_count += len(ignores)

    assert ignore_count < 196, f"DO NOT INCREASE THE NUMBER OF ALLOWED IGNORES. OK TO DECREASE. Current ignores: {ignore_count}"
