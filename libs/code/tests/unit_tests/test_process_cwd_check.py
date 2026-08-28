"""Tests for the process working-directory guard."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from types import ModuleType

_SCRIPT = Path(__file__).parents[2] / "scripts" / "check_process_cwd.py"


def _load_check() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_process_cwd", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_new_process_cwd_read_fails_guard(tmp_path: Path) -> None:
    """An unreviewed process-cwd read must fail with its source location."""
    package = tmp_path / "deepagents_code"
    package.mkdir()
    (package / "request_path.py").write_text(
        "from pathlib import Path as P\n\ndef handle_request():\n    return P.cwd()\n",
        encoding="utf-8",
    )

    check = cast("Any", _load_check())
    check._ALLOWLIST = {}

    assert check.find_violations(package) == [
        "request_path.py:4: unreviewed Path.cwd call in handle_request"
    ]


def test_current_package_matches_allowlist() -> None:
    """The reviewed package must have no new or stale entries."""
    check = cast("Any", _load_check())
    package = Path(__file__).parents[2] / "deepagents_code"

    assert check.find_violations(package) == []
