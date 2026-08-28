"""Tests for the process working-directory guard."""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

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


def _detected_sites(check: Any, package: Path) -> list[Any]:  # noqa: ANN401
    """Return the call sites the guard finds, so a test can allowlist them."""
    sites = []
    for path in sorted(package.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        visitor = check._Visitor(path.relative_to(package).as_posix(), tree)
        visitor.visit(tree)
        sites.extend(site for site, _line in visitor.sites)
    return sites


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


def test_nullable_project_root_read_fails_guard(tmp_path: Path) -> None:
    """A `find_project_root` call with an argument must also fail the guard."""
    package = tmp_path / "deepagents_code"
    package.mkdir()
    (package / "request_path.py").write_text(
        "from deepagents_code.project_utils import find_project_root as root\n"
        "\n"
        "def handle_request(start_path=None):\n"
        "    return root(start_path)\n",
        encoding="utf-8",
    )

    check = cast("Any", _load_check())
    check._ALLOWLIST = {}

    assert check.find_violations(package) == [
        "request_path.py:4: unreviewed find_project_root call in handle_request"
    ]


def test_defining_module_read_fails_guard(tmp_path: Path) -> None:
    """The module that defines `find_project_root` must not be exempt."""
    package = tmp_path / "deepagents_code"
    package.mkdir()
    (package / "project_utils.py").write_text(
        "def find_project_root(start_path=None):\n"
        "    return start_path\n"
        "\n"
        "def get_context(user_cwd):\n"
        "    return find_project_root(user_cwd)\n",
        encoding="utf-8",
    )

    check = cast("Any", _load_check())
    check._ALLOWLIST = {}

    assert check.find_violations(package) == [
        "project_utils.py:5: unreviewed find_project_root call in get_context"
    ]


def test_added_read_does_not_shift_reviewed_entries(tmp_path: Path) -> None:
    """A new read must be reported, and must not take a reviewed reason."""
    package = tmp_path / "deepagents_code"
    package.mkdir()
    module = package / "request_path.py"
    reviewed = (
        "from pathlib import Path\n"
        "\n"
        "def build():\n"
        "    first = Path.cwd()\n"
        "    second = Path.cwd()\n"
    )
    module.write_text(reviewed, encoding="utf-8")

    check = cast("Any", _load_check())
    check._ALLOWLIST = dict.fromkeys(
        _detected_sites(check, package), "reviewed: client process"
    )
    assert check.find_violations(package) == []

    module.write_text(
        reviewed.replace("    first =", "    fresh = Path.cwd()\n    first ="),
        encoding="utf-8",
    )

    # The added read is reported at its own line, and neither reviewed entry
    # goes stale, so no reason moves to a call nobody reviewed.
    assert check.find_violations(package) == [
        "request_path.py:4: unreviewed Path.cwd call in build"
    ]


def test_stale_allowlist_entry_reports_its_reason(tmp_path: Path) -> None:
    """A removed read must report the reason that no longer applies."""
    package = tmp_path / "deepagents_code"
    package.mkdir()
    (package / "request_path.py").write_text(
        "from pathlib import Path\n\ndef build():\n    return Path.cwd()\n",
        encoding="utf-8",
    )

    check = cast("Any", _load_check())
    (site,) = _detected_sites(check, package)
    check._ALLOWLIST = {site: "The builder runs in the client process."}
    (package / "request_path.py").write_text("", encoding="utf-8")

    assert check.find_violations(package) == [
        (
            f"stale allowlist entry: request_path.py: Path.cwd [{site.token}] "
            "in build: The builder runs in the client process."
        )
    ]


def test_unreadable_file_stops_the_check(tmp_path: Path) -> None:
    """A file the check cannot parse must fail instead of going unchecked."""
    package = tmp_path / "deepagents_code"
    package.mkdir()
    (package / "broken.py").write_text("def build(:\n", encoding="utf-8")

    check = cast("Any", _load_check())

    with pytest.raises(SystemExit, match=r"broken\.py: cannot be parsed"):
        check.find_violations(package)


def test_current_package_matches_allowlist() -> None:
    """The reviewed package must have no new or stale entries."""
    check = cast("Any", _load_check())
    package = Path(__file__).parents[2] / "deepagents_code"

    assert check.find_violations(package) == []
