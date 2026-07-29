"""Test that package version is consistent across configuration files."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

import deepagents
from deepagents._version import (
    __version__,
    _is_editable_install,
    _lc_version,
    _with_editable_local_version,
)


def test_version_matches_pyproject() -> None:
    """Verify that __version__ in __init__.py matches version in pyproject.toml."""
    # Get the version from the package __init__.py
    init_version = deepagents.__version__

    # Read the version from pyproject.toml
    pyproject_path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        pyproject_data = tomllib.load(f)

    pyproject_version = pyproject_data["project"]["version"]

    # Assert they match
    assert init_version == pyproject_version, (
        f"Version mismatch: __init__.py has '{init_version}' but "
        f"pyproject.toml has '{pyproject_version}'. "
        "Please update deepagents/__init__.py to match pyproject.toml."
    )


def _dist(*, name: str, direct_url: dict[str, object] | None) -> MagicMock:
    """Build a minimal `importlib.metadata.Distribution` stand-in."""
    dist = MagicMock()
    dist.name = name
    dist.metadata = {"Name": name}
    if direct_url is None:
        dist.read_text.return_value = None
    else:
        dist.read_text.return_value = json.dumps(direct_url)
    return dist


class TestWithEditableLocalVersion:
    """Tests for `_with_editable_local_version`."""

    def test_appends_editable_local_segment(self) -> None:
        assert _with_editable_local_version("0.6.12") == "0.6.12+editable"

    def test_preserves_existing_local_segment(self) -> None:
        assert _with_editable_local_version("0.6.12+build") == "0.6.12+build.editable"

    def test_returns_original_for_invalid_version(self) -> None:
        assert _with_editable_local_version("not-a-version") == "not-a-version"


class TestIsEditableInstall:
    """Tests for `_is_editable_install`."""

    def test_true_when_pep610_marks_editable(self) -> None:
        editable = _dist(
            name="deepagents",
            direct_url={
                "url": "file:///tmp/deepagents",
                "dir_info": {"editable": True},
            },
        )
        with patch("deepagents._version.distributions", return_value=[editable]):
            assert _is_editable_install() is True

    def test_false_when_not_editable(self) -> None:
        wheel = _dist(
            name="deepagents",
            direct_url={
                "url": "https://example.com/deepagents-0.6.12.tar.gz",
                "dir_info": {},
            },
        )
        with patch("deepagents._version.distributions", return_value=[wheel]):
            assert _is_editable_install() is False

    def test_false_when_direct_url_missing(self) -> None:
        egg_info = _dist(name="deepagents", direct_url=None)
        with patch("deepagents._version.distributions", return_value=[egg_info]):
            assert _is_editable_install() is False

    def test_ignores_cwd_egg_info_shadowing_editable_install(self) -> None:
        """A local `*.egg-info` without PEP 610 data must not hide site-packages."""
        egg_info = _dist(name="deepagents", direct_url=None)
        editable = _dist(
            name="deepagents",
            direct_url={
                "url": "file:///tmp/deepagents",
                "dir_info": {"editable": True},
            },
        )
        with patch("deepagents._version.distributions", return_value=[egg_info, editable]):
            assert _is_editable_install() is True

    def test_false_when_no_distributions(self) -> None:
        with patch("deepagents._version.distributions", return_value=[]):
            assert _is_editable_install() is False

    def test_false_when_metadata_lookup_raises(self) -> None:
        with patch(
            "deepagents._version.distributions",
            side_effect=OSError("metadata unavailable"),
        ):
            assert _is_editable_install() is False


class TestLcVersion:
    """Tests for `_lc_version`."""

    def test_plain_release_install(self) -> None:
        with patch("deepagents._version._is_editable_install", return_value=False):
            assert _lc_version() == __version__

    def test_editable_install_gets_local_segment(self) -> None:
        with patch("deepagents._version._is_editable_install", return_value=True):
            assert _lc_version() == f"{__version__}+editable"
