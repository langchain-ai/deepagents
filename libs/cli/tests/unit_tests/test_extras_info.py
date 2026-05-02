"""Tests for optional-dependency status inspection."""

import tomllib
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import MagicMock, patch

from deepagents_cli.extras_info import (
    _COMPOSITE_EXTRAS,
    MODEL_PROVIDER_EXTRAS,
    SANDBOX_EXTRAS,
    format_extras_status,
    format_extras_status_plain,
    get_extras_status,
    get_optional_dependency_status,
)

_PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _declared_extras() -> frozenset[str]:
    """Return non-composite extras declared in `pyproject.toml`."""
    data = tomllib.loads(_PYPROJECT_PATH.read_text(encoding="utf-8"))
    extras = data["project"]["optional-dependencies"]
    return frozenset(extras) - _COMPOSITE_EXTRAS


def test_returns_empty_when_distribution_missing() -> None:
    assert get_extras_status("does-not-exist-pkg-xyz-abc") == {}


def test_real_distribution_groups_entries_by_extra() -> None:
    # `langchain-anthropic` is declared under the `anthropic` extra and
    # also lives in the core dependency list, so it should always resolve
    # to an installed version when the CLI itself is installed.
    extras = get_extras_status()
    assert "anthropic" in extras
    pkgs = dict(extras["anthropic"])
    assert pkgs["langchain-anthropic"]


def test_real_distribution_skips_self_references() -> None:
    # Composite extras like `all-providers` list `deepagents-cli[...]`
    # entries; those should never surface as packages themselves.
    extras = get_extras_status()
    for pkgs in extras.values():
        for pkg_name, _version in pkgs:
            assert pkg_name.lower() != "deepagents-cli"


def test_missing_packages_are_omitted() -> None:
    mock_dist = MagicMock()
    mock_dist.requires = [
        "langchain-anthropic>=1.0.0 ; extra == 'anthropic'",
        "fake-absent-package>=1.0.0 ; extra == 'custom'",
        "partially-present>=1.0.0 ; extra == 'mixed'",
        "also-missing>=1.0.0 ; extra == 'mixed'",
    ]

    def fake_version(name: str) -> str:
        if name == "langchain-anthropic":
            return "1.4.0"
        if name == "partially-present":
            return "2.0.0"
        raise PackageNotFoundError(name)

    with (
        patch("deepagents_cli.extras_info.distribution", return_value=mock_dist),
        patch("deepagents_cli.extras_info.pkg_version", side_effect=fake_version),
    ):
        extras = get_extras_status()

    # Fully absent extras disappear; partially present extras keep only
    # the installed packages.
    assert extras == {
        "anthropic": [("langchain-anthropic", "1.4.0")],
        "mixed": [("partially-present", "2.0.0")],
    }


def test_optional_dependency_status_includes_missing_packages() -> None:
    mock_dist = MagicMock()
    mock_dist.requires = [
        "langchain-anthropic>=1.0.0 ; extra == 'anthropic'",
        "fake-absent-package>=1.0.0 ; extra == 'custom'",
        "partially-present>=1.0.0 ; extra == 'mixed'",
        "also-missing>=1.0.0 ; extra == 'mixed'",
    ]

    def fake_version(name: str) -> str:
        if name == "langchain-anthropic":
            return "1.4.0"
        if name == "partially-present":
            return "2.0.0"
        raise PackageNotFoundError(name)

    with (
        patch("deepagents_cli.extras_info.distribution", return_value=mock_dist),
        patch("deepagents_cli.extras_info.pkg_version", side_effect=fake_version),
    ):
        extras = get_optional_dependency_status()

    by_name = {extra.name: extra for extra in extras}
    assert by_name["anthropic"].ready is True
    assert by_name["anthropic"].installed == (("langchain-anthropic", "1.4.0"),)
    assert by_name["anthropic"].missing == ()
    assert by_name["custom"].ready is False
    assert by_name["custom"].installed == ()
    assert by_name["custom"].missing == ("fake-absent-package",)
    assert by_name["mixed"].ready is False
    assert by_name["mixed"].installed == (("partially-present", "2.0.0"),)
    assert by_name["mixed"].missing == ("also-missing",)


def test_skips_entries_without_extra_marker() -> None:
    # Core dependencies (no `extra ==` marker) must be ignored; only
    # extra-gated entries should be reported.
    mock_dist = MagicMock()
    mock_dist.requires = [
        "some-core-package>=1.0.0",
        "another-core>=1.0.0 ; python_version >= '3.11'",
        "gated-pkg>=1.0.0 ; extra == 'foo'",
    ]

    with (
        patch("deepagents_cli.extras_info.distribution", return_value=mock_dist),
        patch("deepagents_cli.extras_info.pkg_version", return_value="1.2.3"),
    ):
        extras = get_extras_status()

    assert extras == {"foo": [("gated-pkg", "1.2.3")]}


def test_skips_composite_self_referencing_extras() -> None:
    mock_dist = MagicMock()
    mock_dist.requires = [
        "deepagents-cli[anthropic,baseten] ; extra == 'some-bundle'",
        "langchain-anthropic>=1.0.0 ; extra == 'anthropic'",
    ]

    with (
        patch("deepagents_cli.extras_info.distribution", return_value=mock_dist),
        patch("deepagents_cli.extras_info.pkg_version", return_value="1.0.0"),
    ):
        extras = get_extras_status()

    # The self-reference is the only entry under `some-bundle`, so the
    # extra should not appear at all in the output.
    assert "some-bundle" not in extras
    assert extras["anthropic"] == [("langchain-anthropic", "1.0.0")]


def test_skips_known_composite_extras() -> None:
    # The build backend flattens composite extras like `all-providers`
    # into their component packages, so name-based filtering is needed to
    # avoid duplicating the full list in the output.
    mock_dist = MagicMock()
    mock_dist.requires = [
        "langchain-anthropic>=1.0.0 ; extra == 'all-providers'",
        "langchain-baseten>=1.0.0 ; extra == 'all-providers'",
        "langchain-daytona>=1.0.0 ; extra == 'all-sandboxes'",
        "langchain-anthropic>=1.0.0 ; extra == 'anthropic'",
    ]

    with (
        patch("deepagents_cli.extras_info.distribution", return_value=mock_dist),
        patch("deepagents_cli.extras_info.pkg_version", return_value="1.0.0"),
    ):
        extras = get_extras_status()

    assert "all-providers" not in extras
    assert "all-sandboxes" not in extras
    assert extras["anthropic"] == [("langchain-anthropic", "1.0.0")]


def test_format_extras_status_empty() -> None:
    assert format_extras_status({}) == ""


def test_format_extras_status_plain_empty() -> None:
    assert format_extras_status_plain({}) == ""


def test_format_extras_status_plain_columns_are_aligned() -> None:
    status = {
        "anthropic": [("langchain-anthropic", "1.4.0")],
        "google-genai": [("langchain-google-genai", "4.2.1")],
    }
    rendered = format_extras_status_plain(status)
    lines = rendered.splitlines()

    assert lines[0] == "Installed optional dependencies:"
    # Extra column widened to the longest name (`google-genai` -> 12 chars).
    assert lines[1] == "  anthropic     langchain-anthropic     1.4.0"
    assert lines[2] == "  google-genai  langchain-google-genai  4.2.1"


def test_extras_taxonomy_covers_pyproject() -> None:
    """Every declared extra must be classified as provider or sandbox.

    A new extra added to `pyproject.toml` without an entry in
    `MODEL_PROVIDER_EXTRAS` or `SANDBOX_EXTRAS` would silently fall out of
    the onboarding dependency screen. This drift test forces the contributor
    to update one of those constants alongside the dependency.
    """
    declared = _declared_extras()
    classified = MODEL_PROVIDER_EXTRAS | SANDBOX_EXTRAS

    uncategorized = declared - classified
    assert not uncategorized, (
        f"pyproject.toml declares extras not classified in extras_info: "
        f"{sorted(uncategorized)}"
    )

    stale = classified - declared
    assert not stale, (
        f"extras_info classifies extras not declared in pyproject.toml: {sorted(stale)}"
    )


def test_extras_categories_are_disjoint() -> None:
    """An extra cannot be both a model provider and a sandbox."""
    overlap = MODEL_PROVIDER_EXTRAS & SANDBOX_EXTRAS
    assert not overlap, f"Extras classified twice: {sorted(overlap)}"


def test_format_extras_status_renders_markdown_table() -> None:
    status = {
        "anthropic": [("langchain-anthropic", "1.4.0")],
        "daytona": [("langchain-daytona", "0.0.4")],
    }
    rendered = format_extras_status(status)
    lines = rendered.splitlines()

    assert lines[0] == "### Installed optional dependencies"
    assert lines[1] == ""
    assert lines[2] == "| Extra | Package | Version |"
    assert lines[3] == "| --- | --- | --- |"
    assert lines[4] == "| anthropic | langchain-anthropic | 1.4.0 |"
    assert lines[5] == "| daytona | langchain-daytona | 0.0.4 |"
