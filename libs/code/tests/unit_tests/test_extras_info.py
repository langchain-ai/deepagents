"""Tests for optional-dependency status inspection."""

import tomllib
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deepagents_code.extras_info import (
    _COMPOSITE_EXTRAS,
    KNOWN_EXTRAS,
    MODEL_PROVIDER_EXTRAS,
    SANDBOX_EXTRAS,
    STANDALONE_EXTRAS,
    _editable_sdk_source_root,
    extra_for_package,
    format_extras_status,
    format_extras_status_plain,
    format_known_extras,
    get_extras_status,
    get_optional_dependency_status,
    resolve_sdk_version,
    verify_interpreter_deps,
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
    # Composite extras like `all-providers` list `deepagents-code[...]`
    # entries; those should never surface as packages themselves.
    extras = get_extras_status()
    for pkgs in extras.values():
        for pkg_name, _version in pkgs:
            assert pkg_name.lower() != "deepagents-code"


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
        patch("deepagents_code.extras_info.distribution", return_value=mock_dist),
        patch("deepagents_code.extras_info.pkg_version", side_effect=fake_version),
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
        patch("deepagents_code.extras_info.distribution", return_value=mock_dist),
        patch("deepagents_code.extras_info.pkg_version", side_effect=fake_version),
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
        patch("deepagents_code.extras_info.distribution", return_value=mock_dist),
        patch("deepagents_code.extras_info.pkg_version", return_value="1.2.3"),
    ):
        extras = get_extras_status()

    assert extras == {"foo": [("gated-pkg", "1.2.3")]}


def test_extra_for_package_returns_declaring_known_extra() -> None:
    """Package lookup should use declared extras instead of provider-name guesses."""
    mock_dist = MagicMock()
    mock_dist.requires = [
        "langchain-google-vertexai>=3.2.3,<4.0.0 ; extra == 'vertex'",
        "deepagents-code[anthropic,baseten] ; extra == 'all-providers'",
    ]

    with patch("deepagents_code.extras_info.distribution", return_value=mock_dist):
        assert extra_for_package("langchain-google-vertexai") == "vertex"


def test_extra_for_package_returns_none_for_unknown_package() -> None:
    mock_dist = MagicMock()
    mock_dist.requires = [
        "langchain-google-vertexai>=3.2.3,<4.0.0 ; extra == 'vertex'",
    ]

    with patch("deepagents_code.extras_info.distribution", return_value=mock_dist):
        assert extra_for_package("not-declared") is None


def test_skips_composite_self_referencing_extras() -> None:
    mock_dist = MagicMock()
    mock_dist.requires = [
        "deepagents-code[anthropic,baseten] ; extra == 'some-bundle'",
        "langchain-anthropic>=1.0.0 ; extra == 'anthropic'",
    ]

    with (
        patch("deepagents_code.extras_info.distribution", return_value=mock_dist),
        patch("deepagents_code.extras_info.pkg_version", return_value="1.0.0"),
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
        "langchain-vercel-sandbox>=0.0.1 ; extra == 'all-sandboxes'",
        "langchain-anthropic>=1.0.0 ; extra == 'anthropic'",
    ]

    with (
        patch("deepagents_code.extras_info.distribution", return_value=mock_dist),
        patch("deepagents_code.extras_info.pkg_version", return_value="1.0.0"),
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
    """Every declared extra must be classified in one of the taxonomy sets.

    A new extra added to `pyproject.toml` without an entry in
    `MODEL_PROVIDER_EXTRAS`, `SANDBOX_EXTRAS`, or `STANDALONE_EXTRAS` would
    silently fall out of the onboarding dependency screen. This drift test
    forces the contributor to update one of those constants alongside the
    dependency.
    """
    declared = _declared_extras()
    classified = MODEL_PROVIDER_EXTRAS | SANDBOX_EXTRAS | STANDALONE_EXTRAS

    uncategorized = declared - classified
    assert not uncategorized, (
        f"pyproject.toml declares extras not classified in extras_info: "
        f"{sorted(uncategorized)}"
    )

    stale = classified - declared
    assert not stale, (
        f"extras_info classifies extras not declared in pyproject.toml: {sorted(stale)}"
    )


def test_known_extras_is_union_of_categories() -> None:
    """`KNOWN_EXTRAS` must be the union of the three category frozensets.

    `dcode --install <extra>` and `/install <extra>` consult `KNOWN_EXTRAS`
    to decide whether to prompt for confirmation on unknown values, so this
    set has to stay aligned with the taxonomy or callers will see spurious
    prompts for real extras.
    """
    assert KNOWN_EXTRAS == (MODEL_PROVIDER_EXTRAS | SANDBOX_EXTRAS | STANDALONE_EXTRAS)


def test_extras_categories_are_disjoint() -> None:
    """An extra can only be classified in one taxonomy set."""
    pairs = (
        ("providers/sandboxes", MODEL_PROVIDER_EXTRAS & SANDBOX_EXTRAS),
        ("providers/standalone", MODEL_PROVIDER_EXTRAS & STANDALONE_EXTRAS),
        ("sandboxes/standalone", SANDBOX_EXTRAS & STANDALONE_EXTRAS),
    )
    for label, overlap in pairs:
        assert not overlap, f"Extras classified twice in {label}: {sorted(overlap)}"


def _parse_known_extras(rendered: str) -> dict[str, list[str]]:
    """Parse `format_known_extras` output into `{label: [extras]}`.

    Lets tests assert per-line grouping and ordering rather than matching
    substrings against the whole blob, which would pass even if extras were
    rendered under the wrong category or all collapsed onto one line.
    """
    groups: dict[str, list[str]] = {}
    for line in rendered.splitlines()[1:]:  # skip the "Available extras:" header
        label, _, extras = line.strip().partition(": ")
        groups[label] = extras.split(", ")
    return groups


def test_format_known_extras_lists_exactly_known_extras() -> None:
    """The listing must contain every `KNOWN_EXTRAS` member and nothing else."""
    rendered = format_known_extras()
    assert rendered.startswith("Available extras:")
    groups = _parse_known_extras(rendered)
    rendered_extras = {extra for extras in groups.values() for extra in extras}
    # Bidirectional: catches both a new category left out of the listing and a
    # listing that drifts ahead of `KNOWN_EXTRAS`.
    assert rendered_extras == set(KNOWN_EXTRAS)


def test_format_known_extras_groups_extras_under_correct_label() -> None:
    """Each category renders under its own label with alphabetical ordering."""
    groups = _parse_known_extras(format_known_extras())
    assert groups["Model providers"] == sorted(MODEL_PROVIDER_EXTRAS)
    assert groups["Sandboxes"] == sorted(SANDBOX_EXTRAS)
    assert groups["Other"] == sorted(STANDALONE_EXTRAS)


# `verify_interpreter_deps` does a lazy `from deepagents_code.config import
# _is_editable_install` each call, so the symbol is resolved against
# `deepagents_code.config` at call time. Patch the source module — patching
# `deepagents_code.extras_info._is_editable_install` would not work (it isn't
# bound there as a module-level attribute).
def test_verify_interpreter_deps_raises_with_reinstall_hint_for_tool_install() -> None:
    with (
        patch(
            "deepagents_code.extras_info.importlib.util.find_spec", return_value=None
        ),
        patch("deepagents_code.config._is_editable_install", return_value=False),
        pytest.raises(ImportError, match="Reinstall dcode"),
    ):
        verify_interpreter_deps()


def test_verify_interpreter_deps_raises_with_uv_sync_hint_for_editable_install() -> (
    None
):
    with (
        patch(
            "deepagents_code.extras_info.importlib.util.find_spec", return_value=None
        ),
        patch("deepagents_code.config._is_editable_install", return_value=True),
        pytest.raises(ImportError, match="uv sync"),
    ):
        verify_interpreter_deps()


def test_verify_interpreter_deps_passes_when_module_present() -> None:
    fake_spec = MagicMock()
    with patch(
        "deepagents_code.extras_info.importlib.util.find_spec", return_value=fake_spec
    ):
        verify_interpreter_deps()


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


class TestResolveSdkVersion:
    """Tests for the shared `deepagents` SDK version resolver."""

    def test_resolved_returns_metadata_version_for_normal_install(self) -> None:
        """A normal install reports the package metadata version."""
        dist = MagicMock()
        dist.read_text.return_value = None
        with (
            patch(
                "deepagents_code.extras_info.pkg_version", return_value="1.2.3"
            ) as mock,
            patch("deepagents_code.extras_info.distribution", return_value=dist),
        ):
            version, status = resolve_sdk_version()
        mock.assert_called_once_with("deepagents")
        assert (version, status) == ("1.2.3", "resolved")

    def test_resolved_prefers_source_version_for_editable_install(
        self, tmp_path: Path
    ) -> None:
        """An editable SDK reports `_version.py` over stale metadata."""
        version_file = tmp_path / "deepagents" / "_version.py"
        version_file.parent.mkdir()
        version_file.write_text('__version__ = "1.2.4"\n', encoding="utf-8")
        dist = MagicMock()
        dist.read_text.return_value = (
            f'{{"url":"{tmp_path.as_uri()}","dir_info":{{"editable":true}}}}'
        )
        with (
            patch("deepagents_code.extras_info.pkg_version", return_value="1.2.3"),
            patch("deepagents_code.extras_info.distribution", return_value=dist),
        ):
            version, status = resolve_sdk_version()
        assert (version, status) == ("1.2.4", "resolved")

    def test_resolved_falls_back_to_metadata_when_editable_version_file_missing(
        self, tmp_path: Path
    ) -> None:
        """An editable SDK still reports metadata if `_version.py` is unavailable."""
        dist = MagicMock()
        dist.read_text.return_value = (
            f'{{"url":"{tmp_path.as_uri()}","dir_info":{{"editable":true}}}}'
        )
        with (
            patch("deepagents_code.extras_info.pkg_version", return_value="1.2.3"),
            patch("deepagents_code.extras_info.distribution", return_value=dist),
        ):
            version, status = resolve_sdk_version()
        assert (version, status) == ("1.2.3", "resolved")

    @pytest.mark.parametrize(
        "direct_url",
        [
            "[]",  # valid JSON, wrong top-level type
            '{"dir_info": null}',  # valid JSON, dir_info not an object
            '{"url": "file:///repo", "dir_info": {"editable": false}}',  # non-editable
            '{"url": "file:///repo", "dir_info": {}}',  # editable key absent
        ],
    )
    def test_resolved_uses_metadata_when_not_an_editable_install(
        self, direct_url: str
    ) -> None:
        """Non-editable or unexpectedly-shaped metadata never prefers source."""
        dist = MagicMock()
        dist.read_text.return_value = direct_url
        with (
            patch("deepagents_code.extras_info.pkg_version", return_value="1.2.3"),
            patch("deepagents_code.extras_info.distribution", return_value=dist),
        ):
            version, status = resolve_sdk_version()
        assert (version, status) == ("1.2.3", "resolved")

    @pytest.mark.parametrize("side_effect", [ValueError, OSError, TypeError])
    def test_resolved_uses_metadata_when_direct_url_read_fails(
        self, side_effect: type[Exception]
    ) -> None:
        """A failed/invalid `direct_url.json` read degrades to the metadata version.

        Exercises the `_editable_sdk_source_root` except arm: invalid JSON
        (`ValueError`), an unreadable metadata file (`OSError`), and a
        non-text payload (`TypeError`) must all be swallowed rather than
        crashing the resolver.
        """
        dist = MagicMock()
        dist.read_text.side_effect = side_effect("boom")
        with (
            patch("deepagents_code.extras_info.pkg_version", return_value="1.2.3"),
            patch("deepagents_code.extras_info.distribution", return_value=dist),
        ):
            version, status = resolve_sdk_version()
        assert (version, status) == ("1.2.3", "resolved")

    def test_resolved_falls_back_to_metadata_when_editable_version_file_invalid(
        self, tmp_path: Path
    ) -> None:
        """A broken editable SDK version file degrades to the metadata version."""
        version_file = tmp_path / "deepagents" / "_version.py"
        version_file.parent.mkdir()
        version_file.write_text("__version__ = ", encoding="utf-8")
        dist = MagicMock()
        dist.read_text.return_value = (
            f'{{"url":"{tmp_path.as_uri()}","dir_info":{{"editable":true}}}}'
        )
        with (
            patch("deepagents_code.extras_info.pkg_version", return_value="1.2.3"),
            patch("deepagents_code.extras_info.distribution", return_value=dist),
        ):
            version, status = resolve_sdk_version()
        assert (version, status) == ("1.2.3", "resolved")

    @pytest.mark.parametrize("source_version", ["", None, 123])
    def test_resolved_falls_back_to_metadata_when_source_version_unusable(
        self, tmp_path: Path, source_version: object
    ) -> None:
        """An empty or non-string source `__version__` is rejected for metadata."""
        version_file = tmp_path / "deepagents" / "_version.py"
        version_file.parent.mkdir()
        version_file.write_text(f"__version__ = {source_version!r}\n", encoding="utf-8")
        dist = MagicMock()
        dist.read_text.return_value = (
            f'{{"url":"{tmp_path.as_uri()}","dir_info":{{"editable":true}}}}'
        )
        with (
            patch("deepagents_code.extras_info.pkg_version", return_value="1.2.3"),
            patch("deepagents_code.extras_info.distribution", return_value=dist),
        ):
            version, status = resolve_sdk_version()
        assert (version, status) == ("1.2.3", "resolved")

    def test_resolved_uses_metadata_for_editable_non_file_url(self) -> None:
        """An editable install with a non-`file` source URL prefers metadata."""
        dist = MagicMock()
        dist.read_text.return_value = (
            '{"url":"https://example.com/repo","dir_info":{"editable":true}}'
        )
        with (
            patch("deepagents_code.extras_info.pkg_version", return_value="1.2.3"),
            patch("deepagents_code.extras_info.distribution", return_value=dist),
        ):
            version, status = resolve_sdk_version()
        assert (version, status) == ("1.2.3", "resolved")

    @pytest.mark.parametrize(
        "direct_url",
        [
            '{"dir_info":{"editable":true}}',  # url key absent
            '{"url":123,"dir_info":{"editable":true}}',  # url not a string
        ],
    )
    def test_resolved_uses_metadata_when_editable_url_unusable(
        self, direct_url: str
    ) -> None:
        """An editable install without a usable source URL prefers metadata."""
        dist = MagicMock()
        dist.read_text.return_value = direct_url
        with (
            patch("deepagents_code.extras_info.pkg_version", return_value="1.2.3"),
            patch("deepagents_code.extras_info.distribution", return_value=dist),
        ):
            version, status = resolve_sdk_version()
        assert (version, status) == ("1.2.3", "resolved")

    def test_resolved_falls_back_to_metadata_when_version_assignment_absent(
        self, tmp_path: Path
    ) -> None:
        """A valid `_version.py` with no `__version__` assignment uses metadata."""
        version_file = tmp_path / "deepagents" / "_version.py"
        version_file.parent.mkdir()
        version_file.write_text('VERSION = "1.2.4"\n', encoding="utf-8")
        dist = MagicMock()
        dist.read_text.return_value = (
            f'{{"url":"{tmp_path.as_uri()}","dir_info":{{"editable":true}}}}'
        )
        with (
            patch("deepagents_code.extras_info.pkg_version", return_value="1.2.3"),
            patch("deepagents_code.extras_info.distribution", return_value=dist),
        ):
            version, status = resolve_sdk_version()
        assert (version, status) == ("1.2.3", "resolved")

    def test_resolved_falls_back_to_metadata_when_version_is_non_literal(
        self, tmp_path: Path
    ) -> None:
        """A non-literal `__version__` RHS is rejected in favor of metadata.

        Exercises the `ast.literal_eval` except arm — distinct from a
        `SyntaxError` at parse time — where a syntactically valid but
        dynamically-computed assignment cannot be read as a constant.
        """
        version_file = tmp_path / "deepagents" / "_version.py"
        version_file.parent.mkdir()
        version_file.write_text("__version__ = _compute_version()\n", encoding="utf-8")
        dist = MagicMock()
        dist.read_text.return_value = (
            f'{{"url":"{tmp_path.as_uri()}","dir_info":{{"editable":true}}}}'
        )
        with (
            patch("deepagents_code.extras_info.pkg_version", return_value="1.2.3"),
            patch("deepagents_code.extras_info.distribution", return_value=dist),
        ):
            version, status = resolve_sdk_version()
        assert (version, status) == ("1.2.3", "resolved")

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            # UNC authority is folded back into the path...
            ("file://server/share/proj", Path("//server/share/proj")),
            # ...but the conventional `localhost` authority is dropped.
            ("file://localhost/repo", Path("/repo")),
        ],
    )
    def test_editable_source_root_handles_url_authority(
        self, url: str, expected: Path
    ) -> None:
        """`file://` authority handling distinguishes UNC hosts from `localhost`."""
        dist = MagicMock()
        dist.read_text.return_value = (
            f'{{"url":"{url}","dir_info":{{"editable":true}}}}'
        )
        with patch("deepagents_code.extras_info.distribution", return_value=dist):
            assert _editable_sdk_source_root() == expected

    def test_not_installed_distinguished_from_error(self) -> None:
        """A missing package reports `not_installed`, never `error`."""
        with patch(
            "deepagents_code.extras_info.pkg_version",
            side_effect=PackageNotFoundError("deepagents"),
        ):
            version, status = resolve_sdk_version()
        assert (version, status) == (None, "not_installed")

    def test_unexpected_error_reports_error_status(self) -> None:
        """Any non-`PackageNotFoundError` failure reports `error`, not a crash."""
        with patch(
            "deepagents_code.extras_info.pkg_version",
            side_effect=RuntimeError("corrupt metadata"),
        ):
            version, status = resolve_sdk_version()
        assert (version, status) == (None, "error")
