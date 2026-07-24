"""Tests for optional-dependency status inspection."""

import tomllib
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from packaging.requirements import Requirement

from deepagents_code.extras_info import (
    _COMPOSITE_EXTRAS,
    KNOWN_EXTRAS,
    MODEL_PROVIDER_EXTRAS,
    SANDBOX_EXTRAS,
    STANDALONE_EXTRAS,
    DistributionMetadataStatus,
    DistributionVersion,
    VersionReport,
    _display_sdk_version,
    _editable_sdk_is_cli_workspace_sibling,
    _editable_sdk_source_root,
    _requirement_satisfied,
    _resolve_source_path,
    _sdk_requirement_comparison_version,
    collect_cli_version_info,
    collect_sdk_version_info,
    collect_version_report,
    extra_for_package,
    format_cli_version_annotation,
    format_extras_status,
    format_extras_status_plain,
    format_known_extras,
    format_sdk_version_annotation,
    get_extras_status,
    get_optional_dependency_status,
    resolve_sdk_version,
    sdk_requirement_from_cli,
    verify_interpreter_deps,
)

_PYPROJECT_PATH = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _write_cli_pyproject(root: Path, requirement: str) -> None:
    """Write the minimal editable dcode project metadata used by version tests."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text(
        f'[project]\nname = "deepagents-code"\ndependencies = ["{requirement}"]\n',
        encoding="utf-8",
    )


def _optional_dependencies() -> dict[str, list[str]]:
    """Return optional dependencies declared in `pyproject.toml`."""
    data = tomllib.loads(_PYPROJECT_PATH.read_text(encoding="utf-8"))
    return data["project"]["optional-dependencies"]


def _declared_extras() -> frozenset[str]:
    """Return non-composite extras declared in `pyproject.toml`."""
    return frozenset(_optional_dependencies()) - _COMPOSITE_EXTRAS


def test_nvidia_extra_requires_aiohttp_safe_ai_endpoints_release() -> None:
    """The NVIDIA extra must require an aiohttp-safe ai-endpoints release."""
    assert _optional_dependencies()["nvidia"] == [
        "aiohttp>=3.14.3,<3.15.0",
        "langchain-nvidia-ai-endpoints>=1.4.3,<2.0.0",
    ]


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
        # Assert the SDK lookup happened without coupling to call ordering:
        # `resolve_sdk_version` also looks up `deepagents-code` for the CLI.
        assert ("deepagents",) in [call.args for call in mock.call_args_list]
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

    def test_resolved_uses_newer_exact_requirement_for_editable_install(
        self, tmp_path: Path
    ) -> None:
        """A newer exact dcode pin is effective for sibling monorepo packages."""
        libs = tmp_path / "repo" / "libs"
        cli_path = libs / "code"
        sdk_path = libs / "deepagents"
        version_file = sdk_path / "deepagents" / "_version.py"
        version_file.parent.mkdir(parents=True)
        _write_cli_pyproject(cli_path, "deepagents==0.7.0a8")
        version_file.write_text('__version__ = "0.6.12"\n', encoding="utf-8")
        sdk_dist = MagicMock()
        sdk_dist.read_text.return_value = (
            f'{{"url":"{sdk_path.as_uri()}","dir_info":{{"editable":true}}}}'
        )
        code_dist = MagicMock()
        code_dist.requires = ["deepagents==0.7.0a7"]

        def version(name: str) -> str:
            return {"deepagents": "0.6.12", "deepagents-code": "0.1.45"}[name]

        def dist(name: str) -> MagicMock:
            return {"deepagents": sdk_dist, "deepagents-code": code_dist}[name]

        with (
            patch("deepagents_code.extras_info.pkg_version", side_effect=version),
            patch("deepagents_code.extras_info.distribution", side_effect=dist),
            patch(
                "deepagents_code.extras_info._cli_editable_info",
                return_value=(True, str(cli_path)),
            ),
        ):
            version_value, status = resolve_sdk_version()
        assert (version_value, status) == ("0.7.0a8+editable", "resolved")

    def test_resolved_keeps_source_for_unrelated_editable_sdk(
        self, tmp_path: Path
    ) -> None:
        """A newer exact dcode pin does not mask an unrelated editable SDK."""
        sdk_path = tmp_path / "old-sdk"
        version_file = sdk_path / "deepagents" / "_version.py"
        version_file.parent.mkdir(parents=True)
        version_file.write_text('__version__ = "0.6.12"\n', encoding="utf-8")
        sdk_dist = MagicMock()
        sdk_dist.read_text.return_value = (
            f'{{"url":"{sdk_path.as_uri()}","dir_info":{{"editable":true}}}}'
        )
        code_dist = MagicMock()
        code_dist.requires = ["deepagents==0.7.0a8"]

        def version(name: str) -> str:
            return {"deepagents": "0.6.12", "deepagents-code": "0.1.45"}[name]

        def dist(name: str) -> MagicMock:
            return {"deepagents": sdk_dist, "deepagents-code": code_dist}[name]

        with (
            patch("deepagents_code.extras_info.pkg_version", side_effect=version),
            patch("deepagents_code.extras_info.distribution", side_effect=dist),
            patch(
                "deepagents_code.extras_info._cli_editable_info",
                return_value=(True, str(tmp_path / "repo" / "libs" / "code")),
            ),
        ):
            version_value, status = resolve_sdk_version()
        assert (version_value, status) == ("0.6.12", "resolved")

    def test_resolve_source_path_returns_none_for_unusable_path(self) -> None:
        """Malformed editable paths should not crash version diagnostics."""
        with patch.object(Path, "resolve", side_effect=ValueError("nul byte")):
            assert _resolve_source_path("/repo/%00/libs/code") is None

    def test_windows_editable_paths_are_workspace_siblings(self) -> None:
        """PEP 610 drive paths compare equal across URL and Windows forms."""
        cli = _distribution_version(
            name="deepagents-code",
            editable=True,
            source_path="/C:/repo/libs/code",
        )
        sdk = _distribution_version(
            editable=True,
            source_path=r"C:\repo\libs\deepagents",
        )
        assert _editable_sdk_is_cli_workspace_sibling(cli, sdk) is True

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


def _distribution_version(
    *,
    name: str = "deepagents",
    source_version: str | None = None,
    metadata_version: str | None = None,
    editable: bool = False,
    source_path: str | None = None,
    status: DistributionMetadataStatus = "resolved",
) -> DistributionVersion:
    """Build a `DistributionVersion` with sensible defaults for tests."""
    return DistributionVersion(
        name=name,
        source_version=source_version,
        metadata_version=metadata_version,
        editable=editable,
        source_path=source_path,
        status=status,
    )


class TestContractHome:
    """Tests for `~`-contraction of displayed editable source paths."""

    def test_contracts_home_prefix(self) -> None:
        """A path under the home directory is shown with a `~` prefix."""
        from deepagents_code.extras_info import _contract_home

        with patch(
            "deepagents_code.extras_info.Path.home", return_value=Path("/home/dev")
        ):
            assert _contract_home(Path("/home/dev/src/proj")) == "~/src/proj"

    def test_exact_home_becomes_tilde(self) -> None:
        """The home directory itself contracts to a bare `~`."""
        from deepagents_code.extras_info import _contract_home

        with patch(
            "deepagents_code.extras_info.Path.home", return_value=Path("/home/dev")
        ):
            assert _contract_home(Path("/home/dev")) == "~"

    def test_non_home_path_is_unchanged(self) -> None:
        """A path outside the home directory is returned verbatim."""
        from deepagents_code.extras_info import _contract_home

        with patch(
            "deepagents_code.extras_info.Path.home", return_value=Path("/home/dev")
        ):
            assert _contract_home(Path("/opt/other")) == "/opt/other"

    def test_unresolvable_home_returns_raw_path(self) -> None:
        """When the home directory cannot be determined, the raw path is kept."""
        from deepagents_code.extras_info import _contract_home

        with patch("deepagents_code.extras_info.Path.home", side_effect=RuntimeError):
            assert _contract_home(Path("/home/dev/x")) == "/home/dev/x"


class TestDistributionVersion:
    """Tests for the structured single-distribution version representation."""

    def test_primary_prefers_source_for_editable(self) -> None:
        """Editable installs report the live source version as primary."""
        info = _distribution_version(
            source_version="1.2.4", metadata_version="1.2.3", editable=True
        )
        assert info.primary_version == "1.2.4"

    def test_primary_uses_metadata_for_non_editable(self) -> None:
        """Non-editable installs report the metadata version as primary."""
        info = _distribution_version(
            source_version="9.9.9", metadata_version="1.2.3", editable=False
        )
        assert info.primary_version == "1.2.3"

    def test_primary_falls_back_to_metadata_without_source(self) -> None:
        """An editable install with no readable source version uses metadata."""
        info = _distribution_version(
            source_version=None, metadata_version="1.2.3", editable=True
        )
        assert info.primary_version == "1.2.3"

    def test_has_drift_true_when_versions_differ(self) -> None:
        """Drift is reported when both versions are known and differ."""
        info = _distribution_version(source_version="1.2.4", metadata_version="1.2.3")
        assert info.has_drift is True

    def test_has_drift_false_when_versions_agree(self) -> None:
        """No drift is reported when the versions agree."""
        info = _distribution_version(source_version="1.2.3", metadata_version="1.2.3")
        assert info.has_drift is False

    @pytest.mark.parametrize(
        ("source", "metadata"),
        [(None, "1.2.3"), ("1.2.3", None), (None, None)],
    )
    def test_has_drift_false_when_a_version_is_unknown(
        self, source: str | None, metadata: str | None
    ) -> None:
        """Drift requires both versions; a missing one is never drift."""
        info = _distribution_version(source_version=source, metadata_version=metadata)
        assert info.has_drift is False


class TestVersionAnnotations:
    """Tests for the rendered editable/drift/mismatch annotations."""

    def _report(
        self,
        *,
        cli: DistributionVersion | None = None,
        sdk: DistributionVersion | None = None,
        requirement: Requirement | None = None,
        satisfied: bool | None = None,
    ) -> VersionReport:
        return VersionReport(
            cli=cli or _distribution_version(name="deepagents-code"),
            sdk=sdk or _distribution_version(),
            sdk_requirement=requirement,
            sdk_requirement_satisfied=satisfied,
        )

    def test_cli_annotation_empty_for_normal_install(self) -> None:
        """A non-editable install with agreeing versions gets no annotation."""
        info = _distribution_version(
            name="deepagents-code", source_version="0.1.41", metadata_version="0.1.41"
        )
        assert format_cli_version_annotation(info) == ""

    def test_cli_annotation_omits_editable_when_versions_agree(self) -> None:
        """Editable status is shown separately, so an in-sync editable CLI is bare.

        The CLI editable state is rendered by the dedicated `Editable install:`
        line (and doctor's `Install method`), so the inline annotation carries
        only drift — here there is none.
        """
        info = _distribution_version(
            name="deepagents-code",
            source_version="0.1.41",
            metadata_version="0.1.41",
            editable=True,
        )
        assert format_cli_version_annotation(info) == ""

    def test_cli_annotation_shows_stale_metadata(self) -> None:
        """An editable CLI with stale metadata shows the drift, not `editable`."""
        info = _distribution_version(
            name="deepagents-code",
            source_version="0.1.41",
            metadata_version="0.1.40",
            editable=True,
        )
        assert format_cli_version_annotation(info) == " (installed metadata: 0.1.40)"

    def test_sdk_annotation_empty_for_normal_install(self) -> None:
        """A non-editable SDK whose requirement is satisfied gets no annotation."""
        sdk = _distribution_version(source_version="0.7.0", metadata_version="0.7.0")
        report = self._report(
            sdk=sdk, requirement=Requirement("deepagents==0.7.0"), satisfied=True
        )
        assert format_sdk_version_annotation(report) == ""

    def test_sdk_annotation_shows_stale_metadata_without_mismatch(self) -> None:
        """An editable SDK with stale metadata but a satisfied pin shows drift."""
        sdk = _distribution_version(
            source_version="0.7.1", metadata_version="0.7.0", editable=True
        )
        report = self._report(
            sdk=sdk, requirement=Requirement("deepagents>=0.7.0"), satisfied=True
        )
        assert (
            format_sdk_version_annotation(report)
            == " (editable; installed metadata: 0.7.0)"
        )

    def test_sdk_annotation_uses_newer_exact_pin_for_editable_install(self) -> None:
        """A newer exact pin explains why the effective SDK version is ahead."""
        cli = _distribution_version(
            name="deepagents-code", editable=True, source_path="/repo/libs/code"
        )
        sdk = _distribution_version(
            source_version="0.6.12",
            metadata_version="0.6.12",
            editable=True,
            source_path="/repo/libs/deepagents",
        )
        report = self._report(
            cli=cli,
            sdk=sdk,
            requirement=Requirement("deepagents==0.7.0a7"),
            satisfied=True,
        )
        assert report.effective_sdk_version == "0.7.0a7"
        assert report.display_sdk_version == "0.7.0a7+editable"
        assert format_sdk_version_annotation(report) == (
            " (workspace HEAD; source marker: 0.6.12)"
        )

    def test_workspace_head_decorates_display_not_comparison_version(self) -> None:
        """The editable marker never changes the requirement comparison value."""
        cli = _distribution_version(
            name="deepagents-code", editable=True, source_path="/repo/libs/code"
        )
        sdk = _distribution_version(
            source_version="0.6.12",
            metadata_version="0.6.12",
            editable=True,
            source_path="/repo/libs/deepagents",
        )
        requirement = Requirement("deepagents==0.7.0a7+build")
        report = self._report(
            cli=cli,
            sdk=sdk,
            requirement=requirement,
            satisfied=True,
        )
        comparison = _sdk_requirement_comparison_version(cli, sdk, requirement)
        assert comparison == "0.7.0a7+build"
        assert _requirement_satisfied(requirement, comparison) is True
        assert _display_sdk_version(cli, sdk, requirement) == ("0.7.0a7+build.editable")
        assert report.display_sdk_version == "0.7.0a7+build.editable"

    @pytest.mark.parametrize("source_version", [None, "not-a-version"])
    def test_workspace_head_requires_valid_source_version(
        self, source_version: str | None
    ) -> None:
        """Stale metadata cannot mask a missing or malformed source marker."""
        cli = _distribution_version(
            name="deepagents-code", editable=True, source_path="/repo/libs/code"
        )
        sdk = _distribution_version(
            source_version=source_version,
            metadata_version="0.6.12",
            editable=True,
            source_path="/repo/libs/deepagents",
        )
        report = self._report(
            cli=cli,
            sdk=sdk,
            requirement=Requirement("deepagents==0.7.0a7"),
            satisfied=False,
        )
        assert report.sdk_is_workspace_head is False
        expected = source_version or "0.6.12"
        assert report.effective_sdk_version == expected
        assert report.sdk_source_version_invalid is True
        assert report.sdk_requirement_mismatch is True

    @pytest.mark.parametrize(
        ("pin", "satisfied", "mismatch"),
        [
            pytest.param("0.6.10", False, True, id="older-pin"),
            pytest.param("0.6.12", True, False, id="equal-pin"),
        ],
    )
    def test_workspace_head_requires_pin_newer_than_marker(
        self, pin: str, satisfied: bool, mismatch: bool
    ) -> None:
        """A sibling pin that is not strictly newer never overrides the marker.

        Guards the `pinned > max(markers)` comparison: an older pin must still
        surface as a mismatch, and an equal pin must not decorate the version as
        workspace HEAD.
        """
        cli = _distribution_version(
            name="deepagents-code", editable=True, source_path="/repo/libs/code"
        )
        sdk = _distribution_version(
            source_version="0.6.12",
            metadata_version="0.6.12",
            editable=True,
            source_path="/repo/libs/deepagents",
        )
        report = self._report(
            cli=cli,
            sdk=sdk,
            requirement=Requirement(f"deepagents=={pin}"),
            satisfied=satisfied,
        )
        # The observed source marker wins; the pin neither masks it nor adds the
        # `+editable` workspace-HEAD decoration.
        assert report.sdk_is_workspace_head is False
        assert report.effective_sdk_version == "0.6.12"
        assert report.display_sdk_version == "0.6.12"
        assert report.sdk_requirement_mismatch is mismatch

    def test_ranged_requirement_never_overrides_sibling_marker(self) -> None:
        """A ranged sibling requirement stays a mismatch instead of overriding.

        `_exact_pin` returns `None` for a range, so the workspace-HEAD override
        must decline and the marker must still read as a mismatch.
        """
        cli = _distribution_version(
            name="deepagents-code", editable=True, source_path="/repo/libs/code"
        )
        sdk = _distribution_version(
            source_version="0.6.12",
            metadata_version="0.6.12",
            editable=True,
            source_path="/repo/libs/deepagents",
        )
        report = self._report(
            cli=cli,
            sdk=sdk,
            requirement=Requirement("deepagents>=0.7,<0.8"),
            satisfied=False,
        )
        assert report.sdk_is_workspace_head is False
        assert report.effective_sdk_version == "0.6.12"
        assert report.display_sdk_version == "0.6.12"
        assert (
            "required by deepagents-code: <0.8,>=0.7 — mismatch"
            in format_sdk_version_annotation(report)
        )

    @pytest.mark.parametrize(
        ("cli_editable", "sdk_editable"),
        [
            pytest.param(True, False, id="sdk-not-editable"),
            pytest.param(False, True, id="cli-not-editable"),
        ],
    )
    def test_workspace_sibling_requires_both_editable(
        self, cli_editable: bool, sdk_editable: bool
    ) -> None:
        """The sibling shape only holds when both installs are editable."""
        cli = _distribution_version(
            name="deepagents-code",
            editable=cli_editable,
            source_path="/repo/libs/code",
        )
        sdk = _distribution_version(
            editable=sdk_editable,
            source_path="/repo/libs/deepagents",
        )
        assert _editable_sdk_is_cli_workspace_sibling(cli, sdk) is False

    def test_sdk_annotation_shows_ranged_requirement_mismatch(self) -> None:
        """A ranged requirement keeps its full specifier in the mismatch note."""
        sdk = _distribution_version(source_version="0.6.12", metadata_version="0.6.12")
        report = self._report(
            sdk=sdk,
            requirement=Requirement("deepagents>=0.7,<0.8"),
            satisfied=False,
        )
        annotation = format_sdk_version_annotation(report)
        assert "required by deepagents-code: <0.8,>=0.7 — mismatch" in annotation

    def test_sdk_annotation_shows_editable_drift_without_mismatch(self) -> None:
        """An editable source matching the exact pin is healthy.

        Stale metadata is still shown for diagnostics.
        """
        sdk = _distribution_version(
            source_version="0.7.0a7", metadata_version="0.6.12", editable=True
        )
        report = self._report(
            sdk=sdk, requirement=Requirement("deepagents==0.7.0a7"), satisfied=True
        )
        assert report.effective_sdk_version == "0.7.0a7"
        assert format_sdk_version_annotation(report) == (
            " (editable; installed metadata: 0.6.12)"
        )


class TestRequirementSatisfied:
    """Tests for comparing a declared requirement against installed metadata."""

    def test_exact_pin_satisfied(self) -> None:
        assert _requirement_satisfied(Requirement("deepagents==0.7.0"), "0.7.0") is True

    def test_exact_pin_unsatisfied(self) -> None:
        assert (
            _requirement_satisfied(Requirement("deepagents==0.7.0a7"), "0.6.12")
            is False
        )

    def test_ranged_requirement_satisfied(self) -> None:
        assert (
            _requirement_satisfied(Requirement("deepagents>=0.7,<0.8"), "0.7.3") is True
        )

    def test_ranged_requirement_unsatisfied(self) -> None:
        assert (
            _requirement_satisfied(Requirement("deepagents>=0.7,<0.8"), "0.6.12")
            is False
        )

    def test_prerelease_metadata_is_evaluated(self) -> None:
        """A prerelease installed version is compared, not silently dropped."""
        assert (
            _requirement_satisfied(Requirement("deepagents==0.7.0a7"), "0.7.0a7")
            is True
        )

    @pytest.mark.parametrize(
        ("requirement", "metadata"),
        [(None, "0.7.0"), (Requirement("deepagents==0.7.0"), None), (None, None)],
    )
    def test_missing_inputs_are_inconclusive(
        self, requirement: Requirement | None, metadata: str | None
    ) -> None:
        """A missing requirement or version yields `None`, not a boolean."""
        assert _requirement_satisfied(requirement, metadata) is None

    def test_unparseable_version_is_inconclusive(self) -> None:
        """An unparseable installed version degrades to `None` rather than raising."""
        assert (
            _requirement_satisfied(Requirement("deepagents==0.7.0"), "not-a-version")
            is None
        )


class TestSdkRequirementFromCli:
    """Tests for reading the declared `deepagents` requirement from metadata."""

    def _dist(self, requires: list[str]) -> MagicMock:
        dist = MagicMock()
        dist.requires = requires
        return dist

    def test_returns_base_requirement(self) -> None:
        """The unconditional `deepagents` dependency is returned."""
        dist = self._dist(["deepagents==0.7.0a7", "rich>=13"])
        with patch("deepagents_code.extras_info.distribution", return_value=dist):
            req = sdk_requirement_from_cli()
        assert req is not None
        assert str(req) == "deepagents==0.7.0a7"

    def test_skips_extras_gated_requirement(self) -> None:
        """An `extra`-gated `deepagents` entry is not treated as the base pin."""
        dist = self._dist(['deepagents[extra]==9.9.9; extra == "foo"'])
        with patch("deepagents_code.extras_info.distribution", return_value=dist):
            assert sdk_requirement_from_cli() is None

    def test_skips_unparseable_entries(self) -> None:
        """A malformed `Requires-Dist` entry is skipped, not fatal."""
        dist = self._dist(["=not a requirement=", "deepagents>=0.7,<0.8"])
        with patch("deepagents_code.extras_info.distribution", return_value=dist):
            req = sdk_requirement_from_cli()
        assert req is not None
        assert str(req.specifier) == "<0.8,>=0.7"

    def test_missing_distribution_returns_none(self) -> None:
        """A missing distribution yields `None` rather than raising."""
        with patch(
            "deepagents_code.extras_info.distribution",
            side_effect=PackageNotFoundError("deepagents-code"),
        ):
            assert sdk_requirement_from_cli() is None

    @pytest.mark.parametrize(
        "error",
        [
            UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"),
            TypeError("malformed metadata"),
        ],
    )
    def test_unreadable_requirements_return_none(self, error: Exception) -> None:
        """Unreadable requirement metadata is inconclusive rather than fatal."""
        dist = MagicMock()
        type(dist).requires = PropertyMock(side_effect=error)
        with patch("deepagents_code.extras_info.distribution", return_value=dist):
            assert sdk_requirement_from_cli() is None

    def test_returns_applicable_marked_requirement(self) -> None:
        """A base `deepagents` dep whose marker applies is still returned."""
        dist = self._dist(['deepagents==0.7.0a7; python_version >= "3.0"'])
        with patch("deepagents_code.extras_info.distribution", return_value=dist):
            req = sdk_requirement_from_cli()
        assert req is not None
        assert str(req.specifier) == "==0.7.0a7"

    def test_unevaluable_marker_is_treated_as_applicable(self) -> None:
        """A marker that fails to evaluate must not silently drop the pin.

        Dropping it would hide the SDK requirement mismatch this check exists to
        surface, so an unexpected evaluation error is treated as applicable.
        """
        dist = self._dist(['deepagents==0.7.0a7; python_version >= "3.0"'])

        def boom(_self: object, *_a: object, **_k: object) -> bool:
            msg = "marker blew up"
            raise ValueError(msg)

        with (
            patch("deepagents_code.extras_info.distribution", return_value=dist),
            patch("packaging.markers.Marker.evaluate", boom),
        ):
            req = sdk_requirement_from_cli()
        assert req is not None
        assert str(req.specifier) == "==0.7.0a7"


class TestCollectVersionInfo:
    """Tests for the version collectors that read the live environment."""

    def test_collect_cli_normal_install(self) -> None:
        """A non-editable CLI reports source and metadata with no editable flag."""
        with (
            patch(
                "deepagents_code.extras_info._read_cli_source_version",
                return_value="0.1.41",
            ),
            patch("deepagents_code.extras_info.pkg_version", return_value="0.1.41"),
            patch(
                "deepagents_code.extras_info._cli_editable_info",
                return_value=(False, None),
            ),
        ):
            info = collect_cli_version_info()
        assert info.source_version == "0.1.41"
        assert info.metadata_version == "0.1.41"
        assert info.editable is False
        assert info.status == "resolved"
        assert info.has_drift is False

    def test_collect_cli_editable_stale_metadata(self) -> None:
        """An editable CLI surfaces the stale-metadata drift."""
        with (
            patch(
                "deepagents_code.extras_info._read_cli_source_version",
                return_value="0.1.41",
            ),
            patch("deepagents_code.extras_info.pkg_version", return_value="0.1.40"),
            patch(
                "deepagents_code.extras_info._cli_editable_info",
                return_value=(True, "~/src/deepagents/libs/code"),
            ),
        ):
            info = collect_cli_version_info()
        assert info.primary_version == "0.1.41"
        assert info.metadata_version == "0.1.40"
        assert info.editable is True
        assert info.source_path == "~/src/deepagents/libs/code"
        assert info.has_drift is True

    def test_collect_cli_metadata_missing_resolves_from_source(self) -> None:
        """Absent CLI metadata still resolves via the source version."""
        with (
            patch(
                "deepagents_code.extras_info._read_cli_source_version",
                return_value="0.1.41",
            ),
            patch(
                "deepagents_code.extras_info.pkg_version",
                side_effect=PackageNotFoundError("deepagents-code"),
            ),
            patch(
                "deepagents_code.extras_info._cli_editable_info",
                return_value=(False, None),
            ),
        ):
            info = collect_cli_version_info()
        assert info.status == "resolved"
        assert info.metadata_version is None
        assert info.primary_version == "0.1.41"

    def test_collect_cli_metadata_and_source_missing_is_not_installed(self) -> None:
        """No source and no metadata reports `not_installed`."""
        with (
            patch(
                "deepagents_code.extras_info._read_cli_source_version",
                return_value=None,
            ),
            patch(
                "deepagents_code.extras_info.pkg_version",
                side_effect=PackageNotFoundError("deepagents-code"),
            ),
            patch(
                "deepagents_code.extras_info._cli_editable_info",
                return_value=(False, None),
            ),
        ):
            info = collect_cli_version_info()
        assert info.status == "not_installed"
        assert info.primary_version is None

    def test_collect_cli_unexpected_error_with_source_resolves(self) -> None:
        """An unexpected metadata failure still resolves when source is present."""
        with (
            patch(
                "deepagents_code.extras_info._read_cli_source_version",
                return_value="0.1.41",
            ),
            patch(
                "deepagents_code.extras_info.pkg_version",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "deepagents_code.extras_info._cli_editable_info",
                return_value=(False, None),
            ),
        ):
            info = collect_cli_version_info()
        assert info.status == "resolved"
        assert info.primary_version == "0.1.41"

    def test_collect_cli_unexpected_error_without_source_is_error(self) -> None:
        """An unexpected metadata failure with no source reports `error`."""
        with (
            patch(
                "deepagents_code.extras_info._read_cli_source_version",
                return_value=None,
            ),
            patch(
                "deepagents_code.extras_info.pkg_version",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                "deepagents_code.extras_info._cli_editable_info",
                return_value=(False, None),
            ),
        ):
            info = collect_cli_version_info()
        assert info.status == "error"
        assert info.primary_version is None

    def test_collect_sdk_normal_install(self) -> None:
        """A non-editable SDK reports metadata and no editable source."""
        dist = MagicMock()
        dist.read_text.return_value = None
        with (
            patch("deepagents_code.extras_info.pkg_version", return_value="0.7.0"),
            patch("deepagents_code.extras_info.distribution", return_value=dist),
        ):
            info = collect_sdk_version_info()
        assert info.metadata_version == "0.7.0"
        assert info.editable is False
        assert info.source_version is None
        assert info.primary_version == "0.7.0"
        assert info.status == "resolved"

    def test_collect_sdk_editable_stale_metadata(self, tmp_path: Path) -> None:
        """An editable SDK prefers source over stale metadata and reports drift."""
        version_file = tmp_path / "deepagents" / "_version.py"
        version_file.parent.mkdir()
        version_file.write_text('__version__ = "0.6.13"\n', encoding="utf-8")
        dist = MagicMock()
        dist.read_text.return_value = (
            f'{{"url":"{tmp_path.as_uri()}","dir_info":{{"editable":true}}}}'
        )
        with (
            patch("deepagents_code.extras_info.pkg_version", return_value="0.6.12"),
            patch("deepagents_code.extras_info.distribution", return_value=dist),
        ):
            info = collect_sdk_version_info()
        assert info.editable is True
        assert info.source_version == "0.6.13"
        assert info.metadata_version == "0.6.12"
        assert info.primary_version == "0.6.13"
        assert info.has_drift is True

    def test_collect_sdk_not_installed(self) -> None:
        """A missing SDK reports `not_installed` with no versions."""
        with patch(
            "deepagents_code.extras_info.pkg_version",
            side_effect=PackageNotFoundError("deepagents"),
        ):
            info = collect_sdk_version_info()
        assert info.status == "not_installed"
        assert info.metadata_version is None
        assert info.primary_version is None

    def test_collect_version_report_uses_newer_exact_pin_for_editable_sdk(
        self, tmp_path: Path
    ) -> None:
        """Editable main treats a newer exact dcode pin as the effective SDK."""
        libs = tmp_path / "repo" / "libs"
        cli_path = libs / "code"
        sdk_path = libs / "deepagents"
        version_file = sdk_path / "deepagents" / "_version.py"
        version_file.parent.mkdir(parents=True)
        _write_cli_pyproject(cli_path, "deepagents==0.7.0a8")
        version_file.write_text('__version__ = "0.6.12"\n', encoding="utf-8")
        sdk_dist = MagicMock()
        sdk_dist.read_text.return_value = (
            f'{{"url":"{sdk_path.as_uri()}","dir_info":{{"editable":true}}}}'
        )
        code_dist = MagicMock()
        code_dist.read_text.return_value = None
        code_dist.requires = ["deepagents==0.7.0a7"]

        def fake_version(name: str) -> str:
            return {"deepagents": "0.6.12", "deepagents-code": "0.1.45"}[name]

        def fake_dist(name: str) -> MagicMock:
            return {"deepagents": sdk_dist, "deepagents-code": code_dist}[name]

        with (
            patch("deepagents_code.extras_info.pkg_version", side_effect=fake_version),
            patch("deepagents_code.extras_info.distribution", side_effect=fake_dist),
            patch(
                "deepagents_code.extras_info._read_cli_source_version",
                return_value="0.1.45",
            ),
            patch(
                "deepagents_code.extras_info._cli_editable_info",
                return_value=(True, str(cli_path)),
            ),
        ):
            report = collect_version_report()
        assert report.effective_sdk_version == "0.7.0a8"
        assert report.display_sdk_version == "0.7.0a8+editable"
        assert report.sdk_is_workspace_head is True
        assert report.sdk_requirement_satisfied is True
        assert report.sdk_requirement_mismatch is False

    def test_collect_version_report_keeps_unrelated_editable_sdk_mismatch(
        self, tmp_path: Path
    ) -> None:
        """Editable SDKs outside the dcode checkout still report mismatches."""
        cli_path = tmp_path / "repo" / "libs" / "code"
        sdk_path = tmp_path / "old-sdk"
        version_file = sdk_path / "deepagents" / "_version.py"
        version_file.parent.mkdir(parents=True)
        _write_cli_pyproject(cli_path, "deepagents==0.7.0a8")
        version_file.write_text('__version__ = "0.6.12"\n', encoding="utf-8")
        sdk_dist = MagicMock()
        sdk_dist.read_text.return_value = (
            f'{{"url":"{sdk_path.as_uri()}","dir_info":{{"editable":true}}}}'
        )
        code_dist = MagicMock()
        code_dist.read_text.return_value = None
        code_dist.requires = ["deepagents==0.7.0a8"]

        def fake_version(name: str) -> str:
            return {"deepagents": "0.6.12", "deepagents-code": "0.1.45"}[name]

        def fake_dist(name: str) -> MagicMock:
            return {"deepagents": sdk_dist, "deepagents-code": code_dist}[name]

        with (
            patch("deepagents_code.extras_info.pkg_version", side_effect=fake_version),
            patch("deepagents_code.extras_info.distribution", side_effect=fake_dist),
            patch(
                "deepagents_code.extras_info._read_cli_source_version",
                return_value="0.1.45",
            ),
            patch(
                "deepagents_code.extras_info._cli_editable_info",
                return_value=(True, str(cli_path)),
            ),
        ):
            report = collect_version_report()
        assert report.effective_sdk_version == "0.6.12"
        assert report.display_sdk_version == "0.6.12"
        assert report.sdk_is_workspace_head is False
        assert report.sdk_requirement_satisfied is False
        assert report.sdk_requirement_mismatch is True

    def test_collect_version_report_flags_requirement_mismatch(self) -> None:
        """The aggregated report flags an unsatisfied declared SDK requirement."""
        dist = MagicMock()
        dist.read_text.return_value = None
        dist.requires = ["deepagents==0.7.0a7"]

        def fake_version(name: str) -> str:
            return {"deepagents": "0.6.12", "deepagents-code": "0.1.41"}[name]

        with (
            patch("deepagents_code.extras_info.pkg_version", side_effect=fake_version),
            patch("deepagents_code.extras_info.distribution", return_value=dist),
            patch(
                "deepagents_code.extras_info._cli_editable_info",
                return_value=(False, None),
            ),
        ):
            report = collect_version_report()
        assert report.sdk_requirement is not None
        assert report.sdk_requirement_mismatch is True

    def test_collect_version_report_uses_no_network_or_subprocess(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Collecting the report must not open sockets or spawn subprocesses."""
        import socket
        import subprocess

        def _forbidden(*_args: object, **_kwargs: object) -> None:
            msg = "version collection must stay offline and process-free"
            raise AssertionError(msg)

        monkeypatch.setattr(socket, "socket", _forbidden)
        monkeypatch.setattr(subprocess, "run", _forbidden)
        monkeypatch.setattr(subprocess, "Popen", _forbidden)

        report = collect_version_report()
        assert isinstance(report, VersionReport)
