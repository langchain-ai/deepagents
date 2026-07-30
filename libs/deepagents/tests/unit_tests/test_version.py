"""Test that package version is consistent across configuration files."""

from __future__ import annotations

import json
import tomllib
from importlib.metadata import PathDistribution
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

import deepagents

if TYPE_CHECKING:
    from collections.abc import Iterator
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


def _dist(
    *,
    name: str = "deepagents",
    direct_url: dict[str, object] | None = None,
    raw: str | None = None,
) -> MagicMock:
    """Build a minimal `importlib.metadata.Distribution` stand-in.

    Pass `direct_url` for a well-formed payload, or `raw` for the exact
    `direct_url.json` text when the point of the test is malformed content.
    """
    dist = MagicMock()
    dist.name = name
    dist.metadata = {"Name": name}
    if raw is not None:
        dist.read_text.return_value = raw
    elif direct_url is None:
        dist.read_text.return_value = None
    else:
        dist.read_text.return_value = json.dumps(direct_url)
    return dist


def _editable_json(root: Path) -> str:
    """Build a PEP 610 payload as `pip install -e <root>` writes it.

    `root.as_uri()` keeps the URL valid on Windows, where a POSIX-style literal
    such as `file:///src/deepagents` is not a usable path.
    """
    return json.dumps({"url": root.as_uri(), "dir_info": {"editable": True}})


@pytest.fixture
def editable_root(tmp_path: Path) -> Path:
    """A source root laid out like an editable `deepagents` checkout."""
    root = tmp_path / "workspace" / "libs" / "deepagents"
    (root / "deepagents").mkdir(parents=True)
    return root


@pytest.fixture(autouse=True)
def _running_from_editable_root(editable_root: Path) -> Iterator[None]:
    """Place the running module inside `editable_root` for every test.

    `_is_editable_install` only credits an editable record whose source root
    contains the module being imported, so the fixtures have to agree about where
    that module lives. Tests that need them to disagree patch this themselves.
    """
    with patch(
        "deepagents._version._running_package_root",
        return_value=(editable_root / "deepagents").resolve(),
    ):
        yield


def _real_dist(
    root: Path,
    *,
    dirname: str,
    name: str | None = "deepagents",
    version: str = "0.7.0",
    metadata_name: str = "METADATA",
    metadata_bytes: bytes | None = None,
    direct_url: str | None = None,
) -> PathDistribution:
    """Build a real `PathDistribution` backed by an on-disk metadata directory.

    Unlike `_dist`, this drives the genuine `read_text` filename lookup and the
    real `Distribution.name` property, so it catches mistakes a `MagicMock`
    cannot: `MagicMock.read_text` ignores its argument, so a misspelled
    `direct_url.json` would still "work" against a mock while silently disabling
    detection in production.

    Args:
        root: Directory to create the metadata directory under.
        dirname: Metadata directory name, e.g. `deepagents-0.7.0.dist-info` or
            `deepagents.egg-info`. Explicit so tests mirror real layouts.
        name: Distribution name written into the metadata file. `None` writes no
            metadata file at all, mimicking a partial install.
        version: Version written into the metadata file.
        metadata_name: Metadata filename — `PKG-INFO` for `*.egg-info` layouts.
        metadata_bytes: Raw metadata bytes, for non-UTF-8 content.
        direct_url: Exact `direct_url.json` text, or `None` to omit the file.
    """
    info = root / dirname
    info.mkdir(parents=True)
    if metadata_bytes is not None:
        (info / metadata_name).write_bytes(metadata_bytes)
    elif name is not None:
        (info / metadata_name).write_text(f"Name: {name}\nVersion: {version}\n")
    if direct_url is not None:
        (info / "direct_url.json").write_text(direct_url)
    return PathDistribution(info)


class TestWithEditableLocalVersion:
    """Tests for `_with_editable_local_version`."""

    def test_appends_editable_local_segment(self) -> None:
        assert _with_editable_local_version("0.6.12") == "0.6.12+editable"

    def test_preserves_existing_local_segment(self) -> None:
        assert _with_editable_local_version("0.6.12+build") == "0.6.12+build.editable"

    def test_returns_original_for_invalid_version(self) -> None:
        assert _with_editable_local_version("not-a-version") == "not-a-version"

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param("1.0.0-alpha1", "1.0.0a1+editable", id="prerelease-normalized"),
            pytest.param("v1.2.3", "1.2.3+editable", id="v-prefix-stripped"),
            pytest.param("01.2.3", "1.2.3+editable", id="leading-zeros-stripped"),
            pytest.param("1.0+Build", "1.0+build.editable", id="local-lowercased"),
            pytest.param("1!2.0", "1!2.0+editable", id="epoch-preserved"),
        ],
    )
    def test_reemits_base_in_canonical_form(self, value: str, expected: str) -> None:
        """The base version is canonicalized, so output can differ beyond the suffix."""
        assert _with_editable_local_version(value) == expected

    def test_is_not_idempotent(self) -> None:
        """Applying twice stacks segments; callers must apply exactly once."""
        once = _with_editable_local_version(__version__)
        assert _with_editable_local_version(once) == f"{once}.editable"


class TestIsEditableInstall:
    """Tests for `_is_editable_install`."""

    def test_true_when_pep610_marks_editable(self, editable_root: Path) -> None:
        editable = _dist(raw=_editable_json(editable_root))
        with patch("deepagents._version.distributions", return_value=[editable]):
            assert _is_editable_install() is True

    def test_false_for_non_editable_local_dir_install(self, editable_root: Path) -> None:
        local_dir = _dist(
            direct_url={"url": editable_root.as_uri(), "dir_info": {}},
        )
        with patch("deepagents._version.distributions", return_value=[local_dir]):
            assert _is_editable_install() is False

    def test_false_for_archive_install(self) -> None:
        """PEP 610 archive installs carry `archive_info`, never `dir_info`."""
        wheel = _dist(
            direct_url={
                "url": "https://example.com/deepagents-0.6.12.tar.gz",
                "archive_info": {"hashes": {"sha256": "abc"}},
            },
        )
        with patch("deepagents._version.distributions", return_value=[wheel]):
            assert _is_editable_install() is False

    def test_false_when_direct_url_missing(self) -> None:
        egg_info = _dist(direct_url=None)
        with patch("deepagents._version.distributions", return_value=[egg_info]):
            assert _is_editable_install() is False

    def test_ignores_cwd_egg_info_shadowing_editable_install(self, editable_root: Path) -> None:
        """A local `*.egg-info` without PEP 610 data must not hide site-packages."""
        egg_info = _dist(direct_url=None)
        editable = _dist(raw=_editable_json(editable_root))
        with patch("deepagents._version.distributions", return_value=[egg_info, editable]):
            assert _is_editable_install() is True

    def test_ignores_other_editable_distributions(self, editable_root: Path) -> None:
        """Only `deepagents` counts — a sibling editable lib must not qualify."""
        other = _dist(name="langchain-core", raw=_editable_json(editable_root))
        with patch("deepagents._version.distributions", return_value=[other]):
            assert _is_editable_install() is False

    def test_matches_name_case_insensitively(self, editable_root: Path) -> None:
        renamed = _dist(name="DeepAgents", raw=_editable_json(editable_root))
        with patch("deepagents._version.distributions", return_value=[renamed]):
            assert _is_editable_install() is True

    def test_false_when_no_distributions(self) -> None:
        with patch("deepagents._version.distributions", return_value=[]):
            assert _is_editable_install() is False

    def test_false_when_distributions_raises_immediately(self) -> None:
        with patch(
            "deepagents._version.distributions",
            side_effect=OSError("metadata unavailable"),
        ):
            assert _is_editable_install() is False

    def test_false_when_distributions_raises_mid_iteration(self) -> None:
        """`distributions()` yields lazily, so real errors surface while iterating."""

        def _explode() -> Iterator[MagicMock]:
            yield _dist(direct_url=None)
            error = OSError("sys.path entry vanished")
            raise error

        with patch("deepagents._version.distributions", side_effect=_explode):
            assert _is_editable_install() is False

    @pytest.mark.parametrize(
        "raw",
        [
            pytest.param("{not json", id="malformed-json"),
            pytest.param("[1, 2]", id="non-dict-payload"),
            pytest.param('"a string"', id="string-payload"),
            pytest.param('{"dir_info": null}', id="null-dir-info"),
            pytest.param('{"dir_info": "yes"}', id="non-dict-dir-info"),
            pytest.param("{}", id="missing-dir-info"),
            pytest.param('{"dir_info": {"editable": "false"}}', id="stringy-editable"),
            pytest.param('{"dir_info": {"editable": 1}}', id="numeric-editable"),
            pytest.param("[" * 100_000, id="pathological-nesting"),
        ],
    )
    def test_false_for_unusable_direct_url_payloads(self, raw: str) -> None:
        """Unusable metadata reports non-editable and never raises."""
        broken = _dist(raw=raw)
        with patch("deepagents._version.distributions", return_value=[broken]):
            assert _is_editable_install() is False

    def test_false_when_read_text_raises(self) -> None:
        unreadable = _dist()
        unreadable.read_text.side_effect = OSError("unreadable path")
        with patch("deepagents._version.distributions", return_value=[unreadable]):
            assert _is_editable_install() is False

    def test_unreadable_direct_url_does_not_abort_scan(self, editable_root: Path) -> None:
        """A read failure must be contained per-distribution, not end the scan.

        The outer backstop would also swallow this `OSError`, but only by giving
        up on every remaining distribution — which makes the answer depend on
        iteration order.
        """
        unreadable = _dist()
        unreadable.read_text.side_effect = OSError("unreadable path")
        editable = _dist(raw=_editable_json(editable_root))
        with patch("deepagents._version.distributions", return_value=[unreadable, editable]):
            assert _is_editable_install() is True

    def test_malformed_direct_url_does_not_abort_scan(self, editable_root: Path) -> None:
        """Same containment requirement for unparseable JSON."""
        broken = _dist(raw="{not json")
        editable = _dist(raw=_editable_json(editable_root))
        with patch("deepagents._version.distributions", return_value=[broken, editable]):
            assert _is_editable_install() is True


class TestIsEditableInstallAgainstRealMetadata:
    """`_is_editable_install` against real on-disk `Distribution` objects.

    These are the fidelity tests: they exercise the true `direct_url.json`
    filename lookup and the real `Distribution.name` property, which the
    `MagicMock` cases cannot.
    """

    def test_detects_real_editable_dist_info(self, tmp_path: Path, editable_root: Path) -> None:
        dist = _real_dist(
            tmp_path,
            dirname="deepagents-0.7.0.dist-info",
            direct_url=_editable_json(editable_root),
        )
        with patch("deepagents._version.distributions", return_value=[dist]):
            assert _is_editable_install() is True

    def test_real_wheel_install_is_not_editable(self, tmp_path: Path) -> None:
        dist = _real_dist(tmp_path, dirname="deepagents-0.7.0.dist-info")
        with patch("deepagents._version.distributions", return_value=[dist]):
            assert _is_editable_install() is False

    def test_real_egg_info_does_not_shadow_editable_install(self, tmp_path: Path, editable_root: Path) -> None:
        """Reproduces the layout that makes the single-lookup form wrong.

        `setuptools` leaves a `deepagents.egg-info/` in the source tree, and it is
        discovered before site-packages when running from the checkout.
        """
        egg_info = _real_dist(
            tmp_path / "src",
            dirname="deepagents.egg-info",
            metadata_name="PKG-INFO",
        )
        editable = _real_dist(
            tmp_path / "site-packages",
            dirname="deepagents-0.7.0.dist-info",
            direct_url=_editable_json(editable_root),
        )
        with patch("deepagents._version.distributions", return_value=[egg_info, editable]):
            assert _is_editable_install() is True

    def test_corrupt_neighbor_does_not_abort_scan(self, tmp_path: Path, editable_root: Path) -> None:
        """An unreadable *unrelated* distribution must not mask a later editable one.

        `Distribution.name` parses `METADATA`, so non-UTF-8 bytes raise
        `UnicodeDecodeError` (a `ValueError`). If that escaped the per-distribution
        helper it would end the scan early, and the result would depend on
        iteration order.
        """
        corrupt = _real_dist(
            tmp_path / "a",
            dirname="badpkg-1.0.dist-info",
            metadata_bytes=b"Name: badpkg\nSummary: \xff\xfe caf\xe9\n",
        )
        editable = _real_dist(
            tmp_path / "b",
            dirname="deepagents-0.7.0.dist-info",
            direct_url=_editable_json(editable_root),
        )
        with patch("deepagents._version.distributions", return_value=[corrupt, editable]):
            assert _is_editable_install() is True

    def test_metadata_less_dist_info_does_not_raise(self, tmp_path: Path, editable_root: Path) -> None:
        """A partial install leaves a `*.dist-info` whose real `.name` is `None`.

        `AttributeError` from `None.lower()` is not caught by the scan's backstop,
        so an unguarded name lookup would crash `create_deep_agent()`.
        """
        ghost = _real_dist(tmp_path / "a", dirname="ghost-0.1.dist-info", name=None)
        editable = _real_dist(
            tmp_path / "b",
            dirname="deepagents-0.7.0.dist-info",
            direct_url=_editable_json(editable_root),
        )
        with patch("deepagents._version.distributions", return_value=[ghost, editable]):
            assert _is_editable_install() is True

    def test_real_malformed_direct_url_is_not_editable(self, tmp_path: Path) -> None:
        dist = _real_dist(
            tmp_path,
            dirname="deepagents-0.7.0.dist-info",
            direct_url="{not json",
        )
        with patch("deepagents._version.distributions", return_value=[dist]):
            assert _is_editable_install() is False


class TestEditableRecordMustSupplyRunningModule:
    """An editable record only counts if it supplies the module being imported.

    An environment can hold more than one `deepagents` record — for example a
    wheel earlier on `sys.path` and an unrelated editable checkout later on it.
    Only one of them actually provides the imported code, and crediting the wrong
    one would report a published install as a workspace build.
    """

    def test_unrelated_editable_checkout_does_not_mark_wheel_editable(self, tmp_path: Path) -> None:
        """The running wheel must not inherit another checkout's editable status."""
        elsewhere = tmp_path / "some" / "other" / "checkout"
        elsewhere.mkdir(parents=True)
        wheel = _real_dist(tmp_path / "site-packages", dirname="deepagents-0.7.0.dist-info")
        unrelated_editable = _real_dist(
            tmp_path / "other-env",
            dirname="deepagents-0.7.0.dist-info",
            direct_url=_editable_json(elsewhere),
        )
        running_from = tmp_path / "site-packages" / "deepagents"
        running_from.mkdir(parents=True)
        with (
            patch(
                "deepagents._version._running_package_root",
                return_value=running_from.resolve(),
            ),
            patch(
                "deepagents._version.distributions",
                return_value=[wheel, unrelated_editable],
            ),
        ):
            assert _is_editable_install() is False

    def test_editable_wins_when_it_supplies_the_running_module(self, tmp_path: Path, editable_root: Path) -> None:
        """An unrelated editable record ahead of ours must not shadow the real one."""
        elsewhere = tmp_path / "some" / "other" / "checkout"
        elsewhere.mkdir(parents=True)
        unrelated_editable = _real_dist(
            tmp_path / "other-env",
            dirname="deepagents-0.7.0.dist-info",
            direct_url=_editable_json(elsewhere),
        )
        ours = _real_dist(
            tmp_path / "site-packages",
            dirname="deepagents-0.7.0.dist-info",
            direct_url=_editable_json(editable_root),
        )
        with patch(
            "deepagents._version.distributions",
            return_value=[unrelated_editable, ours],
        ):
            assert _is_editable_install() is True

    def test_src_layout_source_root_still_matches(self, tmp_path: Path) -> None:
        """A `src/` layout puts the package below the recorded root, not at it."""
        root = tmp_path / "project"
        package = root / "src" / "deepagents"
        package.mkdir(parents=True)
        dist = _real_dist(
            tmp_path / "site-packages",
            dirname="deepagents-0.7.0.dist-info",
            direct_url=_editable_json(root),
        )
        with (
            patch(
                "deepagents._version._running_package_root",
                return_value=package.resolve(),
            ),
            patch("deepagents._version.distributions", return_value=[dist]),
        ):
            assert _is_editable_install() is True

    @pytest.mark.parametrize(
        "url",
        [
            pytest.param("https://example.com/deepagents", id="non-file-scheme"),
            pytest.param("", id="empty"),
        ],
    )
    def test_false_when_source_url_is_unusable(self, url: str) -> None:
        """An editable record we cannot locate cannot be correlated, so it is skipped."""
        dist = _dist(raw=json.dumps({"url": url, "dir_info": {"editable": True}}))
        with patch("deepagents._version.distributions", return_value=[dist]):
            assert _is_editable_install() is False

    def test_false_when_source_url_absent(self) -> None:
        dist = _dist(raw=json.dumps({"dir_info": {"editable": True}}))
        with patch("deepagents._version.distributions", return_value=[dist]):
            assert _is_editable_install() is False

    def test_false_when_running_module_location_is_unknown(self, editable_root: Path) -> None:
        """Some frozen interpreters have no `__file__`, so nothing can be correlated."""
        editable = _dist(raw=_editable_json(editable_root))
        with (
            patch("deepagents._version._running_package_root", return_value=None),
            patch("deepagents._version.distributions", return_value=[editable]),
        ):
            assert _is_editable_install() is False


class TestLcVersion:
    """Tests for `_lc_version`."""

    def setup_method(self) -> None:
        """Clear the process-level version cache before each test."""
        _lc_version.cache_clear()

    def teardown_method(self) -> None:
        """Avoid leaking cached test values to other tests."""
        _lc_version.cache_clear()

    def test_plain_release_install(self) -> None:
        with patch("deepagents._version._is_editable_install", return_value=False):
            assert _lc_version() == __version__

    def test_editable_install_gets_local_segment(self) -> None:
        with patch("deepagents._version._is_editable_install", return_value=True):
            assert _lc_version() == f"{__version__}+editable"

    def test_caches_editable_install_lookup(self) -> None:
        with patch("deepagents._version._is_editable_install", return_value=False) as is_editable_install:
            assert _lc_version() == __version__
            assert _lc_version() == __version__

        is_editable_install.assert_called_once_with()
