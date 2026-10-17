"""Unit tests for the `hatch_build.py` release-commit stamping hook.

`hatch_build.py` lives at the package root (not inside `deepagents_code`) and is
only on `sys.path` during a build, so it is loaded here directly from its file
path. The hook subclasses hatchling's `BuildHookInterface`, so these tests
require `hatchling` (declared in the `test` dependency group).
"""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_HATCH_BUILD_PATH = Path(__file__).resolve().parents[2] / "hatch_build.py"


def _load_hatch_build() -> ModuleType:
    """Load `hatch_build.py` from its on-disk path (it is not importable)."""
    spec = importlib.util.spec_from_file_location("hatch_build", _HATCH_BUILD_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_hatch_build = _load_hatch_build()
CustomBuildHook = _hatch_build.CustomBuildHook
_COMMIT_ENV = _hatch_build._COMMIT_ENV


def _make_hook(root: Path) -> BuildHookInterface:
    """Build a hook rooted at `root`.

    The hook only reads `self.root`, so the remaining `BuildHookInterface`
    constructor arguments (config, build config, metadata) are inert here.
    """
    return CustomBuildHook(str(root), {}, None, None, str(root), "wheel")


def _stamp_path(root: Path) -> Path:
    return root / "deepagents_code" / "_build_info.py"


class TestInitialize:
    """Tests for the `initialize` build hook (writes the stamp)."""

    def test_env_unset_writes_nothing(self, tmp_path: Path, monkeypatch) -> None:
        """With the env var unset, no file is written and no artifact is added."""
        monkeypatch.delenv(_COMMIT_ENV, raising=False)
        (tmp_path / "deepagents_code").mkdir()
        build_data: dict[str, object] = {}

        _make_hook(tmp_path).initialize("1.0", build_data)

        assert not _stamp_path(tmp_path).exists()
        assert build_data == {}

    def test_valid_sha_is_shortened_lowercased_and_registered(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A full uppercase SHA stamps a 7-char lowercase value and an artifact."""
        monkeypatch.setenv(_COMMIT_ENV, "ABCDEF1234567890ABCDEF1234567890ABCDEF12")
        (tmp_path / "deepagents_code").mkdir()
        build_data: dict[str, object] = {}

        _make_hook(tmp_path).initialize("1.0", build_data)

        assert 'BUILD_COMMIT = "abcdef1"' in _stamp_path(tmp_path).read_text(
            encoding="utf-8"
        )
        assert build_data["artifacts"] == ["deepagents_code/_build_info.py"]

    @pytest.mark.parametrize(
        "bad",
        [
            "main",  # not hex
            "g123abc",  # non-hex char
            "abcdef",  # too short (< 7)
            "0" * 41,  # too long (> 40)
        ],
    )
    def test_invalid_sha_raises_and_writes_nothing(
        self, tmp_path: Path, monkeypatch, bad: str
    ) -> None:
        """A set-but-malformed SHA fails the build loudly and writes no file."""
        monkeypatch.setenv(_COMMIT_ENV, bad)
        (tmp_path / "deepagents_code").mkdir()

        with pytest.raises(ValueError, match=_COMMIT_ENV):
            _make_hook(tmp_path).initialize("1.0", {})

        assert not _stamp_path(tmp_path).exists()

    @pytest.mark.parametrize("blank", ["", "   "])
    def test_blank_env_writes_nothing(
        self, tmp_path: Path, monkeypatch, blank: str
    ) -> None:
        """A blank or whitespace-only env var is treated as unset."""
        monkeypatch.setenv(_COMMIT_ENV, blank)
        (tmp_path / "deepagents_code").mkdir()
        build_data: dict[str, object] = {}

        _make_hook(tmp_path).initialize("1.0", build_data)

        assert not _stamp_path(tmp_path).exists()
        assert build_data == {}


class TestFinalize:
    """Tests for the `finalize` build hook (cleans up the stamp)."""

    def test_removes_stamp_when_env_set(self, tmp_path: Path, monkeypatch) -> None:
        """Cleanup deletes the generated file when the env var is set."""
        monkeypatch.setenv(_COMMIT_ENV, "abc1234")
        (tmp_path / "deepagents_code").mkdir()
        stamp = _stamp_path(tmp_path)
        stamp.write_text("x", encoding="utf-8")

        _make_hook(tmp_path).finalize("1.0", {}, "/tmp/wheel")  # not a real path

        assert not stamp.exists()

    def test_noop_when_env_unset(self, tmp_path: Path, monkeypatch) -> None:
        """Cleanup is gated on the env var, so an unrelated file is untouched."""
        monkeypatch.delenv(_COMMIT_ENV, raising=False)
        (tmp_path / "deepagents_code").mkdir()
        stamp = _stamp_path(tmp_path)
        stamp.write_text("x", encoding="utf-8")

        _make_hook(tmp_path).finalize("1.0", {}, "/tmp/wheel")  # not a real path

        assert stamp.exists()

    def test_missing_file_does_not_raise(self, tmp_path: Path, monkeypatch) -> None:
        """Cleanup tolerates an already-absent file (`missing_ok=True`)."""
        monkeypatch.setenv(_COMMIT_ENV, "abc1234")
        (tmp_path / "deepagents_code").mkdir()

        _make_hook(tmp_path).finalize("1.0", {}, "/tmp/wheel")  # not a real path

        assert not _stamp_path(tmp_path).exists()


def test_initialize_then_finalize_leaves_tree_clean(
    tmp_path: Path, monkeypatch
) -> None:
    """A full stamp/cleanup cycle writes the file then removes it."""
    monkeypatch.setenv(_COMMIT_ENV, "deadbeef")
    (tmp_path / "deepagents_code").mkdir()
    hook = _make_hook(tmp_path)

    hook.initialize("1.0", {})
    assert _stamp_path(tmp_path).exists()

    hook.finalize("1.0", {}, "/tmp/wheel")  # not a real path
    assert not _stamp_path(tmp_path).exists()
