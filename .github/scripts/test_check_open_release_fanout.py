"""Tests for check_open_release_fanout (post-merge lockfile release safety net)."""

import json
from pathlib import Path

from check_open_release_fanout import (
    find_lockfile_only_components,
    is_lockfile_only,
    main,
)


def test_is_lockfile_only() -> None:
    """Only non-empty all-lockfile lists count."""
    assert is_lockfile_only(["libs/cli/uv.lock"])
    assert is_lockfile_only(["libs/cli/uv.lock", "libs/code/uv.lock"])
    assert not is_lockfile_only([])
    assert not is_lockfile_only(["libs/cli/pyproject.toml"])
    assert not is_lockfile_only(["libs/cli/uv.lock", "libs/cli/main.py"])


def test_find_lockfile_only_components_uses_git_diff(tmp_path: Path, monkeypatch) -> None:
    """Components whose package delta is only uv.lock are reported."""
    config = {
        "packages": {
            "libs/cli": {"component": "deepagents-cli"},
            "libs/code": {"component": "deepagents-code"},
        }
    }
    manifest = {
        "libs/cli": "aaa",
        "libs/code": "bbb",
    }

    def fake_diff(path: str, baseline: str, *, repo_root: Path, head: str = "HEAD"):
        del repo_root, head
        if path.startswith("libs/cli") and baseline == "aaa":
            return ["libs/cli/uv.lock"]
        if path.startswith("libs/code") and baseline == "bbb":
            return ["libs/code/deepagents_code/x.py", "libs/code/uv.lock"]
        return []

    monkeypatch.setattr(
        "check_open_release_fanout.package_unreleased_files", fake_diff
    )
    offenders = find_lockfile_only_components(config, manifest, repo_root=tmp_path)
    assert len(offenders) == 1
    assert offenders[0]["component"] == "deepagents-cli"
    assert offenders[0]["files"] == ["libs/cli/uv.lock"]


def test_main_missing_manifest_fails_closed(capsys, tmp_path: Path) -> None:
    """Missing manifest fails closed (exit 2)."""
    config_path = tmp_path / "release-please-config.json"
    config_path.write_text(json.dumps({"packages": {"libs/cli": {"component": "x"}}}), encoding="utf-8")
    rc = main(
        config_path=config_path,
        manifest_path=tmp_path / "missing.json",
        repo_root=tmp_path,
    )
    assert rc == 2
    assert "::error::" in capsys.readouterr().err


def test_main_happy_path(capsys, tmp_path: Path, monkeypatch) -> None:
    """main prints JSON offenders and exits 0."""
    config_path = tmp_path / "release-please-config.json"
    manifest_path = tmp_path / ".release-please-manifest.json"
    config_path.write_text(
        json.dumps({"packages": {"libs/cli": {"component": "deepagents-cli"}}}),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps({"libs/cli": "abc"}), encoding="utf-8")
    monkeypatch.setattr(
        "check_open_release_fanout.package_unreleased_files",
        lambda *a, **k: ["libs/cli/uv.lock"],
    )
    rc = main(config_path=config_path, manifest_path=manifest_path, repo_root=tmp_path)
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload[0]["component"] == "deepagents-cli"
