"""Tests for check_open_release_fanout (post-merge lockfile release safety net)."""

import json
from pathlib import Path

from check_open_release_fanout import (
    find_lockfile_only_components,
    is_lockfile_only,
    main,
    release_tag,
    tag_separator,
)


def test_is_lockfile_only() -> None:
    """Only non-empty all-lockfile lists count."""
    assert is_lockfile_only(["libs/cli/uv.lock"])
    assert is_lockfile_only(["libs/cli/uv.lock", "libs/code/uv.lock"])
    assert not is_lockfile_only([])
    assert not is_lockfile_only(["libs/cli/pyproject.toml"])
    assert not is_lockfile_only(["libs/cli/uv.lock", "libs/cli/main.py"])


def test_release_tag_matches_repo_convention() -> None:
    """Tags are `{component}=={version}` per release-please config."""
    assert release_tag("deepagents-cli", "0.2.2") == "deepagents-cli==0.2.2"
    assert release_tag("deepagents", "0.6.12", separator="==") == "deepagents==0.6.12"
    assert tag_separator({"tag-separator": "=="}) == "=="
    assert tag_separator({}) == "=="


def test_find_lockfile_only_components_resolves_version_to_tag(
    tmp_path: Path, monkeypatch
) -> None:
    """Manifest versions are converted to release tags before git diff."""
    config = {
        "tag-separator": "==",
        "packages": {
            "libs/cli": {"component": "deepagents-cli"},
            "libs/code": {"component": "deepagents-code"},
        },
    }
    manifest = {
        "libs/cli": "0.2.2",
        "libs/code": "0.1.47",
    }
    seen: list[tuple[str, str]] = []

    def fake_diff(path: str, baseline: str, *, repo_root: Path, head: str = "HEAD"):
        del repo_root, head
        seen.append((path, baseline))
        if path.startswith("libs/cli") and baseline == "deepagents-cli==0.2.2":
            return ["libs/cli/uv.lock"]
        if path.startswith("libs/code") and baseline == "deepagents-code==0.1.47":
            return ["libs/code/deepagents_code/x.py", "libs/code/uv.lock"]
        return []

    monkeypatch.setattr(
        "check_open_release_fanout.package_unreleased_files", fake_diff
    )
    offenders = find_lockfile_only_components(config, manifest, repo_root=tmp_path)
    assert ("libs/cli", "deepagents-cli==0.2.2") in seen
    assert ("libs/code", "deepagents-code==0.1.47") in seen
    assert len(offenders) == 1
    assert offenders[0]["component"] == "deepagents-cli"
    assert offenders[0]["version"] == "0.2.2"
    assert offenders[0]["baseline"] == "deepagents-cli==0.2.2"
    assert offenders[0]["files"] == ["libs/cli/uv.lock"]


def test_main_missing_manifest_fails_closed(capsys, tmp_path: Path) -> None:
    """Missing manifest fails closed (exit 2)."""
    config_path = tmp_path / "release-please-config.json"
    config_path.write_text(
        json.dumps({"packages": {"libs/cli": {"component": "x"}}}), encoding="utf-8"
    )
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
    manifest_path.write_text(json.dumps({"libs/cli": "0.2.2"}), encoding="utf-8")
    monkeypatch.setattr(
        "check_open_release_fanout.package_unreleased_files",
        lambda *a, **k: ["libs/cli/uv.lock"],
    )
    rc = main(config_path=config_path, manifest_path=manifest_path, repo_root=tmp_path)
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload[0]["component"] == "deepagents-cli"
    assert payload[0]["baseline"] == "deepagents-cli==0.2.2"


def test_main_missing_tag_fails_closed(capsys, tmp_path: Path, monkeypatch) -> None:
    """Unresolved release tags fail closed rather than skipping packages."""
    config_path = tmp_path / "release-please-config.json"
    manifest_path = tmp_path / ".release-please-manifest.json"
    config_path.write_text(
        json.dumps({"packages": {"libs/cli": {"component": "deepagents-cli"}}}),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps({"libs/cli": "0.2.2"}), encoding="utf-8")

    def boom(*_a, **_k):
        raise RuntimeError(
            "git 'diff --name-only deepagents-cli==0.2.2..HEAD -- libs/cli/' "
            "failed (rc=128): fatal: bad revision"
        )

    monkeypatch.setattr("check_open_release_fanout.package_unreleased_files", boom)
    rc = main(config_path=config_path, manifest_path=manifest_path, repo_root=tmp_path)
    assert rc == 2
    assert "bad revision" in capsys.readouterr().err
