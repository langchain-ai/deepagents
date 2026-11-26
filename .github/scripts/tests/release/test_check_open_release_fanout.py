"""Tests for check_open_release_fanout (post-merge empty-release safety net)."""

import json
import subprocess
from pathlib import Path

import pytest
from check_open_release_fanout import (
    _ref_exists,
    find_empty_release_components,
    main,
    release_tag,
    tag_separator,
)


def test_release_tag_matches_repo_convention() -> None:
    """Tags are `{component}=={version}` per release-please config."""
    assert release_tag("deepagents-acp", "0.2.2") == "deepagents-acp==0.2.2"
    assert release_tag("deepagents", "0.6.12", separator="==") == "deepagents==0.6.12"
    assert tag_separator({"tag-separator": "=="}) == "=="
    assert tag_separator({}) == "=="


def test_ref_exists_returns_false_for_missing_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Git return code 1 means the requested ref does not exist."""
    completed = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="")
    monkeypatch.setattr(
        "check_open_release_fanout.subprocess.run", lambda *_args, **_kwargs: completed
    )
    assert not _ref_exists("deepagents-code==0.1.51", repo_root=tmp_path)


def test_main_ref_lookup_failure_fails_closed(
    capsys, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Operational git failures propagate through `main` as exit code 2."""
    config_path = tmp_path / "release-please-config.json"
    manifest_path = tmp_path / ".release-please-manifest.json"
    config_path.write_text(
        json.dumps({"packages": {"libs/code": {"component": "deepagents-code"}}}),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps({"libs/code": "0.1.51"}), encoding="utf-8")
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=128,
        stdout="",
        stderr="fatal: not a git repository",
    )
    monkeypatch.setattr(
        "check_open_release_fanout.subprocess.run", lambda *_args, **_kwargs: completed
    )

    rc = main(config_path=config_path, manifest_path=manifest_path, repo_root=tmp_path)

    captured = capsys.readouterr()
    assert rc == 2
    assert captured.out == ""
    assert "rc=128" in captured.err
    assert "not a git repository" in captured.err


def _pkg(path: str, component: str) -> dict:
    """Return a package entry that excludes `tests/` and `uv.lock`, like the real config."""
    return {
        "component": component,
        "exclude-paths": [f"{path}/tests", f"{path}/uv.lock"],
    }


def test_find_empty_release_components_resolves_version_to_tag(
    tmp_path: Path, monkeypatch
) -> None:
    """Manifest versions are converted to release tags; excluded-only deltas are flagged."""
    config = {
        "tag-separator": "==",
        "packages": {
            "libs/acp": _pkg("libs/acp", "deepagents-acp"),
            "libs/code": _pkg("libs/code", "deepagents-code"),
            "libs/talon": _pkg("libs/talon", "deepagents-talon"),
        },
    }
    manifest = {
        "libs/acp": "0.2.2",
        "libs/code": "0.1.47",
        "libs/talon": "0.1.0",
    }
    seen: list[tuple[str, str]] = []
    diffs = {
        ("libs/acp", "deepagents-acp==0.2.2"): [
            "libs/acp/tests/test_x.py",
            "libs/acp/uv.lock",
        ],
        ("libs/code", "deepagents-code==0.1.47"): [
            "libs/code/deepagents_code/x.py",
            "libs/code/uv.lock",
        ],
        ("libs/talon", "deepagents-talon==0.1.0"): ["libs/talon/pyproject.toml"],
    }

    def fake_diff(path: str, baseline: str, *, repo_root: Path, head: str = "HEAD"):
        del repo_root, head
        seen.append((path, baseline))
        return diffs.get((path, baseline), [])

    monkeypatch.setattr(
        "check_open_release_fanout._ref_exists", lambda *_a, **_k: True
    )
    monkeypatch.setattr(
        "check_open_release_fanout.package_unreleased_files", fake_diff
    )
    offenders = find_empty_release_components(config, manifest, repo_root=tmp_path)
    assert sorted(seen) == sorted(diffs)
    assert len(offenders) == 1
    assert offenders[0]["component"] == "deepagents-acp"
    assert offenders[0]["version"] == "0.2.2"
    assert offenders[0]["baseline"] == "deepagents-acp==0.2.2"
    assert offenders[0]["files"] == ["libs/acp/tests/test_x.py", "libs/acp/uv.lock"]


def test_find_empty_release_components_flags_no_changes(
    tmp_path: Path, monkeypatch
) -> None:
    """A package with no changes at all since its release has an empty delta too."""
    config = {"packages": {"libs/acp": _pkg("libs/acp", "deepagents-acp")}}
    monkeypatch.setattr(
        "check_open_release_fanout._ref_exists", lambda *_a, **_k: True
    )
    monkeypatch.setattr(
        "check_open_release_fanout.package_unreleased_files", lambda *_a, **_k: []
    )
    offenders = find_empty_release_components(
        config, {"libs/acp": "0.2.2"}, repo_root=tmp_path
    )
    assert [o["component"] for o in offenders] == ["deepagents-acp"]
    assert offenders[0]["files"] == []


def test_lockfile_counts_without_exclude_paths(tmp_path: Path, monkeypatch) -> None:
    """Without an exclusion, a lockfile change is release-relevant and not flagged."""
    config = {"packages": {"libs/acp": {"component": "deepagents-acp"}}}
    monkeypatch.setattr(
        "check_open_release_fanout._ref_exists", lambda *_a, **_k: True
    )
    monkeypatch.setattr(
        "check_open_release_fanout.package_unreleased_files",
        lambda *_a, **_k: ["libs/acp/uv.lock"],
    )
    assert (
        find_empty_release_components(config, {"libs/acp": "0.2.2"}, repo_root=tmp_path)
        == []
    )


def test_main_missing_manifest_fails_closed(capsys, tmp_path: Path) -> None:
    """Missing manifest fails closed (exit 2)."""
    config_path = tmp_path / "release-please-config.json"
    config_path.write_text(
        json.dumps({"packages": {"libs/acp": {"component": "x"}}}), encoding="utf-8"
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
        json.dumps({"packages": {"libs/acp": _pkg("libs/acp", "deepagents-acp")}}),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps({"libs/acp": "0.2.2"}), encoding="utf-8")
    monkeypatch.setattr(
        "check_open_release_fanout._ref_exists", lambda *_a, **_k: True
    )
    monkeypatch.setattr(
        "check_open_release_fanout.package_unreleased_files",
        lambda *a, **k: ["libs/acp/uv.lock"],
    )
    rc = main(config_path=config_path, manifest_path=manifest_path, repo_root=tmp_path)
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload[0]["component"] == "deepagents-acp"
    assert payload[0]["baseline"] == "deepagents-acp==0.2.2"


def test_main_unpublished_tag_is_skipped_not_fatal(
    capsys, tmp_path: Path, monkeypatch
) -> None:
    """A manifest version with no published tag yet is skipped, not an error.

    Covers the release-in-flight window: the release PR merged (manifest
    bumped) but pre-release checks have not created the tag. The watch must
    not hard-crash the whole job on this transient state.
    """
    config_path = tmp_path / "release-please-config.json"
    manifest_path = tmp_path / ".release-please-manifest.json"
    config_path.write_text(
        json.dumps({"packages": {"libs/code": {"component": "deepagents-code"}}}),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps({"libs/code": "0.1.51"}), encoding="utf-8")
    # Tag does not resolve -> component skipped before any diff is attempted.
    monkeypatch.setattr(
        "check_open_release_fanout._ref_exists", lambda *_a, **_k: False
    )
    monkeypatch.setattr(
        "check_open_release_fanout.package_unreleased_files",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("diff must not run for an unpublished tag")
        ),
    )
    rc = main(config_path=config_path, manifest_path=manifest_path, repo_root=tmp_path)
    captured = capsys.readouterr()
    assert rc == 0
    assert json.loads(captured.out) == []
    assert "deepagents-code==0.1.51" in captured.err


def test_main_diff_failure_for_existing_tag_fails_closed(
    capsys, tmp_path: Path, monkeypatch
) -> None:
    """A git diff failure for an existing tag still fails closed (exit 2)."""
    config_path = tmp_path / "release-please-config.json"
    manifest_path = tmp_path / ".release-please-manifest.json"
    config_path.write_text(
        json.dumps({"packages": {"libs/acp": {"component": "deepagents-acp"}}}),
        encoding="utf-8",
    )
    manifest_path.write_text(json.dumps({"libs/acp": "0.2.2"}), encoding="utf-8")
    monkeypatch.setattr(
        "check_open_release_fanout._ref_exists", lambda *_a, **_k: True
    )

    def boom(*_a, **_k):
        raise RuntimeError(
            "git 'diff --name-only deepagents-acp==0.2.2..HEAD -- libs/acp/' "
            "failed (rc=128): fatal: bad object"
        )

    monkeypatch.setattr("check_open_release_fanout.package_unreleased_files", boom)
    rc = main(config_path=config_path, manifest_path=manifest_path, repo_root=tmp_path)
    assert rc == 2
    assert "bad object" in capsys.readouterr().err
