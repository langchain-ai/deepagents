"""Tests for automatic local-consumer lockfile refreshes."""

import subprocess
from pathlib import Path

import check_lockfiles_pre_commit as lockfiles
import pytest
import refresh_lockfiles


def test_discovers_new_transitive_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name, source in (
        ("code", None),
        ("new", "code"),
        ("acp", "new"),
        ("other", None),
    ):
        package = tmp_path / "libs" / name
        package.mkdir(parents=True)
        (package / "Makefile").touch()
        manifest = '[project]\nname = "' + name + '"\nversion = "0.1.0"\n'
        if source:
            manifest += f'[tool.uv.sources]\n{source} = {{path = "../{source}"}}\n'
        (package / "pyproject.toml").write_text(manifest)
    monkeypatch.setattr(lockfiles, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(lockfiles, "LIBS_ROOT", tmp_path / "libs")
    monkeypatch.setattr(lockfiles, "EXAMPLES_ROOT", tmp_path / "examples")
    monkeypatch.setattr(refresh_lockfiles, "REPO_ROOT", tmp_path)
    commands = []
    monkeypatch.setattr(
        refresh_lockfiles.subprocess,
        "run",
        lambda args, **kwargs: commands.append(args),
    )

    refresh_lockfiles.main(["libs/code/pyproject.toml"])

    assert capsys.readouterr().out.splitlines() == [
        "libs/acp/uv.lock",
        "libs/code/uv.lock",
        "libs/new/uv.lock",
    ]
    assert commands == [
        ["uv", "lock", "--directory", "libs/acp", "--python", "3.14"],
        ["uv", "lock", "--directory", "libs/code", "--python", "3.12"],
        ["uv", "lock", "--directory", "libs/new", "--python", "3.12"],
    ]


def test_failed_lock_stops_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        refresh_lockfiles, "stale_lock_dirs", lambda _: ["libs/code", "libs/evals"]
    )
    calls = []

    def fail(args: list[str], **kwargs: object) -> None:
        calls.append(args)
        raise subprocess.CalledProcessError(1, args)

    monkeypatch.setattr(refresh_lockfiles.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        refresh_lockfiles.main(["libs/code/pyproject.toml"])
    assert len(calls) == 1
