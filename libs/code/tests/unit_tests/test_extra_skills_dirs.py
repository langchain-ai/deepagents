"""Tests for parsing extra skill directories from environment variables."""

from __future__ import annotations

from pathlib import Path

import pytest

from deepagents_code import config


@pytest.mark.parametrize(
    ("pathsep", "raw", "expected"),
    [
        (";", r"C:\Users\me\skills", [r"C:\Users\me\skills"]),
        (
            ";",
            r"C:\Users\me\skills;D:\team\skills",
            [r"C:\Users\me\skills", r"D:\team\skills"],
        ),
        (":", "/opt/a:/opt/b", ["/opt/a", "/opt/b"]),
    ],
)
def test_parse_extra_skills_dirs_uses_platform_path_separator(
    monkeypatch: pytest.MonkeyPatch,
    pathsep: str,
    raw: str,
    expected: list[str],
) -> None:
    """Environment values retain complete paths for the configured platform."""
    monkeypatch.setattr(config.os, "pathsep", pathsep)
    monkeypatch.setattr(config, "_resolve_extra_skills_path", Path)

    assert config._parse_extra_skills_dirs(raw) == [Path(path) for path in expected]
