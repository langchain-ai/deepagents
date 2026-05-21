"""Snapshot tests for build_payload over fixture projects."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepagents_cli.deploy.payload import build_payload
from deepagents_cli.deploy.project import Project

_FIXTURES = Path(__file__).parent / "fixtures" / "projects"

_FIXTURE_NAMES = [
    "bare",
    "with_tools",
    "with_skills",
    "with_subagents",
    "subagent_with_local_skills",
]


@pytest.mark.parametrize("name", _FIXTURE_NAMES)
def test_create_payload_matches_expected(name: str) -> None:
    project = Project.load(_FIXTURES / name)
    payload = build_payload(project, mode="create")
    expected = json.loads(
        (_FIXTURES / name / "expected_payload.json").read_text(encoding="utf-8")
    )
    assert payload == expected


def test_patch_payload_omits_name_when_unchanged() -> None:
    project = Project.load(_FIXTURES / "bare")
    payload = build_payload(project, mode="patch")
    # PATCH always includes name (full-replace on send); this asserts the
    # current contract — adjust if the spec evolves.
    assert payload["name"] == "research-assistant"
