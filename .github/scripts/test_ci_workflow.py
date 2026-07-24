"""Static contracts for the main CI workflow."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = ROOT / ".github/workflows/ci.yml"


def test_deepagents_code_collects_coverage_on_python_3_14() -> None:
    """Keep all supported runtimes while collecting coverage on Python 3.14."""
    workflow = yaml.safe_load(CI_WORKFLOW.read_text())
    config = workflow["jobs"]["test-code"]["with"]

    assert json.loads(config["python-versions"]) == ["3.11", "3.12", "3.13", "3.14"]
    assert config["coverage-python-version"] == "3.14"
