"""Shared pytest fixtures for cron tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def jobs_path(tmp_path: Path) -> Path:
    """Isolated jobs.json path for one test."""
    return tmp_path / "cron" / "jobs.json"
