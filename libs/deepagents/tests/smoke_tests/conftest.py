from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def snapshots_dir() -> Path:
    path = Path(__file__).parent / "snapshots"
    path.mkdir(parents=True, exist_ok=True)
    return path
