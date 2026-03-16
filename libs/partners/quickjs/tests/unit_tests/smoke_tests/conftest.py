from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-snapshots",
        action="store_true",
        default=False,
        help="Update smoke test snapshots on disk.",
    )


@pytest.fixture
def snapshots_dir() -> Path:
    path = Path(__file__).parent / "snapshots"
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture
def update_snapshots(request: pytest.FixtureRequest) -> bool:
    return bool(request.config.getoption("--update-snapshots"))
