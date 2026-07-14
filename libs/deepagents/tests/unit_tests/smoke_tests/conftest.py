from __future__ import annotations

from pathlib import Path

import pytest

from deepagents.middleware import filesystem


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


@pytest.fixture(autouse=True)
def _pin_video_dependencies(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the `read_file` video gate so prompt snapshots are env-independent.

    `read_file`'s description switches between a text-only and a video-aware
    variant based on `video_dependencies_available()` (the optional `[video]`
    extra: PyAV + Pillow). Whether that extra is installed varies by environment,
    so without pinning the gate, the same test would snapshot different wording on
    different machines and `--update-snapshots` could silently swap variants.

    The gate defaults to off (the text-only base install); a test marked
    `video_extra` pins it on to cover the `[video]`-extra variant. Rendering the
    description never touches PyAV, so forcing either state is safe regardless of
    what is installed.
    """
    enabled = request.node.get_closest_marker("video_extra") is not None
    monkeypatch.setattr(filesystem, "video_dependencies_available", lambda: enabled)
