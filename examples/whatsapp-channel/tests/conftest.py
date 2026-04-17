"""Shared pytest fixtures for the WhatsApp-channel example tests."""

from __future__ import annotations

from pathlib import Path

import pytest


# Smallest valid PNG: 1x1 red pixel. Enough for base64-encoding tests
# and for _sniff_image_mime magic-byte detection.
_TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf\xc0\x00\x00\x00\x03\x00\x01"
    b"\xff\xfd\x8f\xb5\x00\x00\x00\x00IEND\xaeB`\x82"
)

# Minimal JPEG: SOI + JFIF APP0 + EOI. Not decodable as image but has
# the \xff\xd8\xff magic-byte prefix the adapter's sniffer looks for,
# and the bytes round-trip through base64 for multimodal tests.
_TINY_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01"
    b"\x00\x00\xff\xd9"
)


@pytest.fixture
def jobs_path(tmp_path: Path) -> Path:
    """Isolated jobs.json path for one test."""
    return tmp_path / "cron" / "jobs.json"


@pytest.fixture
def tiny_png(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.png"
    p.write_bytes(_TINY_PNG)
    return p


@pytest.fixture
def tiny_jpeg(tmp_path: Path) -> Path:
    p = tmp_path / "tiny.jpg"
    p.write_bytes(_TINY_JPEG)
    return p


@pytest.fixture
def oversize_image(tmp_path: Path) -> Path:
    """Sparse file with 6 MB apparent size — exceeds the 5 MB inbound cap."""
    p = tmp_path / "big.png"
    with open(p, "wb") as f:
        f.seek(6 * 1024 * 1024)
        f.write(b"\x00")
    return p
