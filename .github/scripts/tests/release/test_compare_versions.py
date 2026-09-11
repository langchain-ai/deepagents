"""Tests for the PEP 440 version relation used by the pin bump workflow."""

import pytest
from compare_versions import compare
from packaging.version import InvalidVersion


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("0.7.12", "0.7.13", "behind"),
        ("0.7.13", "0.7.13", "current"),
        ("0.7.14", "0.7.13", "ahead"),
        # A string comparison would call this one `ahead`.
        ("0.7.9", "0.7.10", "behind"),
        # A prerelease sorts below the release it leads to.
        ("0.8.0rc1", "0.8.0", "behind"),
        # `packaging` normalises these two spellings to the same version.
        ("0.8.0-rc.1", "0.8.0rc1", "current"),
    ],
)
def test_compare_orders_versions_by_pep_440(
    left: str, right: str, expected: str
) -> None:
    assert compare(left, right) == expected


def test_compare_rejects_a_malformed_version() -> None:
    with pytest.raises(InvalidVersion):
        compare("not-a-version", "0.7.13")
