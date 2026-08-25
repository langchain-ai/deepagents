"""Unit tests for UI argument parsers."""

import argparse

import pytest

from deepagents_code.ui import non_negative_int, positive_int


class TestPositiveInt:
    """Direct tests for the `positive_int` argparse type converter."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1", 1),
            ("42", 42),
            ("+5", 5),
            (" 7 ", 7),
            ("0001", 1),
        ],
    )
    def test_accepts_positive_integers(self, raw: str, expected: int) -> None:
        """Valid positive integers are parsed, including leading/trailing space."""
        assert positive_int(raw) == expected

    @pytest.mark.parametrize("raw", ["0", "-1", "-42"])
    def test_rejects_non_positive(self, raw: str) -> None:
        """Zero and negatives surface as ArgumentTypeError with the expected copy."""
        with pytest.raises(argparse.ArgumentTypeError) as exc_info:
            positive_int(raw)
        assert "positive integer" in str(exc_info.value)

    @pytest.mark.parametrize("raw", ["abc", "5.0", "", "  ", "1e2", "1,000"])
    def test_rejects_non_integer(self, raw: str) -> None:
        """Non-integer strings (including floats and empty) raise ArgumentTypeError."""
        with pytest.raises(argparse.ArgumentTypeError) as exc_info:
            positive_int(raw)
        assert "invalid int value" in str(exc_info.value)


class TestNonNegativeInt:
    """Direct tests for the `non_negative_int` argparse type converter."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("0", 0),
            ("1", 1),
            ("42", 42),
            ("+5", 5),
            (" 7 ", 7),
            ("0001", 1),
        ],
    )
    def test_accepts_non_negative_integers(self, raw: str, expected: int) -> None:
        """Zero and positive integers parse, including leading/trailing space."""
        assert non_negative_int(raw) == expected

    @pytest.mark.parametrize("raw", ["-1", "-42"])
    def test_rejects_negative(self, raw: str) -> None:
        """Negatives surface as ArgumentTypeError with the expected copy."""
        with pytest.raises(argparse.ArgumentTypeError) as exc_info:
            non_negative_int(raw)
        assert "non-negative integer" in str(exc_info.value)

    @pytest.mark.parametrize("raw", ["abc", "5.0", "", "  ", "1e2", "1,000"])
    def test_rejects_non_integer(self, raw: str) -> None:
        """Non-integer strings (including floats and empty) raise ArgumentTypeError."""
        with pytest.raises(argparse.ArgumentTypeError) as exc_info:
            non_negative_int(raw)
        assert "invalid int value" in str(exc_info.value)
