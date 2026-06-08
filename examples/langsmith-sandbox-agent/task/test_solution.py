import pytest

from solution import roman_to_int


@pytest.mark.parametrize(
    "numeral,expected",
    [
        ("III", 3),
        ("IV", 4),
        ("IX", 9),
        ("LVIII", 58),
        ("MCMXCIV", 1994),
        ("MMXXVI", 2026),
    ],
)
def test_roman_to_int(numeral, expected):
    assert roman_to_int(numeral) == expected
