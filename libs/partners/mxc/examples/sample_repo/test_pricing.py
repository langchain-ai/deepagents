from pricing import apply_discount, total


def test_no_discount() -> None:
    assert apply_discount(19.99, 0) == 19.99


def test_percent_off() -> None:
    # Fails until the bug in apply_discount is fixed: 25% off 80.00 is 60.00.
    assert apply_discount(80.0, 25) == 60.0


def test_total_without_discount() -> None:
    assert total([10.0, 20.0, 30.0]) == 60.0
