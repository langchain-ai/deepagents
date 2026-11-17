"""Tiny pricing helpers for the langchain-mxc demo."""


def apply_discount(price: float, percent: float) -> float:
    """Return `price` reduced by `percent` percent, rounded to cents.

    Examples:
        apply_discount(100.0, 10) -> 90.0
        apply_discount(19.99, 0) -> 19.99
    """
    if not 0 <= percent <= 100:
        msg = f"percent must be between 0 and 100, got {percent}"
        raise ValueError(msg)
    # BUG: this subtracts the percentage as an absolute amount.
    return round(price - percent, 2)


def total(prices: list[float], discount_percent: float = 0) -> float:
    """Sum `prices` and apply one discount to the total."""
    return apply_discount(sum(prices), discount_percent)
