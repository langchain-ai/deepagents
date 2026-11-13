"""Check USD cost estimates, including provider-specific caching and surcharges."""

import pytest

from deepagents_evals.routing.middleware import ModelPricing


def test_distinct_input_output_and_cache_prices() -> None:
    pricing = ModelPricing(input=2, output=10, cache_read=0.1, cache_write=2.5)

    assert pricing.estimate(10000, 2000, 1000, warm=False) == pytest.approx(0.04)
    assert pricing.estimate(10000, 2000, 1000, warm=True) == pytest.approx(0.016)


def test_missing_cache_prices_use_standard_input_rate() -> None:
    pricing = ModelPricing(input=2, output=10)

    assert pricing.estimate(10000, 2000, 1000, warm=True) == pytest.approx(0.034)
    assert pricing.estimate(10000, 2000, 1000, warm=False) == pytest.approx(0.034)


def test_long_context_surcharge_applies_to_entire_request_including_cache() -> None:
    pricing = ModelPricing(
        input=2,
        output=10,
        cache_read=0.1,
        cache_write=2.5,
        long_context_threshold=10000,
        long_context_input_multiplier=2,
        long_context_output_multiplier=1.5,
    )

    assert pricing.estimate(9000, 1000, 1000, warm=True) == pytest.approx(0.0134)
    assert pricing.estimate(10000, 1000, 1000, warm=True) == pytest.approx(0.022)
    assert pricing.estimate(10000, 1000, 1000, warm=False) == pytest.approx(0.07)


@pytest.mark.parametrize("price", [-1, float("nan"), float("inf")])
def test_invalid_prices_are_rejected(price: float) -> None:
    with pytest.raises(ValueError, match="finite, nonnegative"):
        ModelPricing(input=price, output=10)
