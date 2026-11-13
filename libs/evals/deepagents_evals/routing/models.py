"""Supplied model descriptions and token prices, independent of routing policy."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, kw_only=True)
class ModelPricing:
    """USD per million tokens for the candidate's configured service tier.

    Args:
        input: Uncached input price.
        output: Output price, including billed reasoning tokens.
        cache_read: Cached input price; omitted means the input price applies.
        cache_write: Cache creation price; omitted means the input price applies.
        long_context_threshold: Input length above which surcharges apply.
        long_context_input_multiplier: Surcharge on all input and cache prices.
        long_context_output_multiplier: Surcharge on the output price.
    """

    input: float
    output: float
    cache_read: float | None = None
    cache_write: float | None = None
    long_context_threshold: int | None = None
    long_context_input_multiplier: float = 1.0
    long_context_output_multiplier: float = 1.0

    def __post_init__(self) -> None:
        """Reject prices that cannot be compared meaningfully."""
        prices = (self.input, self.output, self.cache_read, self.cache_write)
        multipliers = (self.long_context_input_multiplier, self.long_context_output_multiplier)
        if any(p is not None and (not isfinite(p) or p < 0) for p in prices):
            msg = "Model prices must be finite, nonnegative USD per million tokens."
            raise ValueError(msg)
        if any(not isfinite(m) or m <= 0 for m in multipliers):
            msg = "Long-context price multipliers must be finite and positive."
            raise ValueError(msg)
        if self.long_context_threshold is not None and self.long_context_threshold <= 0:
            msg = "long_context_threshold must be positive."
            raise ValueError(msg)

    def estimate(self, prefix: int, delta: int, output: int, *, warm: bool) -> float:
        """Estimate a call using the router's existing cached-prefix accounting.

        Args:
            prefix: Previously measured prompt tokens.
            delta: Estimated new prompt tokens.
            output: Estimated billed output tokens.
            warm: Whether the candidate's cached prefix is still warm.

        Returns:
            Estimated call cost in USD.
        """
        read = self.input if self.cache_read is None else self.cache_read
        write = self.input if self.cache_write is None else self.cache_write
        input_cost = prefix * read + delta * write if warm else (prefix + delta) * write
        output_cost = output * self.output
        if self.long_context_threshold is not None and prefix + delta > self.long_context_threshold:
            input_cost *= self.long_context_input_multiplier
            output_cost *= self.long_context_output_multiplier
        return (input_cost + output_cost) * 1e-6


@dataclass(frozen=True, kw_only=True)
class ModelDetails:
    """Candidate metadata that does not change the model-independent assessment.

    Args:
        description: Human-readable description of this configured candidate.
        pricing: Token prices. Missing prices trigger an explicit routing fallback.
    """

    description: str = ""
    pricing: ModelPricing | None = None
