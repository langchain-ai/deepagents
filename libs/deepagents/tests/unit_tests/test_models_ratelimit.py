from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter

from deepagents._models import resolve_model


class MockChatModel(BaseChatModel):
    rate_limiter: InMemoryRateLimiter | None = None

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        pass

    async def _agenerate(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        pass

    @property
    def _llm_type(self) -> str:
        return "mock"


def test_resolve_model_adds_rate_limiter():
    """Verify that resolve_model adds a default rate limiter if missing."""
    model = MockChatModel()
    assert model.rate_limiter is None

    resolved = resolve_model(model)
    assert isinstance(resolved.rate_limiter, InMemoryRateLimiter)
    assert resolved.rate_limiter.requests_per_second == 1.0


def test_resolve_model_preserves_existing_rate_limiter():
    """Verify that resolve_model doesn't overwrite an existing rate limiter."""
    model = MockChatModel()
    existing = InMemoryRateLimiter(requests_per_second=5.0)
    model.rate_limiter = existing

    resolved = resolve_model(model)
    assert resolved.rate_limiter is existing
    assert resolved.rate_limiter.requests_per_second == 5.0
