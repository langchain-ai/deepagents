from unittest.mock import MagicMock, patch

import pytest

from deepagents.middleware._prompt_caching import (
    _create_anthropic_vertex_prompt_caching_middleware,
)


def test_creates_anthropic_vertex_prompt_caching_middleware() -> None:
    middleware = MagicMock()
    middleware_cls = MagicMock(return_value=middleware)
    module = MagicMock(AnthropicVertexPromptCachingMiddleware=middleware_cls)

    with patch("deepagents.middleware._prompt_caching.import_module", return_value=module):
        result = _create_anthropic_vertex_prompt_caching_middleware()

    assert result is middleware
    middleware_cls.assert_called_once_with(unsupported_model_behavior="ignore")


def test_anthropic_vertex_prompt_caching_is_optional() -> None:
    error = ModuleNotFoundError(name="langchain_google_vertexai.middleware")

    with patch("deepagents.middleware._prompt_caching.import_module", side_effect=error):
        assert _create_anthropic_vertex_prompt_caching_middleware() is None


def test_anthropic_vertex_prompt_caching_preserves_unrelated_import_errors() -> None:
    with (
        patch(
            "deepagents.middleware._prompt_caching.import_module",
            side_effect=ImportError(name="missing_transitive"),
        ),
        pytest.raises(ImportError),
    ):
        _create_anthropic_vertex_prompt_caching_middleware()
