"""Modal chat models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Self

import openai
from langchain_core.utils import from_env, secret_from_env
from langchain_core.utils._gateway import (
    _apply_gateway_config,
    _gateway_headers,
    _with_gateway_headers,
)
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator

from langchain_modal._version import __version__

if TYPE_CHECKING:
    from langchain_core.language_models import LangSmithParams

DEFAULT_API_BASE = "https://inference.us-west.modal.direct/v1"


class ChatModal(BaseChatOpenAI):
    """Call Modal Endpoints with `MODAL_PROXY_TOKEN` and a hostname as `model`."""

    model_name: str = Field(alias="model")
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("MODAL_PROXY_TOKEN", default=None),
    )
    api_base: str = Field(
        alias="base_url",
        default_factory=from_env("MODAL_BASE_URL", default=DEFAULT_API_BASE),
    )

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        """Return the chat model type."""
        return "chat-modal"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """Map the provider credential to its environment variable."""
        return {"api_key": "MODAL_PROXY_TOKEN"}

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: object,
    ) -> LangSmithParams:
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "modal"
        return params

    @model_validator(mode="after")
    def _set_modal_version(self) -> Self:
        """Include the integration version in tracing metadata."""
        self._add_version("langchain-modal", __version__)
        return self

    @model_validator(mode="before")
    @classmethod
    def _apply_modal_gateway_config(
        cls, values: dict[str, object] | Self
    ) -> dict[str, object] | Self:
        """Apply provider-specific gateway configuration before resolving fields."""
        if isinstance(values, dict):
            _apply_gateway_config(
                values,
                cls,
                base_url_field="api_base",
                api_key_field="api_key",
                provider_path="modal/v1",
                base_url_env="MODAL_BASE_URL",
                api_key_env="MODAL_PROXY_TOKEN",
                default_base_url=DEFAULT_API_BASE,
            )
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Initialize Modal clients without falling back to OpenAI credentials."""
        if self._langsmith_gateway_headers is None:
            self._langsmith_gateway_headers = _gateway_headers()
        if self.api_base == DEFAULT_API_BASE and not (
            self.api_key and self.api_key.get_secret_value()
        ):
            msg = "MODAL_PROXY_TOKEN must be set when using the shared Modal endpoint."
            raise ValueError(msg)
        token = self.api_key.get_secret_value() if self.api_key else "EMPTY"
        timeout = self.request_timeout
        if isinstance(timeout, tuple):
            timeout = openai.Timeout(timeout[1], connect=timeout[0])
        headers = _with_gateway_headers(
            self.default_headers, self._langsmith_gateway_headers
        )
        retries = (
            self.max_retries
            if self.max_retries is not None
            else openai.DEFAULT_MAX_RETRIES
        )
        if not self.client:
            self.root_client = openai.OpenAI(
                api_key=token,
                base_url=self.api_base,
                timeout=timeout,
                max_retries=retries,
                default_headers=headers,
                default_query=self.default_query,
                http_client=self.http_client,
            )
            self.client = self.root_client.chat.completions
        if not self.async_client:
            self.root_async_client = openai.AsyncOpenAI(
                api_key=token,
                base_url=self.api_base,
                timeout=timeout,
                max_retries=retries,
                default_headers=headers,
                default_query=self.default_query,
                http_client=self.http_async_client,
            )
            self.async_client = self.root_async_client.chat.completions
        return self
