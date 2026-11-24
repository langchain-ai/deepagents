"""Modal chat models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

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
    """Modal chat model integration for deployed Modal Endpoints.

    Modal serves deployed models through an OpenAI-compatible API. The `model`
    parameter is the Endpoint's hostname, e.g. `'my-endpoint.us-west.modal.direct'`.

    Setup:
        Install `langchain-modal` and set environment variable `MODAL_PROXY_TOKEN`
        to a Modal [proxy token](https://modal.com/docs/guide/endpoints#proxy-tokens)
        in its combined `wk-<id>.ws-<secret>` form.

        ```bash
        pip install -U langchain-modal
        export MODAL_PROXY_TOKEN="wk-<id>.ws-<secret>"
        ```

    Key init args — completion params:
        model:
            Hostname of the Modal Endpoint to call, e.g.
            `'my-endpoint.us-west.modal.direct'`. List reachable Endpoints with
            `GET https://inference.us-west.modal.direct/v1/models` and your
            proxy token.
        temperature:
            Sampling temperature.
        max_tokens:
            Max number of tokens to generate.

    Key init args — client params:
        timeout:
            Timeout for requests.
        max_retries:
            Max number of retries.
        api_key:
            Modal proxy token. If not passed in will be read from env var
            `MODAL_PROXY_TOKEN`.
        base_url:
            Modal inference base URL. If not passed in will be read from env var
            `MODAL_BASE_URL`, defaulting to the Shared Endpoint router
            `https://inference.us-west.modal.direct/v1`. Point it at a
            Dedicated Endpoint of the form
            `https://<workspace>-<env>--<app>.<region>.modal.direct/v1` to call
            one directly.

    Instantiate:
        ```python
        from langchain_modal import ChatModal

        model = ChatModal(
            model="my-endpoint.us-west.modal.direct",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # api_key="wk-...ws-...",
            # other params...
        )
        ```

    Invoke:
        ```python
        messages = [
            (
                "system",
                "You are a helpful translator. Translate the user sentence to French.",
            ),
            ("human", "I love programming."),
        ]
        model.invoke(messages)
        ```

    Stream:
        ```python
        for chunk in model.stream(messages):
            print(chunk.text, end="")
        ```
        ```python
        stream = model.stream(messages)
        full = next(stream)
        for chunk in stream:
            full += chunk
        full
        ```

    Async:
        ```python
        await model.ainvoke(messages)

        # stream:
        # async for chunk in (await model.astream(messages))

        # batch:
        # await model.abatch([messages])
        ```

    Tool calling:
        ```python
        from pydantic import BaseModel, Field


        class GetWeather(BaseModel):
            '''Get the current weather in a given location'''

            location: str = Field(
                ..., description="The city and state, e.g. San Francisco, CA"
            )


        class GetPopulation(BaseModel):
            '''Get the current population in a given location'''

            location: str = Field(
                ..., description="The city and state, e.g. San Francisco, CA"
            )


        model_with_tools = model.bind_tools([GetWeather, GetPopulation])
        ai_msg = model_with_tools.invoke(
            "Which city is hotter today and which is bigger: LA or NY?"
        )
        ai_msg.tool_calls
        ```

        See `ChatModal.bind_tools()` method for more.

    Structured output:
        ```python
        from typing import Optional

        from pydantic import BaseModel, Field


        class Joke(BaseModel):
            '''Joke to tell user.'''

            setup: str = Field(description="The setup of the joke")
            punchline: str = Field(description="The punchline to the joke")
            rating: int | None = Field(
                description="How funny the joke is, from 1 to 10"
            )


        structured_model = model.with_structured_output(Joke)
        structured_model.invoke("Tell me a joke about cats")
        ```

        See `ChatModal.with_structured_output()` for more.

    Token usage:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.usage_metadata
        ```
        ```python
        {"input_tokens": 28, "output_tokens": 5, "total_tokens": 33}
        ```

    Response metadata:
        ```python
        ai_msg = model.invoke(messages)
        ai_msg.response_metadata
        ```
    """  # noqa: E501

    model_name: str = Field(alias="model")
    """Hostname of the Modal Endpoint to call."""
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("MODAL_PROXY_TOKEN", default=None),
    )
    """Modal proxy token"""
    api_base: str = Field(
        alias="base_url",
        default_factory=from_env("MODAL_BASE_URL", default=DEFAULT_API_BASE),
    )
    """Modal inference base URL.

    Automatically read from env variable `MODAL_BASE_URL` if not provided.
    """

    model_config = ConfigDict(populate_by_name=True)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-modal"

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids."""
        return {"api_key": "MODAL_PROXY_TOKEN"}

    def _get_ls_params(
        self,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> LangSmithParams:
        ls_params = super()._get_ls_params(stop=stop, **kwargs)
        ls_params["ls_provider"] = "modal"
        return ls_params

    @model_validator(mode="after")
    def _set_modal_version(self) -> Self:
        """Set package version in metadata.

        Named uniquely to avoid shadowing `BaseChatOpenAI._set_openai_chat_version`;
        Pydantic replaces same-named validators rather than chaining them.
        """
        self._add_version("langchain-modal", __version__)
        return self

    @model_validator(mode="before")
    @classmethod
    def _apply_modal_gateway_config(cls, values: Any) -> Any:
        """Resolve the base URL and API key, applying LangSmith gateway settings.

        Routes through the gateway when `LANGSMITH_GATEWAY` is set or the model is
        built inside `langsmith_gateway`. See `langchain_core.utils._gateway` for
        precedence.
        """
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
        """Validate necessary environment vars and client params."""
        if self._langsmith_gateway_headers is None:
            self._langsmith_gateway_headers = _gateway_headers()
        if self.api_base == DEFAULT_API_BASE and not (
            self.api_key and self.api_key.get_secret_value()
        ):
            msg = "If using default api base, MODAL_PROXY_TOKEN must be set."
            raise ValueError(msg)
        client_params: dict = {
            k: v
            for k, v in {
                "api_key": self.api_key.get_secret_value() if self.api_key else None,
                "base_url": self.api_base,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": _with_gateway_headers(
                    self.default_headers, self._langsmith_gateway_headers
                ),
                "default_query": self.default_query,
            }.items()
            if v is not None
        }

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
            self.client = self.root_client.chat.completions
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
            self.async_client = self.root_async_client.chat.completions
        return self
