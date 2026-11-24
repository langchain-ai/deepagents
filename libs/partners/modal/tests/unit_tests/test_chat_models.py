"""Test chat model integration."""

from __future__ import annotations

import json
from contextlib import nullcontext

import pytest
from langchain_core.load import dumpd
from langchain_core.utils._gateway import langsmith_gateway
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import BaseModel as PydanticBaseModel
from pydantic import SecretStr

from langchain_modal.chat_models import ChatModal

MODEL_NAME = "my-endpoint.us-west.modal.direct"


class TestChatModalUnit(ChatModelUnitTests):
    """Standard unit tests for `ChatModal` chat model."""

    @property
    def chat_model_class(self) -> type[ChatModal]:
        """Chat model class being tested."""
        return ChatModal

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        """Parameters to initialize from environment variables."""
        return (
            {
                "MODAL_PROXY_TOKEN": "api_key",
                "MODAL_BASE_URL": "api_base",
            },
            {
                "model": MODEL_NAME,
            },
            {
                "api_key": "api_key",
                "api_base": "api_base",
            },
        )

    @property
    def chat_model_params(self) -> dict:
        """Parameters to create chat model instance for testing."""
        return {
            "model": MODEL_NAME,
            "api_key": "api_key",
        }

    def get_chat_model(self) -> ChatModal:
        """Get a chat model instance for testing."""
        return ChatModal(**self.chat_model_params)


def test_base_url_alias() -> None:
    """Test that `base_url` is accepted as an alias for `api_base`."""
    chat_model = ChatModal(
        model=MODEL_NAME,
        api_key=SecretStr("api_key"),
        base_url="http://example.test/v1",
    )
    assert chat_model.api_base == "http://example.test/v1"


def test_metadata_versions() -> None:
    """Test that metadata reports the correct version info."""
    llm = ChatModal(model=MODEL_NAME, api_key=SecretStr("test_key"))
    assert llm.metadata is not None
    versions = llm.metadata["lc_versions"]
    assert "langchain-core" in versions
    assert "langchain-modal" in versions
    assert "langchain-openai" in versions


# --- LangSmith gateway ---


@pytest.fixture
def _gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "LANGSMITH_GATEWAY",
        "LANGSMITH_GATEWAY_API_KEY",
        "LANGSMITH_API_KEY",
        "MODAL_BASE_URL",
        "MODAL_PROXY_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.usefixtures("_gateway_env")
def test_langsmith_gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """`LANGSMITH_GATEWAY` routes to the gateway with the gateway key."""
    monkeypatch.setenv("LANGSMITH_GATEWAY", "true")
    monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-key")
    llm = ChatModal(model=MODEL_NAME)
    assert llm.api_base == "https://gateway.smith.langchain.com/modal/v1"
    assert llm.api_key is not None
    assert llm.api_key.get_secret_value() == "gateway-key"


@pytest.mark.usefixtures("_gateway_env")
def test_langsmith_gateway_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the gateway, the provider key and default URL are used."""
    monkeypatch.setenv("LANGSMITH_GATEWAY_API_KEY", "gateway-key")
    monkeypatch.setenv("MODAL_PROXY_TOKEN", "provider-key")
    llm = ChatModal(model=MODEL_NAME)
    assert llm.api_base == "https://inference.us-west.modal.direct/v1"
    assert llm.api_key is not None
    assert llm.api_key.get_secret_value() == "provider-key"


@pytest.mark.usefixtures("_gateway_env")
def test_langsmith_gateway_provider_env_base_url_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider base URL env var wins over `LANGSMITH_GATEWAY`."""
    monkeypatch.setenv("LANGSMITH_GATEWAY", "true")
    monkeypatch.setenv("MODAL_BASE_URL", "https://proxy.example.com/v1")
    monkeypatch.setenv("MODAL_PROXY_TOKEN", "provider-key")
    llm = ChatModal(model=MODEL_NAME)
    assert llm.api_base == "https://proxy.example.com/v1"
    assert llm.api_key is not None
    assert llm.api_key.get_secret_value() == "provider-key"


@pytest.mark.usefixtures("_gateway_env")
def test_langsmith_gateway_context_overrides_explicit_values() -> None:
    """`langsmith_gateway` overrides an explicit base URL and API key."""
    gateway = pytest.importorskip("langchain_core.utils._gateway")
    if not hasattr(gateway, "langsmith_gateway"):
        pytest.skip("Requires a langchain-core release with `langsmith_gateway`.")
    with gateway.langsmith_gateway(api_key="service-key"):
        llm = ChatModal(
            model=MODEL_NAME,
            base_url="https://other.example.com/v1",
            api_key=SecretStr("user-key"),
        )
    assert llm.api_base == "https://gateway.smith.langchain.com/modal/v1"
    assert llm.api_key is not None
    assert llm.api_key.get_secret_value() == "service-key"


def test_langsmith_gateway_context_headers() -> None:
    """Gateway headers reach the client but are never serialized."""
    with langsmith_gateway(api_key="placeholder", headers={"X-Service-Key": "svc-key"}):
        llm = ChatModal(
            model=MODEL_NAME,
            api_key=SecretStr("user-key"),
            default_headers={"x-service-key": "forged", "X-Custom": "kept"},
        )
    sent = llm.root_client.default_headers
    assert sent["X-Service-Key"] == "svc-key"
    assert sent["X-Custom"] == "kept"
    assert "svc-key" not in json.dumps(dumpd(llm))


@pytest.mark.usefixtures("_gateway_env")
@pytest.mark.parametrize(
    "initial_headers", [None, {}, {"X-Service-Key": "original-secret"}]
)
def test_langsmith_gateway_headers_survive_revalidation(
    initial_headers: dict[str, str] | None,
) -> None:
    """Revalidation preserves the clients' original gateway header snapshot."""
    context = (
        langsmith_gateway(api_key="original-api-secret", headers=initial_headers)
        if initial_headers is not None
        else nullcontext()
    )
    with context:
        llm = ChatModal(model=MODEL_NAME, api_key=SecretStr("user-key"))
    original_clients = llm.root_client, llm.root_async_client
    headers = {"x-service-key": "forged", "X-Other-Key": "kept"}
    with langsmith_gateway(
        api_key="other-api-secret", headers={"X-Other-Key": "other-secret"}
    ):
        assert ChatModal.model_validate(llm) is llm

        class Wrapper(PydanticBaseModel):
            model: ChatModal

        assert Wrapper(model=llm).model is llm
    assert (llm.root_client, llm.root_async_client) == original_clients
    assert llm._langsmith_gateway_headers == (initial_headers or {})  # noqa: SLF001
    expected_headers = {"X-Other-Key": "kept"} if initial_headers else headers
    payload = llm._get_request_payload("hello", extra_headers=headers)  # noqa: SLF001
    assert payload["extra_headers"] == expected_headers
    for client in original_clients:
        assert client.default_headers.get("X-Service-Key") == (
            (initial_headers or {}).get("X-Service-Key")
        )
        assert "X-Other-Key" not in client.default_headers
