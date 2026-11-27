"""Test chat model integration."""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.load import dumpd
from langchain_core.tracers.context import collect_runs
from langchain_core.utils._gateway import langsmith_gateway
from langchain_tests.unit_tests import ChatModelUnitTests
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


@pytest.fixture(autouse=True)
def _gateway_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "LANGSMITH_GATEWAY",
        "LANGSMITH_GATEWAY_API_KEY",
        "LANGSMITH_API_KEY",
        "MODAL_BASE_URL",
        "MODAL_PROXY_TOKEN",
        "OPENAI_API_KEY",
        "OPENAI_API_BASE",
        "OPENAI_BASE_URL",
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
    clients = llm.root_client, llm.root_async_client
    with langsmith_gateway(
        api_key="other-key", headers={"X-Service-Key": "other-secret"}
    ):
        assert ChatModal.model_validate(llm) is llm
    assert (llm.root_client, llm.root_async_client) == clients
    sent = llm.root_client.default_headers
    assert sent["X-Service-Key"] == "svc-key"
    assert sent["X-Custom"] == "kept"
    assert "svc-key" not in json.dumps(dumpd(llm))


def test_missing_modal_token_does_not_use_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    with pytest.raises(ValueError, match="MODAL_PROXY_TOKEN must be set"):
        ChatModal(model=MODEL_NAME)


def test_explicit_credentials_override_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODAL_PROXY_TOKEN", "environment-token")
    monkeypatch.setenv("MODAL_BASE_URL", "https://environment.example/v1")
    model = ChatModal(
        model=MODEL_NAME,
        api_key=SecretStr("explicit-token"),
        base_url="https://explicit.example/v1",
    )
    assert str(model.root_client.base_url) == "https://explicit.example/v1/"
    assert model.root_client.api_key == "explicit-token"
    assert model.root_async_client.api_key == "explicit-token"
    assert "explicit-token" not in repr(model)
    assert "explicit-token" not in json.dumps(dumpd(model))


def test_custom_endpoint_does_not_receive_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "unrelated-openai-key")
    model = ChatModal(model=MODEL_NAME, base_url="https://custom.example/v1")
    assert model.root_client.api_key == "EMPTY"
    assert model.root_async_client.api_key == "EMPTY"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
async def test_completion_requests(*, asynchronous: bool, streaming: bool) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert (
            request.url == "https://inference.us-west.modal.direct/v1/chat/completions"
        )
        assert request.headers["authorization"] == "Bearer modal-test-token"
        payload = json.loads(request.content)
        assert payload["model"] == MODEL_NAME
        assert payload["stream"] is streaming
        assert payload["temperature"] == 0
        if streaming:
            chunks = [
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": MODEL_NAME,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "hello"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
            events = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=events + "data: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            },
        )

    transport = httpx.MockTransport(respond)
    with httpx.Client(transport=transport) as sync_client:
        async with httpx.AsyncClient(transport=transport) as async_client:
            model = ChatModal(
                model=MODEL_NAME,
                api_key=SecretStr("modal-test-token"),
                temperature=0,
                http_client=sync_client,
                http_async_client=async_client,
            )
            with collect_runs() as runs:
                if streaming:
                    chunks = (
                        [chunk async for chunk in model.astream("hello")]
                        if asynchronous
                        else list(model.stream("hello"))
                    )
                    assert "".join(chunk.text for chunk in chunks) == "hello"
                else:
                    message = (
                        await model.ainvoke("hello")
                        if asynchronous
                        else model.invoke("hello")
                    )
                    assert message.content == "hello"
                    assert message.usage_metadata is not None
                    assert (
                        message.usage_metadata.items()
                        >= {
                            "input_tokens": 2,
                            "output_tokens": 1,
                            "total_tokens": 3,
                        }.items()
                    )
            assert runs.traced_runs[0].extra["metadata"]["ls_provider"] == "modal"
