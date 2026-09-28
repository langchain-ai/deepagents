from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.messages.utils import count_tokens_approximately
from pydantic import Field

from deepagents_talon.background import _IN_SUBAGENT
from deepagents_talon.channels.base import ChannelExposure
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, ChannelMessage
from deepagents_talon.model_selection import (
    ACTIVE_MODEL,
    SelectedModelSummarization,
    discover_models,
)
from deepagents_talon.runtime import DeepAgentRuntime, _resolve_model_from_env
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config, _wait_for_request

if TYPE_CHECKING:
    from pathlib import Path


class ReplyModel(FakeMessagesListChatModel):
    def bind_tools(self, _tools, **_kwargs: object):
        return self


class SmartAgent(BlockingAgent):
    def __init__(self, smart_model: str | None = None) -> None:
        super().__init__()
        self.smart_model = smart_model
        self.prepared: list[str | None] = []

    async def select_smart_model(self, spec: str | None) -> bool:
        if spec is not None and spec not in {"test:helper", "test:alt"}:
            return False
        self.prepared.append(spec)
        self.smart_model = spec
        return True


class SelectableAgent(BlockingAgent):
    default_model = "test:primary"

    def __init__(self) -> None:
        super().__init__()
        self.catalog = {"test": ["primary", "alt"], "other": ["one", "two"]}
        self.prepared: list[str] = []

    async def model_catalog(self) -> dict[str, list[str]]:
        return self.catalog

    async def select_model(self, spec: str) -> bool:
        provider, _, name = spec.partition(":")
        if name not in self.catalog.get(provider, ()):
            return False
        self.prepared.append(spec)
        return True


def _operator_channel() -> RecordingChannel:
    channel = RecordingChannel()
    channel.config = SimpleNamespace(exposure=ChannelExposure(operator_ids=frozenset({"op"})))
    return channel


def _from(sender: str, text: str, chat: str = "chat") -> ChannelMessage:
    return ChannelMessage(conversation_id=chat, text=text, sender_id=sender)


async def _turn_model(host: TalonHost, agent: BlockingAgent, channel, chat: str) -> str | None:
    await host.receive_message(channel, _from("op", f"hello {chat}", chat))
    await _wait_for_request(agent, f"hello {chat}")
    return next(r.model for r in agent.requests if r.text == f"hello {chat}")


async def test_smart_model_command_persists_across_chats_and_restart(tmp_path: Path) -> None:
    config = _config(tmp_path)
    agent, channel = SmartAgent(), _operator_channel()
    host = TalonHost(config=config, agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("op", "/smart-model test:helper"))
        assert channel.sent[-1] == ("chat", "Smart model now uses test:helper across chats.")
        await host.receive_message(channel, _from("op", "/smart-model", "elsewhere"))
        assert "Smart model: test:helper" in channel.sent[-1][1]
        assert config.smart_model_state_path.stat().st_mode & 0o777 == 0o600
        assert json.loads(config.smart_model_state_path.read_text()) == {"model": "test:helper"}
        await host.receive_message(channel, _from("op", "/smart-model off"))
        assert agent.smart_model is None
        assert json.loads(config.smart_model_state_path.read_text()) == {"model": None}
    finally:
        await host.stop()

    restarted = SmartAgent("test:alt")
    host = TalonHost(config=config, agent=restarted, channels=[channel])
    await host.start()
    try:
        assert restarted.smart_model is None
        await host.receive_message(channel, _from("op", "/smart-model default"))
        assert json.loads(config.smart_model_state_path.read_text()) == {}
        assert restarted.smart_model is None
    finally:
        await host.stop()


async def test_smart_model_override_survives_restart(tmp_path: Path) -> None:
    config = _config(tmp_path)
    agent, channel = SmartAgent(), _operator_channel()
    host = TalonHost(config=config, agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("op", "/smart-model test:helper"))
    finally:
        await host.stop()
    restarted = SmartAgent()
    host = TalonHost(config=config, agent=restarted, channels=[channel])
    await host.start()
    try:
        assert restarted.smart_model == "test:helper"
        await host.receive_message(channel, _from("op", "/smart-model default"))
        assert restarted.smart_model is None
        assert json.loads(config.smart_model_state_path.read_text()) == {}
    finally:
        await host.stop()


async def test_smart_model_default_uses_configured_model(tmp_path: Path) -> None:
    config = _config(tmp_path, {"DEEPAGENTS_TALON_HELP_MODEL": "test:alt"})
    agent, channel = SmartAgent("test:alt"), _operator_channel()
    host = TalonHost(config=config, agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("op", "/smart-model test:helper"))
        await host.receive_message(channel, _from("op", "/smart-model default"))
        assert agent.smart_model == "test:alt"
        assert json.loads(config.smart_model_state_path.read_text()) == {}
    finally:
        await host.stop()


async def test_smart_model_default_uses_process_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPAGENTS_TALON_HELP_MODEL", "test:alt")
    config = _config(tmp_path)
    agent, channel = SmartAgent("test:alt"), _operator_channel()
    host = TalonHost(config=config, agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("op", "/smart-model off"))
        await host.receive_message(channel, _from("op", "/smart-model default"))
        assert agent.smart_model == "test:alt"
        assert json.loads(config.smart_model_state_path.read_text()) == {}
    finally:
        await host.stop()


async def test_smart_model_save_failure_restores_previous_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent, channel = SmartAgent("test:alt"), _operator_channel()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:

        def fail_save(_path: Path, _state: object) -> None:
            message = "not writable"
            raise OSError(message)

        monkeypatch.setattr("deepagents_talon.host._write_json_state", fail_save)
        await host.receive_message(channel, _from("op", "/smart-model test:helper"))
        assert channel.sent[-1][1] == "Could not save the smart model selection. Check Talon logs."
        assert agent.smart_model == "test:alt"
    finally:
        await host.stop()


async def test_smart_model_save_failure_reports_failed_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class UnavailablePreviousAgent(SmartAgent):
        async def select_smart_model(self, spec: str | None) -> bool:
            if spec == "test:alt":
                return False
            return await super().select_smart_model(spec)

    agent, channel = UnavailablePreviousAgent("test:alt"), _operator_channel()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:

        def fail_save(_path: Path, _state: object) -> None:
            msg = "not writable"
            raise OSError(msg)

        monkeypatch.setattr("deepagents_talon.host._write_json_state", fail_save)
        await host.receive_message(channel, _from("op", "/smart-model test:helper"))
        assert channel.sent[-1][1] == (
            "Could not save or restore the smart model selection. Check Talon logs."
        )
        assert agent.smart_model == "test:helper"
    finally:
        await host.stop()


async def test_smart_model_command_rejects_nonoperator_and_unknown_model(tmp_path: Path) -> None:
    agent, channel = SmartAgent(), _operator_channel()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("stranger", "/smart-model test:helper"))
        assert channel.sent[-1][1] == "Only an operator can change the smart model."
        await host.receive_message(channel, _from("op", "/smart-model test:unknown"))
        assert channel.sent[-1][1] == "Not an available model. Send /model to list them."
        assert agent.prepared == []
    finally:
        await host.stop()


async def test_switch_applies_to_new_conversations(tmp_path: Path) -> None:
    agent, channel = SelectableAgent(), _operator_channel()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("op", "/model other:two"))

        assert channel.sent[-1] == ("chat", "All chats now use other:two.")
        assert await _turn_model(host, agent, channel, "chat") == "other:two"
        assert await _turn_model(host, agent, channel, "elsewhere") == "other:two"
        await host.receive_message(channel, _from("op", "/model other:one", "elsewhere"))
        assert await _turn_model(host, agent, channel, "third-thread") == "other:one"
    finally:
        await host.stop()


async def test_non_operator_cannot_switch(tmp_path: Path) -> None:
    agent, channel = SelectableAgent(), _operator_channel()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("stranger", "/model other:two"))

        assert channel.sent[-1] == ("chat", "Only an operator can change the model.")
        assert agent.prepared == []
        assert await _turn_model(host, agent, channel, "chat") is None
    finally:
        await host.stop()


@pytest.mark.parametrize("spec", ["other:three", "openai:gpt-4o", "other:two extra"])
async def test_unknown_model_is_rejected_without_echoing_it(tmp_path: Path, spec: str) -> None:
    agent, channel = SelectableAgent(), _operator_channel()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("op", f"/model {spec}"))

        assert channel.sent[-1] == ("chat", "Not an available model. Send /model to list them.")
        assert await _turn_model(host, agent, channel, "chat") is None
    finally:
        await host.stop()


async def test_selection_survives_new_and_restart(tmp_path: Path) -> None:
    agent, channel = SelectableAgent(), _operator_channel()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("op", "/model other:one"))
        await host.receive_message(channel, _from("op", "/new"))
        assert await _turn_model(host, agent, channel, "chat") == "other:one"
    finally:
        await host.stop()

    restarted, agent = (
        TalonHost(config=_config(tmp_path), agent=SelectableAgent(), channels=[channel]),
        None,
    )
    agent = restarted.agent
    await restarted.start()
    try:
        assert await _turn_model(restarted, agent, channel, "new-thread") == "other:one"
    finally:
        await restarted.stop()


async def test_default_clears_the_selection(tmp_path: Path) -> None:
    agent, channel = SelectableAgent(), _operator_channel()
    config = _config(tmp_path)
    host = TalonHost(config=config, agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("op", "/model other:one"))
        await host.receive_message(channel, _from("op", "/model default"))

        assert channel.sent[-1] == ("chat", "All chats now use test:primary.")
        assert json.loads(config.model_state_path.read_text()) == {}
        assert await _turn_model(host, agent, channel, "new-thread") is None
    finally:
        await host.stop()


async def test_listing_shows_current_model_and_providers_to_anyone(tmp_path: Path) -> None:
    agent, channel = SelectableAgent(), _operator_channel()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("stranger", "/model"))
        overview = channel.sent[-1][1]
        await host.receive_message(channel, _from("stranger", "/model other"))
        listing = channel.sent[-1][1]
    finally:
        await host.stop()

    assert overview.startswith("All chats use test:primary (default).")
    assert "other — 2 models" in overview
    assert "test — 2 models" in overview
    assert listing.splitlines()[:2] == ["other:one", "other:two"]
    assert "test:alt" not in listing
    assert agent.requests == []


@pytest.fixture
def discovered(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    catalog = {"test": ["alt"], "nokey": ["x"]}
    credentials = {"test": "TEST_API_KEY", "nokey": "NOKEY_API_KEY"}
    monkeypatch.setattr("deepagents_talon.model_selection.get_available_models", lambda: catalog)
    monkeypatch.setattr("deepagents_talon.model_selection.get_credential_env_var", credentials.get)
    monkeypatch.setenv("TEST_API_KEY", "key")
    monkeypatch.delenv("NOKEY_API_KEY", raising=False)
    return catalog


@pytest.fixture
def built(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    models = {
        "test:primary": ReplyModel(responses=[AIMessage(content="from primary")]),
        "test:alt": ReplyModel(responses=[AIMessage(content="from alt")]),
    }
    calls: list[str] = []

    def resolve(model: str, *_args: object, **_kwargs: object) -> ReplyModel:
        calls.append(model)
        return models[model]

    monkeypatch.setattr("deepagents_talon.runtime._resolve_model_from_env", resolve)
    return calls


def _runtime(tmp_path: Path, **kwargs: object) -> DeepAgentRuntime:
    return DeepAgentRuntime(
        model="test:primary",
        assistant_dir=tmp_path,
        include_web_tools=False,
        skills=(),
        memory=(),
        **kwargs,
    )


@pytest.mark.usefixtures("discovered")
async def test_runtime_answers_with_the_selected_model_built_once(
    tmp_path: Path, built: list[str]
) -> None:
    runtime = _runtime(tmp_path)
    await runtime.start()
    try:
        assert "test:alt" not in built
        first = await runtime.invoke(AgentRequest("a", "hi", model="test:alt"))
        second = await runtime.invoke(AgentRequest("c", "hi", model="test:alt"))
        default = await runtime.invoke(AgentRequest("b", "hi"))
    finally:
        await runtime.stop()

    assert (first.text, second.text, default.text) == ("from alt", "from alt", "from primary")
    assert built.count("test:alt") == 1


@pytest.mark.usefixtures("discovered")
async def test_runtime_ignores_a_selection_that_is_no_longer_available(
    tmp_path: Path, built: list[str]
) -> None:
    runtime = _runtime(tmp_path)
    await runtime.start()
    try:
        result = await runtime.invoke(AgentRequest("a", "hi", model="nokey:x"))
    finally:
        await runtime.stop()

    assert result.text == "from primary"
    assert "nokey:x" not in built


@pytest.mark.usefixtures("discovered")
@pytest.mark.parametrize("spec", ["nokey:x", "test:alt2", "test", "openai:gpt-4o", ":alt"])
async def test_only_discovered_credentialed_models_are_selectable(
    tmp_path: Path, built: list[str], spec: str
) -> None:
    runtime = _runtime(tmp_path)

    assert await runtime.select_model(spec) is False
    assert await runtime.select_model("test:alt") is True
    assert spec not in built


@pytest.mark.usefixtures("discovered")
async def test_catalog_lists_credentialed_providers_and_the_default(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)

    assert await runtime.model_catalog() == {"test": ["alt", "primary"]}


def test_gateway_catalog_lists_models_beyond_installed_profiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"id": "openai/gpt-6-luna", "object": "model"},
                    {"id": "openai/gpt-6-astra", "object": "model"},
                    {"id": "anthropic/claude-test", "object": "model"},
                    {"id": "openai/gpt-6-astra", "object": "model"},
                    {"id": "invalid id with spaces", "object": "model"},
                    {"id": "openai/non-chat", "supported_endpoints": ["/v1/systemone"]},
                ],
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr("deepagents_talon.model_selection.httpx.Client", lambda **_kwargs: client)
    monkeypatch.setattr(
        "deepagents_talon.model_selection.get_available_models",
        lambda: {"openai": ["gpt-6-luna"]},
    )
    catalog = discover_models(
        {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://gateway.smith.langchain.com/v1",
        }
    )
    assert catalog == {
        "openai": ["gpt-6-luna", "openai/gpt-6-luna", "openai/gpt-6-astra", "anthropic/claude-test"]
    }
    assert len(requests) == 1
    assert str(requests[0].url) == "https://gateway.smith.langchain.com/v1/models"
    assert requests[0].headers["Authorization"] == "Bearer test-key"


def test_gateway_flag_uses_langsmith_key_and_unified_model_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://gateway.smith.langchain.com/v1/models"
        assert request.headers["Authorization"] == "Bearer gateway-key"
        return httpx.Response(200, json={"object": "list", "data": [{"id": "openai/gpt-6-terra"}]})

    client = httpx.Client(transport=httpx.MockTransport(respond))
    monkeypatch.setattr("deepagents_talon.model_selection.httpx.Client", lambda **_kwargs: client)
    monkeypatch.setattr("deepagents_talon.model_selection.get_available_models", dict)
    assert discover_models(
        {"LANGSMITH_GATEWAY": "true", "LANGSMITH_GATEWAY_API_KEY": "gateway-key"}
    ) == {"openai": ["openai/gpt-6-terra"]}


def test_gateway_model_build_uses_unified_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    built: list[tuple[str, dict[str, object]]] = []
    model = ReplyModel(responses=[AIMessage(content="ok")])

    def build(spec: str, **kwargs: object) -> ReplyModel:
        built.append((spec, kwargs))
        return model

    monkeypatch.setattr("deepagents_talon.runtime.init_chat_model", build)
    env = {"LANGSMITH_GATEWAY": "true", "LANGSMITH_GATEWAY_API_KEY": "gateway-key"}
    assert _resolve_model_from_env("openai:openai/gpt-6-terra", env) is model
    assert len(built) == 1
    assert built[0][0] == "openai:openai/gpt-6-terra"
    assert built[0][1]["base_url"] == "https://gateway.smith.langchain.com/v1"
    assert built[0][1]["api_key"] == "gateway-key"
    assert built[0][1]["use_responses_api"] is False


@pytest.mark.parametrize(
    "base_url",
    [
        "https://gateway.smith.langchain.com.evil.test/v1",
        "http://gateway.smith.langchain.com/v1",
        "https://gateway.smith.langchain.com:444/v1",
        "https://gateway.smith.langchain.com/v1/other",
        "https://gateway.smith.langchain.com/v1?redirect=evil",
        "https://attacker@gateway.smith.langchain.com/v1",
    ],
)
def test_gateway_catalog_refuses_untrusted_urls(
    monkeypatch: pytest.MonkeyPatch, base_url: str
) -> None:
    def unexpected_client(**_kwargs: object) -> None:
        pytest.fail("unexpected gateway request")

    monkeypatch.setattr("deepagents_talon.model_selection.httpx.Client", unexpected_client)
    monkeypatch.setattr("deepagents_talon.model_selection.get_available_models", dict)
    assert discover_models({"OPENAI_BASE_URL": base_url, "OPENAI_API_KEY": "test-key"}) == {}


def test_gateway_catalog_requires_matching_model_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_client(**_kwargs: object) -> None:
        pytest.fail("unexpected gateway request")

    monkeypatch.setattr("deepagents_talon.model_selection.httpx.Client", unexpected_client)
    monkeypatch.setattr("deepagents_talon.model_selection.get_available_models", dict)
    assert discover_models({"OPENAI_BASE_URL": "https://gateway.smith.langchain.com/v1"}) == {}
    assert (
        discover_models(
            {
                "OPENAI_BASE_URL": "https://gateway.smith.langchain.com/v1",
                "LANGSMITH_GATEWAY_API_KEY": "test-key",
            }
        )
        == {}
    )


class GatedTranscriber:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def transcribe(self, _message: ChannelMessage) -> str:
        self.started.set()
        await self.release.wait()
        return "hello voice"


async def test_switch_during_a_started_turn_waits_for_the_next_turn(tmp_path: Path) -> None:
    agent, channel, transcriber = SelectableAgent(), _operator_channel(), GatedTranscriber()
    host = TalonHost(
        config=_config(tmp_path), agent=agent, channels=[channel], voice_transcriber=transcriber
    )
    await host.start()
    try:
        voice = ChannelMessage(
            conversation_id="chat",
            text="",
            sender_id="op",
            metadata={"media_type": "voice", "voice_path": "voice.ogg"},
        )
        await host.receive_message(channel, voice)
        await asyncio.wait_for(transcriber.started.wait(), timeout=1)
        await host.receive_message(channel, _from("op", "/model other:two"))
        transcriber.release.set()
        await _wait_for_request(agent, "hello voice")
    finally:
        await host.stop()

    assert channel.sent[0] == ("chat", "All chats now use other:two.")
    assert agent.requests[0].model is None


async def test_listing_flags_a_selection_that_is_no_longer_available(tmp_path: Path) -> None:
    agent, channel = SelectableAgent(), _operator_channel()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("op", "/model other:two"))
        agent.catalog = {"test": ["primary"]}
        await host.receive_message(channel, _from("op", "/model"))
    finally:
        await host.stop()

    assert channel.sent[-1][1].startswith(
        "All chats use test:primary (default). Its selected model other:two is unavailable."
    )


class RecordingModel(ReplyModel):
    seen: list[str] = Field(default_factory=list)
    input_sizes: list[int] = Field(default_factory=list)

    def _generate(self, messages, *args: object, **kwargs: object):
        self.seen.append(str(messages[-1].content))
        self.input_sizes.append(count_tokens_approximately(messages))
        return super()._generate(messages, *args, **kwargs)


def _sized(monkeypatch: pytest.MonkeyPatch, primary: int, alt: int) -> RecordingModel:
    models = {
        "test:primary": RecordingModel(
            responses=[AIMessage(content="from primary")], profile={"max_input_tokens": primary}
        ),
        "test:alt": RecordingModel(
            responses=[AIMessage(content="from alt")], profile={"max_input_tokens": alt}
        ),
    }
    monkeypatch.setattr(
        "deepagents_talon.runtime._resolve_model_from_env",
        lambda model, *_args, **_kwargs: models[model],
    )
    return models["test:alt"]


@pytest.mark.usefixtures("discovered")
async def test_selected_model_brings_its_own_context_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A switch to a larger model must not be held to the startup model's limit."""
    alt = _sized(monkeypatch, primary=2_000, alt=1_000_000)
    long_text = "context " * 4_000
    runtime = _runtime(tmp_path)
    await runtime.start()
    try:
        result = await runtime.invoke(AgentRequest("a", long_text, model="test:alt"))
    finally:
        await runtime.stop()

    assert result.text == "from alt"
    assert alt.seen == [long_text]


@pytest.mark.usefixtures("discovered")
async def test_selected_smaller_model_is_held_to_its_own_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A switch to a smaller model must not send it input sized for the startup model."""
    alt = _sized(monkeypatch, primary=1_000_000, alt=2_000)
    long_text = "context " * 4_000
    runtime = _runtime(tmp_path)
    await runtime.start()
    try:
        await runtime.invoke(AgentRequest("a", long_text, model="test:alt"))
    except Exception:  # noqa: BLE001, S110  # rejecting before the call is also correct
        pass
    finally:
        await runtime.stop()

    assert long_text not in alt.seen


@pytest.mark.usefixtures("discovered")
@pytest.mark.parametrize("smaller_default", [False, True])
async def test_long_conversation_can_switch_to_a_smaller_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, smaller_default: bool
) -> None:
    small = RecordingModel(
        responses=[AIMessage(content="from small")], profile={"max_input_tokens": 16_000}
    )
    large = RecordingModel(
        responses=[AIMessage(content="from large")], profile={"max_input_tokens": 1_000_000}
    )
    models = {
        "test:primary": small if smaller_default else large,
        "test:alt": large if smaller_default else small,
    }
    monkeypatch.setattr(
        "deepagents_talon.runtime._resolve_model_from_env",
        lambda model, *_args, **_kwargs: models[model],
    )
    runtime = _runtime(tmp_path)
    await runtime.start()
    try:
        for _ in range(8):
            await runtime.invoke(
                AgentRequest("a", "context " * 2_000, model="test:alt" if smaller_default else None)
            )
        result = await runtime.invoke(
            AgentRequest("a", "Continue.", model=None if smaller_default else "test:alt")
        )
    finally:
        await runtime.stop()

    assert large.input_sizes[-1] > 16_000
    assert result.text == "from small"
    assert len(small.input_sizes) >= 2  # Summarization and the main reply both ran.
    assert max(small.input_sizes) <= 16_000


class _FakeSummarizer:
    name = "SummarizationMiddleware"
    trace_policy = None

    def __init__(self, model: object) -> None:
        self.model = model

    def wrap_model_call(self, request, handler):
        return handler((self.model, request))


class _Request:
    def __init__(self) -> None:
        self.model = "startup"

    def override(self, *, model: object) -> tuple[str, object]:
        return ("overridden", model)


def _summarization(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SelectedModelSummarization, list[object]]:
    builds: list[object] = []

    def build(model: object, _backend: object, **_kwargs: object) -> _FakeSummarizer:
        builds.append(model)
        return _FakeSummarizer(model)

    monkeypatch.setattr("deepagents_talon.model_selection.create_summarization_middleware", build)
    middleware = SelectedModelSummarization(lambda: cast("Any", "startup"), cast("Any", None))
    return middleware, builds


def _run(middleware: SelectedModelSummarization, selected: object | None) -> tuple:
    token = ACTIVE_MODEL.set(cast("Any", selected))
    try:
        return middleware.wrap_model_call(cast("Any", _Request()), lambda seen: seen)
    finally:
        ACTIVE_MODEL.reset(token)


def test_summarizer_follows_the_selected_model_and_is_built_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    middleware, builds = _summarization(monkeypatch)
    selected = object()

    first = _run(middleware, selected)
    second = _run(middleware, selected)

    assert middleware.name == "SummarizationMiddleware"
    assert first == second == (selected, ("overridden", selected))
    assert builds == [selected]
    assert _run(middleware, None)[0] == "startup"
    assert builds == [selected, "startup"]


def test_subagents_keep_the_startup_summarizer(monkeypatch: pytest.MonkeyPatch) -> None:
    middleware, _builds = _summarization(monkeypatch)
    token = _IN_SUBAGENT.set(True)
    try:
        model, request = _run(middleware, object())
    finally:
        _IN_SUBAGENT.reset(token)

    assert model == "startup"
    assert isinstance(request, _Request)
