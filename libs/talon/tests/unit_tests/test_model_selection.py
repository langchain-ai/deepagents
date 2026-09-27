from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from deepagents_talon.channels.base import ChannelExposure
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest, ChannelMessage
from deepagents_talon.runtime import DeepAgentRuntime
from tests.conftest import RecordingChannel
from tests.test_host import BlockingAgent, _config, _wait_for_request

if TYPE_CHECKING:
    from pathlib import Path


class ReplyModel(FakeMessagesListChatModel):
    def bind_tools(self, _tools, **_kwargs: object):
        return self


class UnavailableModel(ReplyModel):
    def _generate(self, *_args: object, **_kwargs: object):
        error = RuntimeError("service unavailable")
        error.status_code = 503  # ty: ignore[unresolved-attribute]
        raise error


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


async def test_switch_applies_to_the_next_turn_of_that_chat_only(tmp_path: Path) -> None:
    agent, channel = SelectableAgent(), _operator_channel()
    host = TalonHost(config=_config(tmp_path), agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("op", "/model other:two"))

        assert channel.sent[-1] == ("chat", "This chat now uses other:two.")
        assert await _turn_model(host, agent, channel, "chat") == "other:two"
        assert await _turn_model(host, agent, channel, "elsewhere") is None
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
        assert await _turn_model(restarted, agent, channel, "chat") == "other:one"
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

        assert channel.sent[-1] == ("chat", "This chat now uses test:primary.")
        assert json.loads(config.model_state_path.read_text()) == {}
        assert await _turn_model(host, agent, channel, "chat") is None
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

    assert overview.startswith("This chat uses test:primary (default).")
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
        "test:down": UnavailableModel(responses=[AIMessage(content="unused")]),
        "test:backup": ReplyModel(responses=[AIMessage(content="from backup")]),
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


@pytest.mark.usefixtures("built")
async def test_selected_model_heads_the_fallback_chain(
    tmp_path: Path, discovered: dict[str, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    discovered["test"].append("down")
    monkeypatch.setenv("DEEPAGENTS_TALON_MODEL_FALLBACKS", "test:backup")
    runtime = _runtime(tmp_path, max_retries=1)
    await runtime.start()
    try:
        result = await runtime.invoke(AgentRequest("a", "hi", model="test:down"))
    finally:
        await runtime.stop()

    assert result.text == "from backup"


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

    assert channel.sent[0] == ("chat", "This chat now uses other:two.")
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
        "This chat uses test:primary (default). Its selected model other:two is unavailable."
    )
