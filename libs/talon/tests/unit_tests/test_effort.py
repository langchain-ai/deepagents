from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import httpx
import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from deepagents_talon.effort import (
    ACTIVE_EFFORT,
    effort_levels,
    effort_settings,
)
from deepagents_talon.host import TalonHost
from deepagents_talon.interfaces import AgentRequest
from deepagents_talon.model_selection import ModelSelectionMiddleware
from deepagents_talon.runtime import DeepAgentRuntime
from tests.test_host import _config, _wait_for_request
from tests.unit_tests.test_model_selection import (
    ReplyModel,
    SelectableAgent,
    _from,
    _operator_channel,
)

if TYPE_CHECKING:
    from pathlib import Path


BASETEN_DEEPSEEK_MODELS = (
    "deepseek-ai/DeepSeek-V4-Flash-0731",
    "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-ai/DeepSeek-V4-Pro-0813",
    "deepseek-ai/DeepSeek-V4.1-Flash",
    "deepseek-ai/DeepSeek-V4.1-Flash-Fast",
)


class EffortAgent(SelectableAgent):
    default_model = "openai:gpt-6.1-sol"

    def __init__(self) -> None:
        super().__init__()
        self.catalog = {
            "openai": ["gpt-6.1-sol", "baseten/zai-org/GLM-5.3-Flash"],
            "baseten": ["zai-org/GLM-5.3-Flash", *BASETEN_DEEPSEEK_MODELS],
            "deepseek": ["deepseek-flash"],
        }


@pytest.mark.parametrize(
    "spec",
    [
        "baseten:zai-org/GLM-5.3-Flash",
        "openai:baseten/zai-org/GLM-5.3-Flash",
        "deepseek:deepseek-flash",
        "baseten:deepseek-ai/DeepSeek-V4-Pro-0813",
    ],
)
async def test_effort_persists_per_model_and_captures_each_turn(tmp_path: Path, spec: str) -> None:
    config = _config(tmp_path)
    agent, channel = EffortAgent(), _operator_channel()
    host = TalonHost(config=config, agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("op", "/effort xhigh"))
        await host.receive_message(channel, _from("op", "first"))
        await _wait_for_request(agent, "first")
        await host.receive_message(channel, _from("op", "/effort low", "other"))
        await host.receive_message(channel, _from("op", "second", "second"))
        await _wait_for_request(agent, "second")
        assert [r.effort for r in agent.requests] == ["xhigh", "low"]
        await host.receive_message(channel, _from("op", f"/model {spec}"))
        await host.receive_message(channel, _from("op", "/effort"))
        assert "default" in channel.sent[-1][1]
        assert "none, low, high, max, default" in channel.sent[-1][1]
        await host.receive_message(channel, _from("op", "/effort high"))
    finally:
        await host.stop()

    assert json.loads(config.effort_state_path.read_text()) == {
        "openai:gpt-6.1-sol": "low",
        spec: "high",
    }
    assert config.effort_state_path.stat().st_mode & 0o777 == 0o600
    host = TalonHost(config=config, agent=EffortAgent(), channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("op", "/effort"))
        assert f"{spec}: high" in channel.sent[-1][1]
        await host.receive_message(channel, _from("op", "/model default"))
        await host.receive_message(channel, _from("op", "/effort"))
        assert "openai:gpt-6.1-sol: low" in channel.sent[-1][1]
        await host.receive_message(channel, _from("op", "/effort default"))
        assert json.loads(config.effort_state_path.read_text()) == {spec: "high"}
    finally:
        await host.stop()


async def test_effort_rejects_unauthorized_and_unsupported_changes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    agent, channel = EffortAgent(), _operator_channel()
    host = TalonHost(config=config, agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("stranger", "/effort high"))
        assert "Only an operator" in channel.sent[-1][1]
        await host.receive_message(channel, _from("op", "/effort arbitrary"))
        assert "Unsupported effort" in channel.sent[-1][1]
        assert not config.effort_state_path.exists()
        assert not agent.requests
    finally:
        await host.stop()


async def test_failed_save_keeps_previous_effort(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    agent, channel = EffortAgent(), _operator_channel()
    host = TalonHost(config=config, agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("op", "/effort high"))

        def fail(*_args: object):
            raise OSError

        monkeypatch.setattr("deepagents_talon.host._write_json_state", fail)
        await host.receive_message(channel, _from("op", "/effort low"))
        assert "Could not save" in channel.sent[-1][1]
        await host.receive_message(channel, _from("op", "/effort"))
        assert "gpt-6.1-sol: high" in channel.sent[-1][1]
    finally:
        await host.stop()


def test_glm_uses_native_thinking_modes() -> None:
    assert effort_settings("openai:z-ai/glm-4.7", "off") == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }
    assert effort_levels("openai:glm-5.3") == ("on",)
    with pytest.raises(ValueError, match="Unsupported"):
        effort_settings("openai:glm-4.7", "max")


@pytest.mark.parametrize("provider", ["deepseek:", "openai:", "openai:deepseek/"])
def test_deepseek_uses_documented_efforts(provider: str, monkeypatch) -> None:
    monkeypatch.setattr("deepagents_talon.effort.supported_efforts_for_model", lambda _spec: ())
    for name in (
        "deepseek-flash",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v4-flash-vision-exp",
    ):
        spec = f"{provider}{name}"
        assert effort_levels(spec) == ("none", "low", "high", "max")
        assert effort_settings(spec, "none") == {"extra_body": {"thinking": {"type": "disabled"}}}
        for effort in ("low", "high", "max"):
            assert effort_settings(spec, effort) == {
                "reasoning_effort": effort,
                "extra_body": {"thinking": {"type": "enabled"}},
            }
        for invalid in ("medium", "xhigh", "on"):
            with pytest.raises(ValueError, match="Unsupported"):
                effort_settings(spec, invalid)
    assert effort_levels(f"{provider}deepseek-unknown") == ()
    assert effort_levels(f"{provider}deepseek-r1") == ()


@pytest.mark.parametrize("name", BASETEN_DEEPSEEK_MODELS)
@pytest.mark.parametrize("prefix", ["baseten:", "openai:baseten/"])
def test_baseten_deepseek_efforts_and_provider_payload(name: str, prefix: str, monkeypatch) -> None:
    pytest.importorskip("langchain_baseten")
    monkeypatch.setattr("deepagents_talon.effort.supported_efforts_for_model", lambda _spec: ())
    spec = f"{prefix}{name}"
    levels = ("none", "low", "high", "max")
    if name == "deepseek-ai/DeepSeek-V4-Flash-0731":
        levels = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
    assert effort_levels(spec) == levels
    provider = "baseten" if prefix == "baseten:" else "openai"
    model = init_chat_model(
        spec,
        api_key="test",
        base_url="https://inference.baseten.co/v1",
    )
    assert model._llm_type == ("baseten-chat" if provider == "baseten" else "openai-chat")
    for level in levels:
        settings = effort_settings(spec, level)
        assert settings == {"reasoning_effort": level}
        payload = model._get_request_payload([HumanMessage(content="hello")], **settings)
        assert payload["model"] == (name if provider == "baseten" else f"baseten/{name}")
        assert payload["reasoning_effort"] == level
        assert "thinking" not in payload
        assert "extra_body" not in payload
    with pytest.raises(ValueError, match="Unsupported"):
        effort_settings(spec, "on")
    if name != "deepseek-ai/DeepSeek-V4-Flash-0731":
        with pytest.raises(ValueError, match="Unsupported"):
            effort_settings(spec, "xhigh")


def test_anthropic_serializes_effort_to_output_config() -> None:
    module = pytest.importorskip("langchain_anthropic")
    model = module.ChatAnthropic(model="claude-opus-4-6", api_key="test")
    payload = model._get_request_payload(
        [HumanMessage(content="hello")], **effort_settings("anthropic:claude-opus-4-6", "high")
    )
    assert payload["output_config"]["effort"] == "high"
    assert "reasoning_effort" not in payload


def test_documented_levels_survive_missing_profile_metadata(monkeypatch) -> None:
    monkeypatch.setattr("deepagents_talon.effort.supported_efforts_for_model", lambda _spec: ())
    assert effort_levels("openai:gpt-5.6") == ("none", "low", "medium", "high", "xhigh", "max")
    assert effort_levels("openai:gpt-6.1-sol") == ("low", "medium", "high", "xhigh", "max")
    assert effort_levels("openai:openai/gpt-6.1-sol") == effort_levels("openai:gpt-6.1-sol")
    assert effort_levels("baseten:zai-org/GLM-5.3-Flash") == ("none", "low", "high", "max")
    assert effort_levels("openai:baseten/zai-org/GLM-5.3-Flash") == ("none", "low", "high", "max")
    for spec, invalid in (
        ("openai:gpt-6.1-sol", "none"),
        ("baseten:zai-org/GLM-5.3-Flash", "xhigh"),
        ("openai:baseten/zai-org/GLM-5.3-Flash", "xhigh"),
    ):
        with pytest.raises(ValueError, match="Unsupported"):
            effort_settings(spec, invalid)
    assert effort_levels("openai:unknown-model") == ()


def test_openai_responses_serializes_effort() -> None:
    module = pytest.importorskip("langchain_openai")
    model = module.ChatOpenAI(model="gpt-6.1-sol", api_key="test", use_responses_api=True)
    payload = model._get_request_payload(
        [HumanMessage(content="hello")], **effort_settings("openai:gpt-6.1-sol", "xhigh")
    )
    assert payload["reasoning"] == {"effort": "xhigh"}
    assert "reasoning_effort" not in payload


@pytest.mark.parametrize(
    ("spec", "level", "expected"),
    [
        ("openai:qwen3.5-plus", "high", {"enable_thinking": True, "thinking_budget": 8192}),
        ("openai:deepseek-flash", "none", {"thinking": {"type": "disabled"}}),
        (
            "openai:deepseek-v4-pro",
            "low",
            {"reasoning_effort": "low", "thinking": {"type": "enabled"}},
        ),
        (
            "openai:deepseek/deepseek-v4-flash",
            "high",
            {"reasoning_effort": "high", "thinking": {"type": "enabled"}},
        ),
        (
            "openai:deepseek-flash",
            "max",
            {"reasoning_effort": "max", "thinking": {"type": "enabled"}},
        ),
        ("openai:glm-4.7", "off", {"thinking": {"type": "disabled"}}),
        ("openai:gpt-6.1-sol", "xhigh", {"reasoning_effort": "xhigh"}),
        ("baseten:zai-org/GLM-5.3-Flash", "high", {"reasoning_effort": "high"}),
        ("openai:baseten/zai-org/GLM-5.3-Flash", "high", {"reasoning_effort": "high"}),
        ("openai:baseten/zai-org/GLM-5.3-Flash", "max", {"reasoning_effort": "max"}),
        ("openai:openai/gpt-6.1-sol", "xhigh", {"reasoning_effort": "xhigh"}),
        ("openai:anthropic/claude-opus-4-6", "high", {"reasoning_effort": "high"}),
    ],
)
def test_openai_compatible_wire_payload(spec: str, level: str, expected: dict) -> None:
    module = pytest.importorskip("langchain_openai")
    payloads: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": spec.split(":", 1)[1],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        model = module.ChatOpenAI(
            model=spec.split(":", 1)[1],
            api_key="test",
            http_client=client,
            use_responses_api=False,
        )
        model.invoke("hello", **effort_settings(spec, level))
    assert all(payloads[0].get(key) == value for key, value in expected.items())
    assert "extra_body" not in payloads[0]
    if level == "none":
        assert "reasoning_effort" not in payloads[0]


class EffortRecorder(ReplyModel):
    efforts: list[object] = Field(default_factory=list)

    def bind_tools(self, _tools, **kwargs: object):
        return self.bind(**kwargs)

    def _generate(self, messages, *args: object, **kwargs: object):
        self.efforts.append(kwargs.get("extra_body"))
        return super()._generate(messages, *args, **kwargs)


async def test_runtime_applies_effort_per_turn_without_leaking(tmp_path: Path, monkeypatch) -> None:
    model = EffortRecorder(responses=[AIMessage(content="done")])
    monkeypatch.setattr(
        "deepagents_talon.runtime._resolve_model_from_env", lambda *_a, **_kw: model
    )
    runtime = DeepAgentRuntime(
        assistant_dir=tmp_path,
        model="openai:qwen3.5-plus",
        env={},
        include_web_tools=False,
        skills=(),
        memory=(),
    )
    await runtime.start()
    try:
        await asyncio.gather(
            runtime.invoke(AgentRequest("a", "first", effort="low")),
            runtime.invoke(AgentRequest("b", "second", effort="max")),
        )
        await runtime.invoke(AgentRequest("c", "third"))
    finally:
        await runtime.stop()
    assert {item.get("thinking_budget") for item in model.efforts[:2]} == {1024, None}
    assert model.efforts[-1] is None
    assert ACTIVE_EFFORT.get() is None


def test_effort_preserves_unrelated_model_settings() -> None:
    model = ReplyModel(responses=[AIMessage(content="done")])
    request = ModelRequest(
        model=model,
        messages=[HumanMessage(content="hi")],
        model_settings={"extra_body": {"unrelated": True}, "temperature": 0.5},
    )
    token = ACTIVE_EFFORT.set(effort_settings("openai:glm-4.7", "on"))
    try:
        updated = ModelSelectionMiddleware().wrap_model_call(request, lambda item: item)
    finally:
        ACTIVE_EFFORT.reset(token)
    assert updated.model_settings["extra_body"] == {
        "unrelated": True,
        "thinking": {"type": "enabled"},
    }
    assert updated.model_settings["temperature"] == 0.5
    assert request.model_settings["extra_body"] == {"unrelated": True}
