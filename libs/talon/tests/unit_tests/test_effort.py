from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import httpx
import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from deepagents_talon.effort import (
    ACTIVE_EFFORT,
    QWEN_BUDGETS,
    effort_levels,
    effort_settings,
    normalize_effort,
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


class EffortAgent(SelectableAgent):
    default_model = "openai:qwen3.5-plus"

    def __init__(self) -> None:
        super().__init__()
        self.catalog = {"openai": ["qwen3.5-plus", "glm-4.7"]}


async def test_effort_persists_per_model_and_captures_each_turn(tmp_path: Path) -> None:
    config = _config(tmp_path)
    agent, channel = EffortAgent(), _operator_channel()
    host = TalonHost(config=config, agent=agent, channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("op", "/effort extra high"))
        await host.receive_message(channel, _from("op", "first"))
        await _wait_for_request(agent, "first")
        await host.receive_message(channel, _from("op", "/effort low", "other"))
        await host.receive_message(channel, _from("op", "second", "second"))
        await _wait_for_request(agent, "second")
        assert [r.effort for r in agent.requests] == ["xhigh", "low"]
        await host.receive_message(channel, _from("op", "/model openai:glm-4.7"))
        await host.receive_message(channel, _from("op", "/effort"))
        assert "default" in channel.sent[-1][1]
        assert "off, on, default" in channel.sent[-1][1]
        await host.receive_message(channel, _from("op", "/effort on"))
    finally:
        await host.stop()

    assert json.loads(config.effort_state_path.read_text()) == {
        "openai:qwen3.5-plus": "low",
        "openai:glm-4.7": "on",
    }
    assert config.effort_state_path.stat().st_mode & 0o777 == 0o600
    host = TalonHost(config=config, agent=EffortAgent(), channels=[channel])
    await host.start()
    try:
        await host.receive_message(channel, _from("op", "/effort"))
        assert "openai:glm-4.7: on" in channel.sent[-1][1]
        await host.receive_message(channel, _from("op", "/model default"))
        await host.receive_message(channel, _from("op", "/effort"))
        assert "openai:qwen3.5-plus: low" in channel.sent[-1][1]
        await host.receive_message(channel, _from("op", "/effort default"))
        assert json.loads(config.effort_state_path.read_text()) == {"openai:glm-4.7": "on"}
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
        assert "qwen3.5-plus: high" in channel.sent[-1][1]
    finally:
        await host.stop()


@pytest.mark.parametrize("alias", ["extra high", "extra-high", "EXTRA_HIGH", "xhigh"])
def test_extra_high_alias(alias: str) -> None:
    assert normalize_effort(alias) == "xhigh"


@pytest.mark.parametrize(("level", "budget"), QWEN_BUDGETS.items())
def test_qwen_budget_presets(level: str, budget: int | None) -> None:
    assert effort_settings("openai:qwen/qwen3.5-plus", level) == {
        "extra_body": {"enable_thinking": True, **({"thinking_budget": budget} if budget else {})}
    }


@pytest.mark.parametrize("spec", ["openai:gpt-4o", "openai:qwen3-coder-plus", "unknown:model"])
def test_nonreasoning_models_reject_effort(spec: str) -> None:
    assert not effort_levels(spec)
    with pytest.raises(ValueError, match="Unsupported reasoning effort"):
        effort_settings(spec, "high")


def test_glm_uses_native_thinking_modes() -> None:
    assert effort_settings("openai:z-ai/glm-4.7", "off") == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }
    assert effort_levels("openai:glm-5.3") == ("on",)
    with pytest.raises(ValueError, match="Unsupported"):
        effort_settings("openai:glm-4.7", "max")


@pytest.mark.parametrize("spec", ["openai:gpt-5.5", "anthropic:claude-opus-4-6"])
def test_native_and_gateway_profiles_share_levels(spec: str) -> None:
    provider, name = spec.split(":")
    assert "high" in effort_levels(spec)
    assert effort_levels(f"openai:{provider}/{name}") == effort_levels(spec)
    assert effort_settings(spec, "high") == {"reasoning_effort": "high"}


def test_anthropic_serializes_effort_to_output_config() -> None:
    module = pytest.importorskip("langchain_anthropic")
    model = module.ChatAnthropic(model="claude-opus-4-6", api_key="test")
    payload = model._get_request_payload(
        [HumanMessage(content="hello")], **effort_settings("anthropic:claude-opus-4-6", "high")
    )
    assert payload["output_config"]["effort"] == "high"
    assert "reasoning_effort" not in payload


@pytest.mark.parametrize(
    ("spec", "level", "expected"),
    [
        ("openai:qwen3.5-plus", "high", {"enable_thinking": True, "thinking_budget": 8192}),
        ("openai:glm-4.7", "off", {"thinking": {"type": "disabled"}}),
        ("openai:gpt-5.5", "xhigh", {"reasoning_effort": "xhigh"}),
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
