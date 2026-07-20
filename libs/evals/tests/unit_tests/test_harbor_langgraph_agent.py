"""Tests for running Deep Agents through Harbor's built-in LangGraph agent."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast, get_args, get_type_hints

import pytest
from deepagents_code.config import settings
from langchain.agents.middleware.types import ExtendedModelResponse, ModelRequest, ModelResponse
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    UsageMetadata,
)
from pydantic import ValidationError

from deepagents_harbor.langgraph_project import langgraph_agent

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langchain.agents.middleware.types import ToolCallRequest
    from langchain_core.messages import AnyMessage
    from langchain_core.tools import BaseTool

_MODEL_IDENTITY_FIELDS = (
    "model_name",
    "model_provider",
    "model_context_limit",
    "model_unsupported_modalities",
)


@pytest.fixture(autouse=True)
def _restore_model_identity_settings() -> Iterator[None]:
    """Snapshot/restore dcode's `settings` model-identity fields.

    `make_graph` writes these process-level singleton fields (so the system
    prompt's Model Identity section is populated). Without restoring them, a
    test that runs `make_graph` would leak the model identity into later tests.
    """
    saved = {field: getattr(settings, field) for field in _MODEL_IDENTITY_FIELDS}
    try:
        yield
    finally:
        for field, value in saved.items():
            setattr(settings, field, value)


def test_langgraph_config_points_to_deepagent_factory() -> None:
    project_path = Path("deepagents_harbor/langgraph_project")
    config_path = project_path / "langgraph.json"

    config = json.loads(config_path.read_text())

    assert config["graphs"] == {
        "deepagent": "./langgraph_agent.py:make_graph",
        "bare_deepagent": "./langgraph_agent.py:make_bare_graph",
        "tau3_deepagent": "./langgraph_agent.py:make_tau3_graph",
    }
    assert not (project_path / "langsmith.py").exists()


def test_langgraph_config_uses_harbor_env_for_fireworks_prereleases() -> None:
    config_path = Path("deepagents_harbor/langgraph_project/langgraph.json")

    dependencies = json.loads(config_path.read_text())["dependencies"]

    assert "langchain-fireworks>=1.4.2,<1.5.0" in dependencies
    assert not any(dependency.startswith("fireworks-ai") for dependency in dependencies)


def test_make_graph_scrubs_credentials_from_shell_backend_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    graph = object()
    captured_env: list[dict[str, str]] = []

    def fake_create_cli_agent(**_kwargs: object) -> tuple[object, object]:
        captured_env.append(dict(langgraph_agent.os.environ))
        return graph, object()

    monkeypatch.setattr(langgraph_agent, "init_chat_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(langgraph_agent, "create_cli_agent", fake_create_cli_agent)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    monkeypatch.setenv("LANGSMITH_API_KEY", "secret")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("HARBOR_MODEL", "anthropic:test-model")
    monkeypatch.setenv("HARBOR_SESSION_ID", "trial-session")

    result = langgraph_agent.make_graph({"configurable": {"cwd": str(tmp_path)}})

    assert result is graph
    assert captured_env
    assert "ANTHROPIC_API_KEY" not in captured_env[0]
    assert "LANGSMITH_API_KEY" not in captured_env[0]
    assert "LANGSMITH_TRACING" not in captured_env[0]
    assert langgraph_agent.os.environ["ANTHROPIC_API_KEY"] == "secret"
    assert langgraph_agent.os.environ["LANGSMITH_API_KEY"] == "secret"
    assert langgraph_agent.os.environ["LANGSMITH_TRACING"] == "true"


def test_make_graph_builds_headless_local_deepagent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_init: list[dict[str, object]] = []
    captured_create: list[dict[str, object]] = []
    graph = object()

    def fake_init_chat_model(model: str, **kwargs: object) -> object:
        captured_init.append({"model": model, "kwargs": kwargs})
        return "chat-model"

    def fake_create_cli_agent(**kwargs: object) -> tuple[object, object]:
        captured_create.append(kwargs)
        return graph, object()

    monkeypatch.setattr(langgraph_agent, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(langgraph_agent, "create_cli_agent", fake_create_cli_agent)
    monkeypatch.setenv("HARBOR_SESSION_ID", "trial-session")

    result = langgraph_agent.make_graph(
        {
            "configurable": {
                "model": "test-provider:test-model",
                "cwd": str(tmp_path),
                "model_kwargs": {"temperature": 0.0},
            }
        }
    )

    assert result is graph
    assert captured_init == [
        {
            "model": "test-provider:test-model",
            "kwargs": {"temperature": 0.0},
        }
    ]
    assert captured_create
    assert captured_create[0]["model"] == "chat-model"
    assert captured_create[0]["assistant_id"] == "trial-session"
    assert captured_create[0]["cwd"] == tmp_path
    assert captured_create[0]["sandbox"] is None
    # `make_graph` must NOT pass `sandbox_type`: it runs locally (sandbox=None), and
    # a non-None sandbox_type routes get_system_prompt through
    # get_default_working_dir(), which raises for unregistered providers like
    # "harbor". Omitting it selects the local-mode prompt rooted at `cwd`.
    assert "sandbox_type" not in captured_create[0]
    assert captured_create[0]["interactive"] is False
    assert captured_create[0]["auto_approve"] is True
    assert captured_create[0]["enable_memory"] is False
    assert captured_create[0]["enable_skills"] is False
    assert captured_create[0]["enable_shell"] is True
    # `make_graph` must NOT pass a system prompt: create_cli_agent then builds the
    # real dcode production prompt (via get_system_prompt), which is what the CLI
    # harness eval is supposed to evaluate. Passing an override would bypass it.
    assert "system_prompt" not in captured_create[0]


def test_make_graph_populates_model_identity_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`make_graph` must feed `configurable.model` into dcode `settings`.

    create_cli_agent -> get_system_prompt renders the prompt's Model Identity
    section from these settings, so without this wiring the eval agent's prompt
    would omit which model it is running as.
    """

    class _Model:
        # Shape mirrors a langchain model profile dict.
        profile = {  # noqa: RUF012
            "max_input_tokens": 200_000,
            "image_inputs": True,
            "audio_inputs": False,
            "video_inputs": False,
            "pdf_inputs": True,
        }

    monkeypatch.setattr(langgraph_agent, "init_chat_model", lambda *_a, **_k: _Model())
    monkeypatch.setattr(
        langgraph_agent,
        "create_cli_agent",
        lambda **_kwargs: (object(), object()),
    )
    monkeypatch.setenv("HARBOR_SESSION_ID", "trial-session")

    langgraph_agent.make_graph(
        {
            "configurable": {
                "model": "anthropic:claude-sonnet-4-5",
                "cwd": str(tmp_path),
            }
        }
    )

    assert settings.model_name == "claude-sonnet-4-5"
    assert settings.model_provider == "anthropic"
    assert settings.model_context_limit == 200_000
    assert settings.model_unsupported_modalities == frozenset({"audio", "video"})


def test_make_graph_defaults_to_app_workdir(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_create: list[dict[str, object]] = []

    monkeypatch.setattr(langgraph_agent, "init_chat_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        langgraph_agent,
        "create_cli_agent",
        lambda **kwargs: (captured_create.append(kwargs) or object(), object()),
    )
    monkeypatch.delenv("HARBOR_SESSION_ID", raising=False)

    langgraph_agent.make_graph(
        {
            "configurable": {
                "model": "test-provider:test-model",
            }
        }
    )

    assert captured_create[0]["cwd"] == Path("/app")
    assert captured_create[0]["assistant_id"]


def test_make_bare_graph_builds_sdk_deepagent_with_local_shell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured_init: list[dict[str, object]] = []
    captured_backend: list[dict[str, object]] = []
    captured_create: list[dict[str, object]] = []
    backend = object()
    graph = object()

    def fake_init_chat_model(model: str, **kwargs: object) -> object:
        captured_init.append({"model": model, "kwargs": kwargs})
        return "chat-model"

    def fake_local_shell_backend(**kwargs: object) -> object:
        captured_backend.append(kwargs)
        return backend

    def fake_create_deep_agent(**kwargs: object) -> object:
        captured_create.append(kwargs)
        return graph

    monkeypatch.setattr(langgraph_agent, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr(langgraph_agent, "LocalShellBackend", fake_local_shell_backend)
    monkeypatch.setattr(langgraph_agent, "create_deep_agent", fake_create_deep_agent)

    result = langgraph_agent.make_bare_graph(
        {
            "configurable": {
                "model": "test-provider:test-model",
                "cwd": str(tmp_path),
                "model_kwargs": {"temperature": 0.0},
            }
        }
    )

    assert result is graph
    assert captured_init == [
        {
            "model": "test-provider:test-model",
            "kwargs": {"temperature": 0.0},
        }
    ]
    assert captured_backend == [{"root_dir": tmp_path, "inherit_env": False}]
    assert captured_create
    assert captured_create[0]["model"] == "chat-model"
    assert captured_create[0]["backend"] is backend
    assert isinstance(captured_create[0]["system_prompt"], str)
    assert "Harbor benchmark sandbox" in captured_create[0]["system_prompt"]


def test_mcp_connections_maps_streamable_http_server() -> None:
    connections = langgraph_agent._mcp_connections(
        {
            "mcp_servers": [
                {
                    "name": "tau3-runtime",
                    "transport": "streamable-http",
                    "url": "http://tau3-runtime:8000/mcp",
                    "command": None,
                    "args": [],
                }
            ]
        }
    )

    assert connections == {
        "tau3-runtime": {
            "transport": "streamable_http",
            "url": "http://tau3-runtime:8000/mcp",
        }
    }


def test_mcp_connections_rejects_stdio_servers() -> None:
    # A dataset-provided stdio server would run an arbitrary local command in the
    # agent sandbox; the tau3 graph must reject it rather than execute it.
    with pytest.raises(ValueError, match="stdio"):
        langgraph_agent._mcp_connections(
            {
                "mcp_servers": [
                    {
                        "name": "evil",
                        "transport": "stdio",
                        "command": "/bin/sh",
                        "args": ["-c", "exfiltrate-secrets"],
                    }
                ]
            }
        )


def test_mcp_connections_requires_forwarded_servers() -> None:
    with pytest.raises(ValueError, match="mcp_servers"):
        langgraph_agent._mcp_connections({})


def test_make_structured_graph_exposes_minimal_background_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The structured A/B must retain the prompt's process-management tools."""

    captured_create: list[dict[str, object]] = []

    class _Backend:
        def __init__(self) -> None:
            self._env: dict[str, str] = {}

    monkeypatch.setattr(langgraph_agent, "init_chat_model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(langgraph_agent, "LocalShellBackend", lambda **_kwargs: _Backend())
    monkeypatch.setattr(
        langgraph_agent, "_TerminusSummarizationMiddleware", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        langgraph_agent,
        "create_agent",
        lambda **kwargs: captured_create.append(kwargs) or object(),
    )

    langgraph_agent.make_structured_graph(
        {"configurable": {"model": "test-provider:test-model", "cwd": str(tmp_path)}}
    )

    tools = cast("list[BaseTool]", captured_create[0]["tools"])
    assert {tool.name for tool in tools} == {
        "execute",
        "run_background",
        "poll",
        "write_stdin",
        "kill",
    }


def _strategy_plan_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "objective": "Build and install CompCert.",
        "observations": ["Ubuntu packages Coq 8.18."],
        "assumptions": ["CompCert accepts ignored Coq version checks."],
        "steps": ["Configure against packaged dependencies.", "Build serially."],
        "costly_commitments": [],
        "fallback": "Inspect the first concrete configure or build error.",
        "verification": "Run ccomp and the task verifier.",
    }
    payload.update(overrides)
    return payload


class _StructuredSequenceRunnable:
    def __init__(self, model: _StructuredSequenceModel, schema: type[object]) -> None:
        self.model = model
        self.schema = schema

    def invoke(
        self,
        messages: list[object],
        *,
        config: dict[str, object] | None = None,
        **kwargs: object,
    ) -> object:
        expected_schema, result = self.model.results.pop(0)
        assert self.schema is expected_schema
        self.model.calls.append(
            {
                "schema": self.schema,
                "messages": messages,
                "config": config,
                "kwargs": kwargs,
            }
        )
        if isinstance(result, Exception):
            raise result
        return result


class _StructuredSequenceModel:
    def __init__(self, results: list[tuple[type[object], object]]) -> None:
        self.results = results
        self.calls: list[dict[str, object]] = []

    def with_structured_output(
        self, schema: type[object], *, include_raw: bool
    ) -> _StructuredSequenceRunnable:
        assert include_raw is True
        return _StructuredSequenceRunnable(self, schema)


def _raw_result(parsed: object, usage: UsageMetadata) -> dict[str, object]:
    return {"parsed": parsed, "raw": AIMessage(content="", usage_metadata=usage)}


def _response_strategy_record(
    response: object,
) -> langgraph_agent._StrategyGateRecord:
    assert isinstance(response, ExtendedModelResponse)
    assert response.command is not None
    assert isinstance(response.command.update, dict)
    record = response.command.update.get("_strategy_gate")
    assert isinstance(record, dict)
    return cast("langgraph_agent._StrategyGateRecord", record)


def test_strategy_reconnaissance_saves_task_and_emits_only_execute() -> None:
    planning = langgraph_agent._PlanningTurn.model_validate(
        {
            "analysis": "Check the packaged toolchain before choosing a build path.",
            "control": {
                "kind": "reconnaissance",
                "actions": [
                    {"action": "execute", "command": "coqc --version"},
                    {"action": "execute", "command": "./configure --help"},
                ],
            },
        }
    )
    usage: UsageMetadata = {"input_tokens": 8, "output_tokens": 3, "total_tokens": 11}
    model = _StructuredSequenceModel(
        [(langgraph_agent._PlanningTurn, _raw_result(planning, usage))]
    )
    request = cast(
        "ModelRequest[None]",
        SimpleNamespace(
            model=model,
            messages=[HumanMessage(content="Build CompCert from the provided source.")],
            system_message=SystemMessage(content="SECRET_ACTOR_SYSTEM_PROSE"),
            state={"messages": []},
        ),
    )
    fallback_calls = 0

    def fallback(_request: object) -> ModelResponse[object]:
        nonlocal fallback_calls
        fallback_calls += 1
        return ModelResponse(result=[AIMessage(content="fallback")])

    response = langgraph_agent._StructuredTurnMiddleware().wrap_model_call(request, fallback)

    assert isinstance(response, ExtendedModelResponse)
    assert fallback_calls == 0
    assert [call["schema"] for call in model.calls] == [langgraph_agent._PlanningTurn]
    planning_messages = cast("list[AnyMessage]", model.calls[0]["messages"])
    assert isinstance(planning_messages[-1], HumanMessage)
    assert "cheap, non-mutating" in planning_messages[-1].text
    assert "install" in planning_messages[-1].text
    assert "build" in planning_messages[-1].text
    assert "edit" in planning_messages[-1].text

    message = response.model_response.result[0]
    assert isinstance(message, AIMessage)
    assert [call["name"] for call in message.tool_calls] == ["execute", "execute"]
    assert message.usage_metadata == usage
    gate_metadata = message.response_metadata["strategy_gate"]
    assert gate_metadata["phase"] == "planning"
    assert "task" not in gate_metadata

    record = _response_strategy_record(response)
    assert record["phase"] == "planning"
    assert record["task"] == "Build CompCert from the provided source."
    assert record["selected_plan"] is None
    assert record["selected_source"] is None


def test_strategy_evaluator_returns_verdict_with_usage_and_isolated_context() -> None:
    proposal = langgraph_agent._StrategyPlan.model_validate(_strategy_plan_payload())
    verdict = langgraph_agent._PlanVerdict.model_validate(
        {"result": {"decision": "approve", "critique": "The plan is economical."}}
    )
    usage: UsageMetadata = {"input_tokens": 13, "output_tokens": 5, "total_tokens": 18}
    model = _StructuredSequenceModel([(langgraph_agent._PlanVerdict, _raw_result(verdict, usage))])
    request = cast(
        "ModelRequest[None]",
        SimpleNamespace(
            model=model,
            messages=[
                HumanMessage(content="Build CompCert."),
                AIMessage(content="SECRET_ACTOR_ANALYSIS"),
            ],
            system_message=SystemMessage(content="SECRET_ACTOR_SYSTEM_PROSE"),
            state={"messages": []},
        ),
    )

    result = langgraph_agent._StructuredTurnMiddleware()._invoke_evaluator(
        request, "SAVED_ORIGINAL_TASK", proposal
    )

    assert result == langgraph_agent._EvaluationCallResult(verdict=verdict, usage=usage, error=None)
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["schema"] is langgraph_agent._PlanVerdict
    assert call["config"] == {"metadata": {"lc_source": "strategy-evaluator"}}
    assert call["kwargs"] == {"max_tokens": langgraph_agent._MAX_STRATEGY_EVALUATOR_OUTPUT_TOKENS}
    messages = cast("list[AnyMessage]", call["messages"])
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    payload = messages[1].text
    assert "SAVED_ORIGINAL_TASK" in payload
    assert "SECRET_ACTOR_ANALYSIS" not in payload
    assert "SECRET_ACTOR_SYSTEM_PROSE" not in payload


def test_strategy_evaluator_non_dict_result_is_parse_failure_with_usage() -> None:
    usage: UsageMetadata = {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}

    result = langgraph_agent._StructuredTurnMiddleware._evaluation_result_parts(
        AIMessage(content="unwrapped malformed result", usage_metadata=usage)
    )

    assert result == langgraph_agent._EvaluationCallResult(
        verdict=None, usage=usage, error="parse_failure"
    )


def test_strategy_approval_runs_selected_actor_plan_with_summed_usage() -> None:
    proposal = langgraph_agent._StrategyPlan.model_validate(_strategy_plan_payload())
    planning = langgraph_agent._PlanningTurn.model_validate(
        {"control": {"kind": "propose_plan", "proposal": proposal.model_dump()}}
    )
    verdict = langgraph_agent._PlanVerdict.model_validate(
        {"result": {"decision": "approve", "critique": "Uses packaged dependencies."}}
    )
    turn = langgraph_agent._Turn.model_validate(
        {
            "control": {
                "kind": "continue",
                "actions": [{"action": "execute", "command": "./configure -use-external-menhir"}],
            }
        }
    )
    planning_usage: UsageMetadata = {
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
    }
    evaluator_usage: UsageMetadata = {
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
    }
    turn_usage: UsageMetadata = {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10}
    model = _StructuredSequenceModel(
        [
            (langgraph_agent._PlanningTurn, _raw_result(planning, planning_usage)),
            (langgraph_agent._PlanVerdict, _raw_result(verdict, evaluator_usage)),
            (langgraph_agent._Turn, _raw_result(turn, turn_usage)),
        ]
    )
    request = ModelRequest(
        model=cast("Any", model),
        messages=[HumanMessage(content="Build CompCert economically.")],
        system_message=SystemMessage(content="Actor system prompt."),
        state={"messages": []},
    )
    fallback_calls = 0

    def fallback(_request: object) -> ModelResponse[object]:
        nonlocal fallback_calls
        fallback_calls += 1
        return ModelResponse(result=[AIMessage(content="fallback")])

    response = langgraph_agent._StructuredTurnMiddleware().wrap_model_call(request, fallback)

    assert isinstance(response, ExtendedModelResponse)
    assert fallback_calls == 0
    assert [call["schema"] for call in model.calls] == [
        langgraph_agent._PlanningTurn,
        langgraph_agent._PlanVerdict,
        langgraph_agent._Turn,
    ]
    execution_messages = cast("list[AnyMessage]", model.calls[2]["messages"])
    strategy_message = execution_messages[-2]
    assert isinstance(strategy_message, HumanMessage)
    assert "Required execution strategy" in strategy_message.text
    assert proposal.objective in strategy_message.text

    message = response.model_response.result[0]
    assert isinstance(message, AIMessage)
    assert [call["name"] for call in message.tool_calls] == ["execute"]
    assert message.usage_metadata == {
        "input_tokens": 30,
        "output_tokens": 9,
        "total_tokens": 39,
    }
    gate_metadata = message.response_metadata["strategy_gate"]
    assert gate_metadata["phase"] == "approved"
    assert gate_metadata["current_proposal"] == proposal.model_dump(mode="json")
    assert gate_metadata["evaluator_decision"] == "approve"
    assert gate_metadata["selected_plan"] == proposal.model_dump(mode="json")
    assert gate_metadata["selected_source"] == "actor"
    assert gate_metadata["revision_count"] == 0
    assert "task" not in gate_metadata

    record = _response_strategy_record(response)
    assert record["phase"] == "approved"
    assert record["selected_plan"] == proposal.model_dump(mode="json")
    assert record["selected_source"] == "actor"
    assert isinstance(record["proposal_id"], str)


def test_strategy_first_revision_can_return_to_reconnaissance() -> None:
    rejected_plan = langgraph_agent._StrategyPlan.model_validate(_strategy_plan_payload())
    recommended_plan = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(
            objective="Confirm packaged dependencies before building CompCert.",
            steps=["Inspect package versions.", "Configure only after confirming compatibility."],
        )
    )
    initial_planning = langgraph_agent._PlanningTurn.model_validate(
        {
            "control": {
                "kind": "propose_plan",
                "proposal": rejected_plan.model_dump(),
            }
        }
    )
    revision = langgraph_agent._PlanVerdict.model_validate(
        {
            "result": {
                "decision": "revise",
                "critique": "The build commitment is unsupported.",
                "missing_evidence": ["Installed Coq and Menhir versions."],
                "cheaper_alternative": "Inspect package metadata first.",
                "recommended_plan": recommended_plan.model_dump(),
            }
        }
    )
    revision_planning = langgraph_agent._PlanningTurn.model_validate(
        {
            "control": {
                "kind": "reconnaissance",
                "actions": [{"action": "execute", "command": "apt-cache policy coq menhir"}],
            }
        }
    )
    model = _StructuredSequenceModel(
        [
            (
                langgraph_agent._PlanningTurn,
                _raw_result(
                    initial_planning,
                    {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
                ),
            ),
            (
                langgraph_agent._PlanVerdict,
                _raw_result(
                    revision,
                    {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
                ),
            ),
            (
                langgraph_agent._PlanningTurn,
                _raw_result(
                    revision_planning,
                    {"input_tokens": 15, "output_tokens": 2, "total_tokens": 17},
                ),
            ),
        ]
    )
    request = ModelRequest(
        model=cast("Any", model),
        messages=[HumanMessage(content="Build CompCert.")],
        state={"messages": []},
    )

    response = langgraph_agent._StructuredTurnMiddleware().wrap_model_call(
        request,
        lambda _request: ModelResponse(result=[AIMessage(content="UNEXPECTED_FALLBACK")]),
    )

    assert isinstance(response, ExtendedModelResponse)
    assert [call["schema"] for call in model.calls] == [
        langgraph_agent._PlanningTurn,
        langgraph_agent._PlanVerdict,
        langgraph_agent._PlanningTurn,
    ]
    feedback_messages = cast("list[AnyMessage]", model.calls[2]["messages"])
    feedback = feedback_messages[-1]
    assert isinstance(feedback, HumanMessage)
    assert "The build commitment is unsupported." in feedback.text
    assert "Installed Coq and Menhir versions." in feedback.text
    assert "Inspect package metadata first." in feedback.text
    assert recommended_plan.objective in feedback.text

    message = response.model_response.result[0]
    assert isinstance(message, AIMessage)
    assert [call["name"] for call in message.tool_calls] == ["execute"]
    assert "The build commitment is unsupported." in message.text
    assert message.usage_metadata == {
        "input_tokens": 37,
        "output_tokens": 10,
        "total_tokens": 47,
    }
    record = _response_strategy_record(response)
    assert record["phase"] == "planning"
    assert record["evaluator_decision"] == "revise"
    assert record["critique"] == "The build commitment is unsupported."
    assert record["revision_count"] == 1
    assert record["selected_plan"] is None
    assert record["selected_source"] is None


def test_strategy_second_rejection_selects_second_evaluator_recommendation() -> None:
    original = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(objective="Build everything from source.")
    )
    first_recommendation = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(objective="Use packaged Coq before building CompCert.")
    )
    revised = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(objective="Build CompCert with packaged Coq and source Menhir.")
    )
    second_recommendation = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(objective="Use both packaged Coq and packaged Menhir.")
    )
    initial_planning = langgraph_agent._PlanningTurn.model_validate(
        {"control": {"kind": "propose_plan", "proposal": original.model_dump()}}
    )
    first_revision = langgraph_agent._PlanVerdict.model_validate(
        {
            "result": {
                "decision": "revise",
                "critique": "Avoid rebuilding Coq.",
                "recommended_plan": first_recommendation.model_dump(),
            }
        }
    )
    revision_planning = langgraph_agent._PlanningTurn.model_validate(
        {"control": {"kind": "propose_plan", "proposal": revised.model_dump()}}
    )
    second_revision = langgraph_agent._PlanVerdict.model_validate(
        {
            "result": {
                "decision": "revise",
                "critique": "Menhir is packaged too.",
                "recommended_plan": second_recommendation.model_dump(),
            }
        }
    )
    turn = langgraph_agent._Turn.model_validate(
        {
            "control": {
                "kind": "continue",
                "actions": [{"action": "execute", "command": "./configure"}],
            }
        }
    )
    one_token: UsageMetadata = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    model = _StructuredSequenceModel(
        [
            (langgraph_agent._PlanningTurn, _raw_result(initial_planning, one_token)),
            (langgraph_agent._PlanVerdict, _raw_result(first_revision, one_token)),
            (langgraph_agent._PlanningTurn, _raw_result(revision_planning, one_token)),
            (langgraph_agent._PlanVerdict, _raw_result(second_revision, one_token)),
            (langgraph_agent._Turn, _raw_result(turn, one_token)),
        ]
    )
    request = ModelRequest(
        model=cast("Any", model),
        messages=[HumanMessage(content="Build CompCert.")],
        state={"messages": []},
    )

    response = langgraph_agent._StructuredTurnMiddleware().wrap_model_call(
        request,
        lambda _request: ModelResponse(result=[AIMessage(content="UNEXPECTED_FALLBACK")]),
    )

    assert isinstance(response, ExtendedModelResponse)
    assert [call["schema"] for call in model.calls] == [
        langgraph_agent._PlanningTurn,
        langgraph_agent._PlanVerdict,
        langgraph_agent._PlanningTurn,
        langgraph_agent._PlanVerdict,
        langgraph_agent._Turn,
    ]
    execution_messages = cast("list[AnyMessage]", model.calls[-1]["messages"])
    guidance = execution_messages[-2]
    assert isinstance(guidance, HumanMessage)
    assert second_recommendation.objective in guidance.text
    assert original.objective not in guidance.text

    message = response.model_response.result[0]
    assert isinstance(message, AIMessage)
    assert message.usage_metadata == {
        "input_tokens": 5,
        "output_tokens": 5,
        "total_tokens": 10,
    }
    record = _response_strategy_record(response)
    assert record["phase"] == "approved"
    assert record["current_proposal"] == revised.model_dump(mode="json")
    assert record["selected_plan"] == second_recommendation.model_dump(mode="json")
    assert record["selected_source"] == "evaluator"
    assert record["evaluator_decision"] == "revise"
    assert record["critique"] == "Menhir is packaged too."
    assert record["revision_count"] == 2
    assert record["evaluator_usage"] == {
        "input_tokens": 2,
        "output_tokens": 2,
        "total_tokens": 4,
    }


@pytest.mark.parametrize(
    ("evaluation_result", "reason", "expected_usage"),
    [
        (
            RuntimeError("provider unavailable"),
            "evaluator_invocation_failure",
            {"input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        ),
        (
            {
                "parsed": None,
                "raw": AIMessage(
                    content="malformed",
                    usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
                ),
            },
            "evaluator_parse_failure",
            {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
        ),
    ],
)
def test_strategy_first_evaluator_failure_bypasses_to_actor_plan(
    evaluation_result: object,
    reason: str,
    expected_usage: UsageMetadata,
) -> None:
    proposal = langgraph_agent._StrategyPlan.model_validate(_strategy_plan_payload())
    planning = langgraph_agent._PlanningTurn.model_validate(
        {"control": {"kind": "propose_plan", "proposal": proposal.model_dump()}}
    )
    turn = langgraph_agent._Turn.model_validate(
        {
            "control": {
                "kind": "continue",
                "actions": [{"action": "execute", "command": "./configure"}],
            }
        }
    )
    one_token: UsageMetadata = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    model = _StructuredSequenceModel(
        [
            (langgraph_agent._PlanningTurn, _raw_result(planning, one_token)),
            (langgraph_agent._PlanVerdict, evaluation_result),
            (langgraph_agent._Turn, _raw_result(turn, one_token)),
        ]
    )
    request = ModelRequest(
        model=cast("Any", model),
        messages=[HumanMessage(content="Build CompCert.")],
        state={"messages": []},
    )

    response = langgraph_agent._StructuredTurnMiddleware().wrap_model_call(
        request,
        lambda _request: ModelResponse(result=[AIMessage(content="UNEXPECTED_FALLBACK")]),
    )

    assert isinstance(response, ExtendedModelResponse)
    assert [call["schema"] for call in model.calls] == [
        langgraph_agent._PlanningTurn,
        langgraph_agent._PlanVerdict,
        langgraph_agent._Turn,
    ]
    message = response.model_response.result[0]
    assert isinstance(message, AIMessage)
    assert message.usage_metadata == expected_usage
    record = _response_strategy_record(response)
    assert record["phase"] == "bypassed"
    assert record["selected_plan"] == proposal.model_dump(mode="json")
    assert record["selected_source"] == "actor"
    assert record["bypass_reason"] == reason
    if reason == "evaluator_parse_failure":
        assert record["evaluator_usage"] == {
            "input_tokens": 3,
            "output_tokens": 1,
            "total_tokens": 4,
        }


def test_strategy_revision_planning_failure_selects_first_recommendation() -> None:
    original = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(objective="Rejected original source-build plan.")
    )
    recommendation = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(objective="Use the packaged toolchain.")
    )
    planning = langgraph_agent._PlanningTurn.model_validate(
        {"control": {"kind": "propose_plan", "proposal": original.model_dump()}}
    )
    revision = langgraph_agent._PlanVerdict.model_validate(
        {
            "result": {
                "decision": "revise",
                "critique": "The source build is unnecessary.",
                "recommended_plan": recommendation.model_dump(),
            }
        }
    )
    turn = langgraph_agent._Turn.model_validate(
        {
            "control": {
                "kind": "continue",
                "actions": [{"action": "execute", "command": "./configure"}],
            }
        }
    )
    one_token: UsageMetadata = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    model = _StructuredSequenceModel(
        [
            (langgraph_agent._PlanningTurn, _raw_result(planning, one_token)),
            (langgraph_agent._PlanVerdict, _raw_result(revision, one_token)),
            (langgraph_agent._PlanningTurn, RuntimeError("revision model unavailable")),
            (langgraph_agent._Turn, _raw_result(turn, one_token)),
        ]
    )
    request = ModelRequest(
        model=cast("Any", model),
        messages=[HumanMessage(content="Build CompCert.")],
        state={"messages": []},
    )

    response = langgraph_agent._StructuredTurnMiddleware().wrap_model_call(
        request,
        lambda _request: ModelResponse(result=[AIMessage(content="UNEXPECTED_FALLBACK")]),
    )

    assert isinstance(response, ExtendedModelResponse)
    execution_messages = cast("list[AnyMessage]", model.calls[-1]["messages"])
    guidance = execution_messages[-2]
    assert isinstance(guidance, HumanMessage)
    assert recommendation.objective in guidance.text
    assert original.objective not in guidance.text
    record = _response_strategy_record(response)
    assert record["phase"] == "approved"
    assert record["selected_plan"] == recommendation.model_dump(mode="json")
    assert record["selected_source"] == "evaluator"
    assert record["revision_count"] == 1
    assert record["bypass_reason"] == "revision_planning_failure"


def test_strategy_second_evaluator_failure_selects_first_recommendation() -> None:
    original = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(objective="Rejected original source-build plan.")
    )
    recommendation = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(objective="Use only packaged dependencies.")
    )
    revised = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(objective="Revised actor plan still awaiting review.")
    )
    planning = langgraph_agent._PlanningTurn.model_validate(
        {"control": {"kind": "propose_plan", "proposal": original.model_dump()}}
    )
    revision = langgraph_agent._PlanVerdict.model_validate(
        {
            "result": {
                "decision": "revise",
                "critique": "Prefer available packages.",
                "recommended_plan": recommendation.model_dump(),
            }
        }
    )
    revision_planning = langgraph_agent._PlanningTurn.model_validate(
        {"control": {"kind": "propose_plan", "proposal": revised.model_dump()}}
    )
    turn = langgraph_agent._Turn.model_validate(
        {
            "control": {
                "kind": "continue",
                "actions": [{"action": "execute", "command": "./configure"}],
            }
        }
    )
    one_token: UsageMetadata = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    second_parse_usage: UsageMetadata = {
        "input_tokens": 3,
        "output_tokens": 1,
        "total_tokens": 4,
    }
    model = _StructuredSequenceModel(
        [
            (langgraph_agent._PlanningTurn, _raw_result(planning, one_token)),
            (langgraph_agent._PlanVerdict, _raw_result(revision, one_token)),
            (langgraph_agent._PlanningTurn, _raw_result(revision_planning, one_token)),
            (
                langgraph_agent._PlanVerdict,
                {
                    "parsed": None,
                    "raw": AIMessage(content="bad", usage_metadata=second_parse_usage),
                },
            ),
            (langgraph_agent._Turn, _raw_result(turn, one_token)),
        ]
    )
    request = ModelRequest(
        model=cast("Any", model),
        messages=[HumanMessage(content="Build CompCert.")],
        state={"messages": []},
    )

    response = langgraph_agent._StructuredTurnMiddleware().wrap_model_call(
        request,
        lambda _request: ModelResponse(result=[AIMessage(content="UNEXPECTED_FALLBACK")]),
    )

    assert isinstance(response, ExtendedModelResponse)
    execution_messages = cast("list[AnyMessage]", model.calls[-1]["messages"])
    guidance = execution_messages[-2]
    assert isinstance(guidance, HumanMessage)
    assert recommendation.objective in guidance.text
    assert original.objective not in guidance.text
    assert revised.objective not in guidance.text
    message = response.model_response.result[0]
    assert isinstance(message, AIMessage)
    assert message.usage_metadata == {
        "input_tokens": 7,
        "output_tokens": 5,
        "total_tokens": 12,
    }
    record = _response_strategy_record(response)
    assert record["phase"] == "approved"
    assert record["current_proposal"] == revised.model_dump(mode="json")
    assert record["selected_plan"] == recommendation.model_dump(mode="json")
    assert record["selected_source"] == "evaluator"
    assert record["revision_count"] == 1
    assert record["bypass_reason"] == "second_evaluator_parse_failure"
    assert record["evaluator_usage"] == {
        "input_tokens": 4,
        "output_tokens": 2,
        "total_tokens": 6,
    }


def test_strategy_initial_planning_failure_bypasses_once_with_saved_task() -> None:
    planning_usage: UsageMetadata = {
        "input_tokens": 4,
        "output_tokens": 1,
        "total_tokens": 5,
    }
    turn_usage: UsageMetadata = {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
    turn = langgraph_agent._Turn.model_validate(
        {
            "control": {
                "kind": "continue",
                "actions": [{"action": "execute", "command": "pwd"}],
            }
        }
    )
    model = _StructuredSequenceModel(
        [
            (
                langgraph_agent._PlanningTurn,
                {"parsed": None, "raw": AIMessage(content="bad", usage_metadata=planning_usage)},
            ),
            (langgraph_agent._Turn, _raw_result(turn, turn_usage)),
        ]
    )
    request = ModelRequest(
        model=cast("Any", model),
        messages=[HumanMessage(content="Preserve this original task.")],
        state={"messages": []},
    )

    response = langgraph_agent._StructuredTurnMiddleware().wrap_model_call(
        request,
        lambda _request: ModelResponse(result=[AIMessage(content="UNEXPECTED_FALLBACK")]),
    )

    assert isinstance(response, ExtendedModelResponse)
    assert [call["schema"] for call in model.calls] == [
        langgraph_agent._PlanningTurn,
        langgraph_agent._Turn,
    ]
    message = response.model_response.result[0]
    assert isinstance(message, AIMessage)
    assert message.usage_metadata == {
        "input_tokens": 6,
        "output_tokens": 2,
        "total_tokens": 8,
    }
    record = _response_strategy_record(response)
    assert record["phase"] == "bypassed"
    assert record["task"] == "Preserve this original task."
    assert record["selected_plan"] is None
    assert record["selected_source"] is None
    assert record["bypass_reason"] == "planning_failure"


def test_strategy_approved_state_skips_review_and_preserves_finish_repair() -> None:
    selected = langgraph_agent._StrategyPlan.model_validate(_strategy_plan_payload())
    record = langgraph_agent._StructuredTurnMiddleware._strategy_record(
        "Saved original task.",
        phase="approved",
        proposal_id="strategy_proposal_saved",
        current_proposal=selected.model_dump(mode="json"),
        evaluator_decision="approve",
        selected_plan=selected.model_dump(mode="json"),
        selected_source="actor",
    )
    finish = langgraph_agent._Turn.model_validate(
        {"control": {"kind": "finish", "summary": "Not yet evidenced."}}
    )
    repair = langgraph_agent._Turn.model_validate(
        {
            "control": {
                "kind": "continue",
                "actions": [{"action": "execute", "command": "run-verifier"}],
            }
        }
    )
    first_usage: UsageMetadata = {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
    repair_usage: UsageMetadata = {"input_tokens": 3, "output_tokens": 1, "total_tokens": 4}
    model = _StructuredSequenceModel(
        [
            (langgraph_agent._Turn, _raw_result(finish, first_usage)),
            (langgraph_agent._Turn, _raw_result(repair, repair_usage)),
        ]
    )
    request = ModelRequest(
        model=cast("Any", model),
        messages=[HumanMessage(content="A summary replaced the original task.")],
        state=cast("Any", {"messages": [], "_strategy_gate": record}),
    )

    response = langgraph_agent._StructuredTurnMiddleware().wrap_model_call(
        request,
        lambda _request: ModelResponse(result=[AIMessage(content="UNEXPECTED_FALLBACK")]),
    )

    assert isinstance(response, ExtendedModelResponse)
    assert [call["schema"] for call in model.calls] == [
        langgraph_agent._Turn,
        langgraph_agent._Turn,
    ]
    for call in model.calls:
        messages = cast("list[AnyMessage]", call["messages"])
        assert any(
            isinstance(message, HumanMessage)
            and "Required execution strategy" in message.text
            and selected.objective in message.text
            for message in messages
        )
    message = response.model_response.result[0]
    assert isinstance(message, AIMessage)
    assert message.response_metadata["structured_turn"]["repair"] is True
    assert message.usage_metadata == {
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
    }
    assert _response_strategy_record(response)["task"] == "Saved original task."


def test_strategy_post_recon_revision_uses_saved_task_and_second_review_bound() -> None:
    original = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(objective="Rejected original plan.")
    )
    first_recommendation = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(objective="First evaluator replacement.")
    )
    revised = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(objective="Actor revision after reconnaissance.")
    )
    second_recommendation = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(objective="Second evaluator replacement.")
    )
    record = langgraph_agent._StructuredTurnMiddleware._strategy_record(
        "SAVED_ORIGINAL_TASK",
        phase="planning",
        proposal_id="strategy_proposal_original",
        current_proposal=original.model_dump(mode="json"),
        evaluator_decision="revise",
        critique="Original critique.",
        revision_count=1,
    )
    cast("dict[str, Any]", record)["recommended_plan"] = first_recommendation.model_dump(
        mode="json"
    )
    revision_planning = langgraph_agent._PlanningTurn.model_validate(
        {"control": {"kind": "propose_plan", "proposal": revised.model_dump()}}
    )
    second_revision = langgraph_agent._PlanVerdict.model_validate(
        {
            "result": {
                "decision": "revise",
                "critique": "The revision is still too costly.",
                "recommended_plan": second_recommendation.model_dump(),
            }
        }
    )
    turn = langgraph_agent._Turn.model_validate(
        {
            "control": {
                "kind": "continue",
                "actions": [{"action": "execute", "command": "./configure"}],
            }
        }
    )
    one_token: UsageMetadata = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
    model = _StructuredSequenceModel(
        [
            (langgraph_agent._PlanningTurn, _raw_result(revision_planning, one_token)),
            (langgraph_agent._PlanVerdict, _raw_result(second_revision, one_token)),
            (langgraph_agent._Turn, _raw_result(turn, one_token)),
        ]
    )
    request = ModelRequest(
        model=cast("Any", model),
        messages=[
            HumanMessage(
                content="SUMMARY_REPLACED_TASK",
                additional_kwargs={"lc_source": "summarization"},
            )
        ],
        state=cast("Any", {"messages": [], "_strategy_gate": record}),
    )

    response = langgraph_agent._StructuredTurnMiddleware().wrap_model_call(
        request,
        lambda _request: ModelResponse(result=[AIMessage(content="UNEXPECTED_FALLBACK")]),
    )

    assert isinstance(response, ExtendedModelResponse)
    assert [call["schema"] for call in model.calls] == [
        langgraph_agent._PlanningTurn,
        langgraph_agent._PlanVerdict,
        langgraph_agent._Turn,
    ]
    assert model.calls[0]["config"] == {"metadata": {"lc_source": "strategy-planning-revision"}}
    planning_messages = cast("list[AnyMessage]", model.calls[0]["messages"])
    assert first_recommendation.objective in planning_messages[-1].text
    evaluator_messages = cast("list[AnyMessage]", model.calls[1]["messages"])
    evaluator_payload = evaluator_messages[1].text
    assert "SAVED_ORIGINAL_TASK" in evaluator_payload
    assert "SUMMARY_REPLACED_TASK" not in evaluator_payload
    updated = _response_strategy_record(response)
    assert updated["selected_plan"] == second_recommendation.model_dump(mode="json")
    assert updated["selected_source"] == "evaluator"
    assert updated["revision_count"] == 2


def test_strategy_post_recon_revision_failure_uses_saved_recommendation() -> None:
    original = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(objective="Rejected original plan.")
    )
    recommendation = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(objective="Saved evaluator replacement.")
    )
    record = langgraph_agent._StructuredTurnMiddleware._strategy_record(
        "SAVED_ORIGINAL_TASK",
        phase="planning",
        proposal_id="strategy_proposal_original",
        current_proposal=original.model_dump(mode="json"),
        recommended_plan=recommendation.model_dump(mode="json"),
        evaluator_decision="revise",
        critique="Original critique.",
        revision_count=1,
    )
    planning_usage: UsageMetadata = {
        "input_tokens": 3,
        "output_tokens": 1,
        "total_tokens": 4,
    }
    turn_usage: UsageMetadata = {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
    turn = langgraph_agent._Turn.model_validate(
        {
            "control": {
                "kind": "continue",
                "actions": [{"action": "execute", "command": "./configure"}],
            }
        }
    )
    model = _StructuredSequenceModel(
        [
            (
                langgraph_agent._PlanningTurn,
                {"parsed": None, "raw": AIMessage(content="bad", usage_metadata=planning_usage)},
            ),
            (langgraph_agent._Turn, _raw_result(turn, turn_usage)),
        ]
    )
    request = ModelRequest(
        model=cast("Any", model),
        messages=[HumanMessage(content="Summary only.")],
        state=cast("Any", {"messages": [], "_strategy_gate": record}),
    )

    response = langgraph_agent._StructuredTurnMiddleware().wrap_model_call(
        request,
        lambda _request: ModelResponse(result=[AIMessage(content="UNEXPECTED_FALLBACK")]),
    )

    assert isinstance(response, ExtendedModelResponse)
    assert [call["schema"] for call in model.calls] == [
        langgraph_agent._PlanningTurn,
        langgraph_agent._Turn,
    ]
    messages = cast("list[AnyMessage]", model.calls[-1]["messages"])
    guidance = messages[-2]
    assert isinstance(guidance, HumanMessage)
    assert recommendation.objective in guidance.text
    assert original.objective not in guidance.text
    message = response.model_response.result[0]
    assert isinstance(message, AIMessage)
    assert message.usage_metadata == {
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
    }
    updated = _response_strategy_record(response)
    assert updated["phase"] == "approved"
    assert updated["selected_plan"] == recommendation.model_dump(mode="json")
    assert updated["selected_source"] == "evaluator"
    assert updated["revision_count"] == 1
    assert updated["bypass_reason"] == "revision_planning_failure"


def test_strategy_planning_contract_only_allows_recon_or_plan() -> None:
    recon = langgraph_agent._PlanningTurn.model_validate(
        {
            "analysis": "Need version and configuration facts.",
            "control": {
                "kind": "reconnaissance",
                "actions": [{"action": "execute", "command": "./configure --help"}],
            },
        }
    )
    assert isinstance(recon.control, langgraph_agent._ReconControl)

    proposal = langgraph_agent._PlanningTurn.model_validate(
        {
            "control": {
                "kind": "propose_plan",
                "proposal": _strategy_plan_payload(),
            }
        }
    )
    assert isinstance(proposal.control, langgraph_agent._SubmitPlanControl)

    with pytest.raises(ValidationError):
        langgraph_agent._PlanningTurn.model_validate(
            {
                "control": {
                    "kind": "reconnaissance",
                    "actions": [{"action": "run_background", "command": "opam install coq"}],
                }
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("objective", " \t "),
        ("steps", ["\n"]),
        ("fallback", "   "),
        ("verification", "\r\n"),
    ],
)
def test_strategy_plan_rejects_whitespace_required_text(field: str, value: object) -> None:
    payload = _strategy_plan_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        langgraph_agent._StrategyPlan.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("critique", " \t "),
        ("missing_evidence", ["\n"]),
    ],
)
def test_strategy_revision_rejects_whitespace_required_text(field: str, value: object) -> None:
    result: dict[str, object] = {
        "decision": "revise",
        "critique": "The strategy needs stronger evidence.",
        "missing_evidence": [],
        "recommended_plan": _strategy_plan_payload(),
    }
    result[field] = value

    with pytest.raises(ValidationError):
        langgraph_agent._PlanVerdict.model_validate({"result": result})


def test_strategy_plan_strips_text_and_bounds_list_items() -> None:
    steps = [f" step {index} " for index in range(8)]

    plan = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(objective=" Build CompCert ", steps=steps)
    )

    assert plan.objective == "Build CompCert"
    assert plan.steps == [f"step {index}" for index in range(8)]

    with pytest.raises(ValidationError):
        langgraph_agent._StrategyPlan.model_validate(
            _strategy_plan_payload(steps=[f"step {index}" for index in range(9)])
        )


def test_strategy_plan_verdict_contracts() -> None:
    approve = langgraph_agent._PlanVerdict.model_validate({"result": {"decision": "approve"}})
    revise = langgraph_agent._PlanVerdict.model_validate(
        {
            "result": {
                "decision": "revise",
                "critique": "Prefer the packaged dependencies.",
                "recommended_plan": _strategy_plan_payload(),
            }
        }
    )

    assert isinstance(approve.result, langgraph_agent._ApprovePlan)
    assert isinstance(revise.result, langgraph_agent._RevisePlan)

    with pytest.raises(ValidationError):
        langgraph_agent._PlanVerdict.model_validate(
            {"result": {"decision": "approve", "unexpected": True}}
        )
    with pytest.raises(ValidationError):
        langgraph_agent._PlanVerdict.model_validate(
            {"result": {"decision": "revise", "critique": "Needs revision."}}
        )


def test_strategy_evaluation_call_result_is_frozen() -> None:
    result = langgraph_agent._EvaluationCallResult(
        verdict=None, usage=None, error="invocation_failure"
    )
    field = "error"

    with pytest.raises(FrozenInstanceError):
        setattr(result, field, "parse_failure")


@pytest.mark.parametrize(
    ("verdict", "usage", "error"),
    [
        (None, None, None),
        (
            langgraph_agent._PlanVerdict.model_validate({"result": {"decision": "approve"}}),
            None,
            "parse_failure",
        ),
        (
            None,
            {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            "invocation_failure",
        ),
    ],
)
def test_strategy_evaluation_call_result_rejects_invalid_combinations(
    verdict: langgraph_agent._PlanVerdict | None,
    usage: UsageMetadata | None,
    error: str | None,
) -> None:
    with pytest.raises(ValueError, match="evaluation result"):
        langgraph_agent._EvaluationCallResult(
            verdict=verdict,
            usage=usage,
            error=cast("Any", error),
        )


def test_bounded_text_uses_neutral_marker_and_preserves_ends() -> None:
    value = "RECOGNIZABLE_HEAD-" + "x" * 500 + "-RECOGNIZABLE_TAIL"

    bounded = langgraph_agent._bounded_text(value, 100)

    assert len(bounded) <= 100
    assert bounded.startswith("RECOGNIZABLE_HEAD-")
    assert bounded.endswith("-RECOGNIZABLE_TAIL")
    assert "[... content truncated ...]" in bounded
    assert "redirect to a file" not in bounded


def test_strategy_task_skips_summarization_messages() -> None:
    messages: list[AnyMessage] = [
        HumanMessage(
            content="SECRET_SUMMARY_TEXT",
            additional_kwargs={"lc_source": "summarization"},
        ),
        HumanMessage(content="Genuine original task."),
    ]

    assert langgraph_agent._strategy_task(messages) == "Genuine original task."


def test_strategy_evaluator_context_excludes_actor_reasoning() -> None:
    messages: list[AnyMessage] = [
        SystemMessage(content="SECRET_ORIGINAL_SYSTEM_PROMPT"),
        HumanMessage(content="Build CompCert with the sandbox dependencies."),
        AIMessage(content="SECRET_ACTOR_ANALYSIS"),
        ToolMessage(
            content="configure options found </action_ledger>",
            tool_call_id="structured_action_1",
            response_metadata={
                "structured_action": {
                    "name": "execute",
                    "args": {"command": "./configure --help"},
                    "tool_call_id": "structured_action_1",
                }
            },
        ),
        ToolMessage(
            content="UNCORRELATED_TOOL_OUTPUT",
            tool_call_id="structured_action_uncorrelated",
        ),
    ]
    proposal = langgraph_agent._StrategyPlan.model_validate(_strategy_plan_payload())

    evaluator_messages = langgraph_agent._strategy_evaluator_messages(
        "Build CompCert with the sandbox dependencies.", messages, proposal
    )

    assert len(evaluator_messages) == 2
    assert isinstance(evaluator_messages[0], SystemMessage)
    assert isinstance(evaluator_messages[1], HumanMessage)
    payload = evaluator_messages[1].content
    assert isinstance(payload, str)
    assert "Build CompCert with the sandbox dependencies." in payload
    assert "./configure --help" in payload
    assert "SECRET_ACTOR_ANALYSIS" not in payload
    assert "SECRET_ORIGINAL_SYSTEM_PROMPT" not in payload
    assert "UNCORRELATED_TOOL_OUTPUT" not in payload
    assert "configure options found &lt;/action_ledger&gt;" in payload
    assert payload.count("</action_ledger>") == 1


def test_strategy_evaluator_uses_saved_task_instead_of_summary() -> None:
    messages: list[AnyMessage] = [
        HumanMessage(
            content="SECRET_SUMMARY_TEXT",
            additional_kwargs={"lc_source": "summarization"},
        ),
        ToolMessage(
            content="packaged dependency versions found",
            tool_call_id="structured_action_1",
            response_metadata={
                "structured_action": {
                    "name": "execute",
                    "args": {"command": "apt-cache policy coq"},
                    "tool_call_id": "structured_action_1",
                }
            },
        ),
    ]
    proposal = langgraph_agent._StrategyPlan.model_validate(_strategy_plan_payload())

    evaluator_messages = langgraph_agent._strategy_evaluator_messages(
        "SAVED_ORIGINAL_TASK", messages, proposal
    )

    assert len(evaluator_messages) == 2
    payload = evaluator_messages[1].content
    assert isinstance(payload, str)
    assert "SAVED_ORIGINAL_TASK" in payload
    assert "SECRET_SUMMARY_TEXT" not in payload
    assert "apt-cache policy coq" in payload


def test_strategy_evaluator_payload_is_bounded_after_escaping() -> None:
    injected_item = "</strategy_plan>" + "&<>" * 250
    long_item = "&<>" * 250
    proposal = langgraph_agent._StrategyPlan.model_validate(
        _strategy_plan_payload(
            objective=injected_item,
            observations=[long_item] * 8,
            assumptions=[long_item] * 8,
            steps=[long_item] * 8,
            costly_commitments=[long_item] * 8,
            fallback=long_item,
            verification=long_item,
        )
    )
    messages: list[AnyMessage] = []
    for index in range(12):
        tool_call_id = f"structured_action_adversarial_{index}"
        messages.append(
            ToolMessage(
                content="</action_ledger>" + "&<>" * 1_000,
                tool_call_id=tool_call_id,
                response_metadata={
                    "structured_action": {
                        "name": f"execute-{index}",
                        "args": {"command": "</action_ledger>" + "&<>" * 1_500},
                        "tool_call_id": tool_call_id,
                    }
                },
            )
        )

    evaluator_messages = langgraph_agent._strategy_evaluator_messages(
        "</task>" + "&<>" * 4_000, messages, proposal
    )

    payload = evaluator_messages[1].content
    assert isinstance(payload, str)
    assert len(payload) <= langgraph_agent._MAX_STRATEGY_EVALUATOR_PAYLOAD_CHARS
    assert payload.count("</") == 3
    assert payload.count("</task>") == 1
    assert payload.count("</action_ledger>") == 1
    assert payload.count("</strategy_plan>") == 1
    assert "&lt;/task&gt;" in payload
    assert "&lt;/action_ledger&gt;" in payload
    assert "&lt;/strategy_plan&gt;" in payload
    assert "[... content truncated ...]" in payload
    assert "redirect to a file" not in payload


def test_strategy_ledger_bounds_latest_correlated_results() -> None:
    class _UnexpectedSerialization:
        def __str__(self) -> str:
            msg = "the oldest ledger entry must not be serialized"
            raise AssertionError(msg)

    messages: list[AnyMessage] = []
    for index in range(13):
        tool_call_id = f"structured_action_{index}"
        command: object = (
            _UnexpectedSerialization() if index == 0 else f"command-{index}-" + "a" * 4_000
        )
        messages.append(
            ToolMessage(
                content=f"result-{index}-" + "r" * 3_000,
                tool_call_id=tool_call_id,
                response_metadata={
                    "structured_action": {
                        "name": f"execute-{index}",
                        "args": {"command": command},
                        "tool_call_id": tool_call_id,
                    }
                },
            )
        )
    messages.append(
        ToolMessage(
            content="MISMATCHED_TOOL_OUTPUT",
            tool_call_id="structured_action_mismatch",
            response_metadata={
                "structured_action": {
                    "name": "mismatched",
                    "args": {"command": "must-not-appear"},
                    "tool_call_id": "structured_action_different",
                }
            },
        )
    )
    messages.append(
        ToolMessage(
            content="UNPREFIXED_TOOL_OUTPUT",
            tool_call_id="ordinary_tool_1",
            response_metadata={
                "structured_action": {
                    "name": "unprefixed",
                    "args": {"command": "UNPREFIXED_COMMAND"},
                    "tool_call_id": "ordinary_tool_1",
                }
            },
        )
    )

    ledger = langgraph_agent._strategy_ledger(messages)
    ledger_payload = json.dumps(ledger)

    assert [entry["name"] for entry in ledger] == [f"execute-{index}" for index in range(1, 13)]
    assert all("MISMATCHED_TOOL_OUTPUT" not in entry["result"] for entry in ledger)
    assert "unprefixed" not in ledger_payload
    assert "UNPREFIXED_COMMAND" not in ledger_payload
    assert "UNPREFIXED_TOOL_OUTPUT" not in ledger_payload
    assert all(len(entry["args"]) <= langgraph_agent._MAX_STRATEGY_ARGS_CHARS for entry in ledger)
    assert all(
        len(entry["result"]) <= langgraph_agent._MAX_STRATEGY_RESULT_CHARS for entry in ledger
    )


def test_structured_turn_middleware_uses_private_strategy_state() -> None:
    assert langgraph_agent._StructuredTurnMiddleware.state_schema is (
        langgraph_agent._StructuredAgentState
    )


def test_strategy_gate_state_is_private() -> None:
    gate_hints = get_type_hints(langgraph_agent._StrategyGateRecord)
    hints = get_type_hints(langgraph_agent._StructuredAgentState, include_extras=True)
    private_hint = get_args(hints["_strategy_gate"])[0]
    metadata = getattr(private_hint, "__metadata__", ())
    assert gate_hints["task"] is str
    assert langgraph_agent.PrivateStateAttr in metadata


def test_strategy_gate_record_rejects_invalid_phase_selection_combinations() -> None:
    selected = cast("dict[str, Any]", _strategy_plan_payload())

    planning = langgraph_agent._StructuredTurnMiddleware._strategy_record("task", phase="planning")
    approved = langgraph_agent._StructuredTurnMiddleware._strategy_record(
        "task",
        phase="approved",
        selected_plan=selected,
        selected_source="actor",
    )
    planning_bypass = langgraph_agent._StructuredTurnMiddleware._strategy_record(
        "task", phase="bypassed", bypass_reason="planning_failure"
    )
    evaluator_bypass = langgraph_agent._StructuredTurnMiddleware._strategy_record(
        "task",
        phase="bypassed",
        selected_plan=selected,
        selected_source="actor",
        bypass_reason="evaluator_parse_failure",
    )

    assert planning["selected_plan"] is None
    assert approved["selected_source"] == "actor"
    assert planning_bypass["selected_source"] is None
    assert evaluator_bypass["selected_plan"] == selected

    with pytest.raises(ValueError, match="planning strategy record"):
        langgraph_agent._StructuredTurnMiddleware._strategy_record(
            "task",
            phase="planning",
            selected_plan=selected,
            selected_source="actor",
        )
    with pytest.raises(ValueError, match="approved strategy record"):
        langgraph_agent._StructuredTurnMiddleware._strategy_record("task", phase="approved")
    with pytest.raises(ValueError, match="bypassed strategy record"):
        langgraph_agent._StructuredTurnMiddleware._strategy_record("task", phase="bypassed")
    with pytest.raises(ValueError, match="selected plan and source"):
        langgraph_agent._StructuredTurnMiddleware._strategy_record(
            "task",
            phase="bypassed",
            selected_plan=selected,
            bypass_reason="evaluator_parse_failure",
        )


def test_structured_turn_routes_multiple_continue_actions() -> None:
    middleware = langgraph_agent._StructuredTurnMiddleware()
    turn = langgraph_agent._Turn.model_validate(
        {
            "analysis": "The build needs monitoring.",
            "plan": "Start the build and poll it.",
            "control": {
                "kind": "continue",
                "actions": [
                    {"action": "execute", "command": "pwd", "timeout": 10},
                    {"action": "run_background", "command": "make all"},
                    {"action": "poll", "handle": "proc-1", "wait_seconds": 20},
                    {"action": "write_stdin", "handle": "proc-1", "text": "yes"},
                    {"action": "kill", "handle": "proc-1"},
                ],
            },
        }
    )

    response = middleware._to_response(turn)

    assert response is not None
    message = response.result[0]
    assert isinstance(message, AIMessage)
    assert [(call["name"], call["args"]) for call in message.tool_calls] == [
        ("execute", {"command": "pwd", "timeout": 10}),
        ("run_background", {"command": "make all"}),
        ("poll", {"handle": "proc-1", "wait_seconds": 20}),
        ("write_stdin", {"handle": "proc-1", "text": "yes"}),
        ("kill", {"handle": "proc-1"}),
    ]
    action_metadata = message.response_metadata["structured_turn"]["actions"]
    assert [action["tool_call_id"] for action in action_metadata] == [
        call["id"] for call in message.tool_calls
    ]
    assert all(
        isinstance(call["id"], str) and call["id"].startswith("structured_action_")
        for call in message.tool_calls
    )


def test_structured_turn_rejects_recorded_actions_plus_finish_shape() -> None:
    with pytest.raises(ValidationError):
        langgraph_agent._Turn.model_validate(
            {
                "analysis": "Verification passed.",
                "actions": [
                    {"action": "poll", "handle": "proc-4", "wait_seconds": 5},
                ],
                "finish": {
                    "summary": "Task complete.",
                    "evidence_tool_call_ids": ["proc-4"],
                },
            }
        )


def test_structured_turn_schema_discriminates_control_kind() -> None:
    control_schema = langgraph_agent._Turn.model_json_schema()["properties"]["control"]

    assert control_schema["discriminator"]["propertyName"] == "kind"


def test_structured_action_instruction_prefers_background_processes() -> None:
    instruction = langgraph_agent._STRUCTURED_ACTION_INSTRUCTION

    assert "run_background" in instruction
    assert "poll" in instruction
    assert "compile" in instruction
    assert "test suite" in instruction
    assert "execute" in instruction


def test_structured_turn_finish_uses_latest_tool_result_automatically() -> None:
    middleware = langgraph_agent._StructuredTurnMiddleware()
    turn = langgraph_agent._Turn.model_validate(
        {
            "analysis": "The verifier passed.",
            "control": {
                "kind": "finish",
                "summary": "Verifier passed.",
            },
        }
    )

    assert middleware._to_response(turn, evidence_tool_call_id=None) is None

    uncorrelated_result = ToolMessage(
        content="ACCEPTANCE: PASS", tool_call_id="structured_action_lookalike"
    )
    assert middleware._latest_tool_call_id([uncorrelated_result]) is None

    structured_result = ToolMessage(
        content="ACCEPTANCE: PASS",
        tool_call_id="structured_action_1",
        response_metadata={
            "structured_action": {
                "name": "execute",
                "args": {"command": "run-verifier"},
                "tool_call_id": "structured_action_1",
            }
        },
    )
    evidence_tool_call_id = middleware._latest_tool_call_id([structured_result])
    assert evidence_tool_call_id == "structured_action_1"

    response = middleware._to_response(turn, evidence_tool_call_id=evidence_tool_call_id)
    assert response is not None
    message = response.result[0]
    assert isinstance(message, AIMessage)
    assert not message.tool_calls
    assert message.response_metadata["structured_turn"]["terminal"] is True
    assert message.response_metadata["structured_turn"]["actions"] == [
        {
            "name": "finish",
            "args": {
                "summary": "Verifier passed.",
                "evidence_tool_call_id": "structured_action_1",
            },
            "tool_call_id": None,
        }
    ]


def test_structured_turn_repairs_finish_without_prior_tool_result() -> None:
    class _StructuredModel:
        def __init__(self) -> None:
            self.messages: list[list[object]] = []
            self.results: list[dict[str, object]] = [
                {
                    "parsed": langgraph_agent._Turn.model_validate(
                        {
                            "control": {
                                "kind": "finish",
                                "summary": "verified",
                            }
                        }
                    ),
                    "raw": AIMessage(
                        content="",
                        usage_metadata={
                            "input_tokens": 10,
                            "output_tokens": 2,
                            "total_tokens": 12,
                        },
                    ),
                },
                {
                    "parsed": langgraph_agent._Turn.model_validate(
                        {
                            "control": {
                                "kind": "continue",
                                "actions": [{"action": "execute", "command": "pwd"}],
                            }
                        }
                    ),
                    "raw": AIMessage(
                        content="",
                        usage_metadata={
                            "input_tokens": 5,
                            "output_tokens": 3,
                            "total_tokens": 8,
                        },
                    ),
                },
            ]

        def with_structured_output(self, *_args: object, **_kwargs: object) -> _StructuredModel:
            return self

        def invoke(self, messages: list[object], **_kwargs: object) -> dict[str, object]:
            self.messages.append(messages)
            return self.results.pop(0)

    model = _StructuredModel()
    selected_plan = cast("dict[str, Any]", _strategy_plan_payload())
    record = langgraph_agent._StructuredTurnMiddleware._strategy_record(
        "saved task",
        phase="approved",
        selected_plan=selected_plan,
        selected_source="actor",
    )
    request = ModelRequest(
        model=cast("Any", model),
        messages=[],
        state=cast("Any", {"messages": [], "_strategy_gate": record}),
    )
    fallback_calls = 0

    def fallback(_request: object) -> ModelResponse[object]:
        nonlocal fallback_calls
        fallback_calls += 1
        return ModelResponse(result=[AIMessage(content="fallback")])

    response = langgraph_agent._StructuredTurnMiddleware().wrap_model_call(request, fallback)

    assert fallback_calls == 0
    assert len(model.messages) == 2
    repair_message = model.messages[1][-1]
    assert isinstance(repair_message, langgraph_agent.HumanMessage)
    assert "continue" in repair_message.content
    assert isinstance(response, ExtendedModelResponse)
    message = response.model_response.result[0]
    assert isinstance(message, AIMessage)
    assert message.usage_metadata == {
        "input_tokens": 15,
        "output_tokens": 5,
        "total_tokens": 20,
    }
    assert message.response_metadata["structured_turn"]["repair"] is True


def test_structured_tool_result_records_its_matching_action() -> None:
    request = cast(
        "ToolCallRequest",
        SimpleNamespace(
            tool_call={
                "name": "poll",
                "args": {"handle": "proc-1", "wait_seconds": 20},
                "id": "structured_action_123",
                "type": "tool_call",
            }
        ),
    )

    result = langgraph_agent._StructuredTurnMiddleware().wrap_tool_call(
        request,
        lambda _request: ToolMessage(
            content="[proc-1: still running]",
            tool_call_id="structured_action_123",
        ),
    )

    assert isinstance(result, ToolMessage)
    assert result.response_metadata["structured_action"] == {
        "name": "poll",
        "args": {"handle": "proc-1", "wait_seconds": 20},
        "tool_call_id": "structured_action_123",
    }


def test_structured_turn_preserves_raw_usage_metadata() -> None:
    middleware = langgraph_agent._StructuredTurnMiddleware()
    usage: UsageMetadata = {"input_tokens": 11, "output_tokens": 7, "total_tokens": 18}

    response = middleware._to_response(
        langgraph_agent._Turn.model_validate(
            {
                "control": {
                    "kind": "continue",
                    "actions": [{"action": "execute", "command": "pwd"}],
                }
            }
        ),
        usage_metadata=usage,
    )

    assert response is not None
    message = response.result[0]
    assert isinstance(message, AIMessage)
    assert message.usage_metadata == usage
