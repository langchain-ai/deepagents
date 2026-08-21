"""Unit tests for SubAgentMiddleware initialization and configuration."""

import json
from collections.abc import Callable
from typing import Any, get_type_hints
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.structured_output import AutoStrategy
from langchain.tools import ToolRuntime
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph

from deepagents.backends.state import StateBackend
from deepagents.graph import create_deep_agent
from deepagents.middleware.memory import MemoryMiddleware
from deepagents.middleware.skills import SkillsMiddleware
from deepagents.middleware.subagents import (
    _PARENT_SYSTEM_MESSAGE_KEY,
    GENERAL_PURPOSE_SUBAGENT,
    SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY,
    SubAgentMiddleware,
    _build_task_tool,
    create_sub_agent,
)
from tests.unit_tests.chat_model import GenericFakeChatModel


@tool
def get_weather(city: str) -> str:
    """Get the weather in a city."""
    return f"The weather in {city} is sunny."


class _DynamicStructuredOutputModel(GenericFakeChatModel):
    """Fake model that calls whichever structured-output tool is bound."""

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_name = getattr(self.tools[-1], "name", None)
        if not isinstance(tool_name, str):
            msg = "Expected a structured-output tool to be bound"
            raise TypeError(msg)
        self.messages = iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": tool_name,
                            "args": {
                                "name": "Maya Thornton",
                                "age": 29,
                                "city": "Portland",
                            },
                            "id": "call_payload",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        )
        return super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )


class TestSubagentMiddlewareInit:
    """Tests for SubAgentMiddleware initialization that don't require LLM invocation."""

    @pytest.fixture(autouse=True)
    def set_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Set dummy API key for model initialization."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def test_subagent_middleware_init(self) -> None:
        """Test basic SubAgentMiddleware initialization with general-purpose subagent."""
        middleware = SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[
                {
                    **GENERAL_PURPOSE_SUBAGENT,
                    "model": "gpt-5.4-mini",
                    "tools": [],
                }
            ],
        )
        assert middleware is not None
        # Lean default: no usage prose. Agents reach the model via the task tool.
        assert middleware.system_prompt is None
        assert len(middleware.tools) == 1
        assert middleware.tools[0].name == "task"

    def test_public_init_type_hints_are_runtime_resolvable(self) -> None:
        """Public constructor annotations should support runtime introspection."""
        hints = get_type_hints(SubAgentMiddleware.__init__)
        create_hints = get_type_hints(create_sub_agent)

        assert "state_schema" in hints
        assert "state_schema" in create_hints
        assert "return" in create_hints

    def test_create_sub_agent_defaults_to_empty_system_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, object] = {}

        def fake_create_agent(model: object, **kwargs: object) -> object:
            captured["model"] = model
            captured.update(kwargs)
            return object()

        monkeypatch.setattr("deepagents.middleware.subagents.create_agent", fake_create_agent)
        model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))

        create_sub_agent(
            {
                "name": "worker",
                "description": "Does work.",
                "model": model,
                "tools": [],
            }
        )

        assert captured["system_prompt"] == ""

    def test_rejects_invalid_subagent_mode(self) -> None:
        invalid_spec: Any = {
            "name": "worker",
            "description": "Does work.",
            "runnable": RunnableLambda(lambda _state: {"messages": [AIMessage(content="done")]}),
            "mode": "dynamic",
        }

        with pytest.raises(ValueError, match="invalid mode 'dynamic'"):
            SubAgentMiddleware(backend=StateBackend(), subagents=[invalid_spec])

    def test_rejects_system_prompt_on_forked_subagent(self) -> None:
        invalid_spec: Any = {
            "name": "worker",
            "description": "Does work.",
            "mode": "fork",
            "system_prompt": "You are a SQL expert.",
        }

        with pytest.raises(ValueError, match="cannot set system_prompt"):
            SubAgentMiddleware(backend=StateBackend(), subagents=[invalid_spec])

    def test_forked_subagent_inherits_prompt_and_skips_memory_middleware(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[dict[str, Any]] = []
        runnable = self._make_echo_graph()

        def fake_create_sub_agent(
            spec: dict[str, Any],
            *,
            state_schema: type | None = None,
            response_format: object = None,
        ) -> object:
            del state_schema, response_format
            captured.append(spec)
            return runnable

        monkeypatch.setattr("deepagents.middleware.subagents.create_sub_agent", fake_create_sub_agent)
        model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
        worker_model = GenericFakeChatModel(messages=iter([AIMessage(content="worker done")]))
        fake_agent = MagicMock()
        fake_agent.with_config.return_value = "compiled-agent"

        with patch("deepagents.graph.create_agent", return_value=fake_agent):
            create_deep_agent(
                model=model,
                system_prompt="PARENT_PROMPT",
                memory=["/memory.md"],
                subagents=[
                    {
                        "name": "forked-worker",
                        "description": "Continues with context.",
                        "model": worker_model,
                        "mode": "fork",
                    },
                    {
                        "name": "isolated-worker",
                        "description": "Starts fresh.",
                        "system_prompt": "ISOLATED_PROMPT",
                    },
                ],
            )

        forked = next(spec for spec in captured if spec["name"] == "forked-worker")
        isolated = next(spec for spec in captured if spec["name"] == "isolated-worker")
        assert forked["system_prompt"] == "PARENT_PROMPT"
        # No mirrored MemoryMiddleware for the fork: the captured parent
        # message already carries the parent's own memory content, so a
        # second instance would only reload files to build a prompt
        # fragment that _ForkSystemMessageMiddleware immediately discards.
        assert not any(isinstance(middleware, MemoryMiddleware) for middleware in forked["middleware"])
        assert not any(isinstance(middleware, MemoryMiddleware) for middleware in isolated["middleware"])

    def test_forked_subagent_skips_its_own_skills_middleware(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: list[dict[str, Any]] = []
        runnable = self._make_echo_graph()

        def fake_create_sub_agent(
            spec: dict[str, Any],
            *,
            state_schema: type | None = None,
            response_format: object = None,
        ) -> object:
            del state_schema, response_format
            captured.append(spec)
            return runnable

        monkeypatch.setattr("deepagents.middleware.subagents.create_sub_agent", fake_create_sub_agent)
        model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
        worker_model = GenericFakeChatModel(messages=iter([AIMessage(content="worker done")]))
        fake_agent = MagicMock()
        fake_agent.with_config.return_value = "compiled-agent"

        with patch("deepagents.graph.create_agent", return_value=fake_agent):
            create_deep_agent(
                model=model,
                system_prompt="PARENT_PROMPT",
                subagents=[
                    {
                        "name": "forked-worker",
                        "description": "Continues with context.",
                        "model": worker_model,
                        "mode": "fork",
                        "skills": ["/skills/fork-only/"],
                    },
                    {
                        "name": "isolated-worker",
                        "description": "Starts fresh.",
                        "system_prompt": "ISOLATED_PROMPT",
                        "skills": ["/skills/isolated/"],
                    },
                ],
            )

        forked = next(spec for spec in captured if spec["name"] == "forked-worker")
        isolated = next(spec for spec in captured if spec["name"] == "isolated-worker")
        # The fork's own skills index would only feed a prompt fragment that
        # _ForkSystemMessageMiddleware immediately overwrites with the
        # parent's captured message, so building it is skipped entirely.
        assert not any(isinstance(middleware, SkillsMiddleware) for middleware in forked["middleware"])
        assert any(isinstance(middleware, SkillsMiddleware) for middleware in isolated["middleware"])

    def test_forked_subagent_inherits_dynamic_parent_prompt(self) -> None:
        class _AppendPromptMiddleware(AgentMiddleware):
            def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
                message = request.system_message
                content = "DYNAMIC_PROMPT" if message is None else f"{message.text}\n\nDYNAMIC_PROMPT"
                return handler(request.override(system_message=SystemMessage(content=content)))

        parent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {"description": "continue", "subagent_type": "worker"},
                                "id": "call_worker",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="parent done"),
                ]
            )
        )
        worker_model = GenericFakeChatModel(messages=iter([AIMessage(content="worker done")]))
        agent = create_deep_agent(
            model=parent_model,
            system_prompt="PARENT_PROMPT",
            middleware=[_AppendPromptMiddleware()],
            subagents=[
                {
                    "name": "worker",
                    "description": "Continues with context.",
                    "model": worker_model,
                    "mode": "fork",
                }
            ],
        )

        result = agent.invoke({"messages": [HumanMessage(content="start")]})

        worker_system_message = worker_model.call_history[0]["messages"][0]
        assert worker_system_message.text == "PARENT_PROMPT\n\nDYNAMIC_PROMPT"
        assert _PARENT_SYSTEM_MESSAGE_KEY not in result

    def test_forked_subagent_replaces_repeated_prompt_updates(self) -> None:
        class _AppendPromptMiddleware(AgentMiddleware):
            def wrap_model_call(self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]) -> ModelResponse:
                message = request.system_message
                content = "MEMORY" if message is None else f"{message.text}\n\nMEMORY"
                return handler(request.override(system_message=SystemMessage(content=content)))

        parent_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {"description": "continue", "subagent_type": "worker"},
                                "id": "call_worker",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="parent done"),
                ]
            )
        )
        worker_model = GenericFakeChatModel(messages=iter([AIMessage(content="worker done")]))
        agent = create_deep_agent(
            model=parent_model,
            system_prompt="PARENT_PROMPT",
            middleware=[_AppendPromptMiddleware()],
            subagents=[
                {
                    "name": "worker",
                    "description": "Continues with context.",
                    "model": worker_model,
                    "middleware": [_AppendPromptMiddleware()],
                    "mode": "fork",
                }
            ],
        )

        agent.invoke({"messages": [HumanMessage(content="start")]})

        worker_system_message = worker_model.call_history[0]["messages"][0]
        assert worker_system_message.text == "PARENT_PROMPT\n\nMEMORY"
        assert worker_system_message.text.count("MEMORY") == 1

    def test_create_sub_agent_compiles_declarative_spec(self) -> None:
        """The public helper should compile and invoke declarative specs."""

        class CustomState(MessagesState):
            pass

        model = GenericFakeChatModel(messages=iter([AIMessage(content="done")]))
        runnable = create_sub_agent(
            {
                "name": "worker",
                "description": "Does work.",
                "system_prompt": "Work on the task.",
                "model": model,
                "tools": [get_weather],
                "interrupt_on": {"get_weather": True},
            },
            state_schema=CustomState,
        )

        assert "HumanInTheLoopMiddleware.after_model" in runnable.nodes

        result = runnable.invoke({"messages": [HumanMessage(content="Do work.")]})

        assert result["messages"][-1].content == "done"
        assert result["messages"][-1].name == "worker"
        assert [getattr(tool, "name", None) for tool in model.tools] == ["get_weather"]

    def test_task_tool_compiles_dynamic_response_format_for_declarative_subagent(self) -> None:
        """Dynamic schemas are present when declarative subagent variants compile."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"},
            },
            "required": ["name", "age", "city"],
        }
        response_format = AutoStrategy(schema)
        middleware = SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[
                {
                    "name": "worker",
                    "description": "Does work.",
                    "system_prompt": "Return structured data.",
                    "model": _DynamicStructuredOutputModel(messages=iter(())),
                    "tools": [],
                }
            ],
            system_prompt=None,
        )
        task_tool = middleware.tools[0]
        runtime = ToolRuntime(
            state={},
            context={},
            config={
                "configurable": {
                    SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY: response_format,
                }
            },
            stream_writer=lambda _chunk: None,
            tools=[task_tool],
            tool_call_id="call_worker",
            store=None,
        )

        result = task_tool.func(
            description="Make a person.",
            subagent_type="worker",
            runtime=runtime,
        )

        assert json.loads(result.update["messages"][0].content) == {
            "name": "Maya Thornton",
            "age": 29,
            "city": "Portland",
        }

    def test_task_tool_rejects_response_format_for_compiled_subagent(self) -> None:
        """Dynamic schemas require a declarative spec to compile a variant."""
        schema = {
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }
        response_format = AutoStrategy(schema)

        runnable = RunnableLambda(lambda _state, _config: (_ for _ in ()).throw(AssertionError("compiled runnable should not be invoked")))

        task_tool = _build_task_tool(
            [
                {
                    "name": "worker",
                    "description": "Does work.",
                    "runnable": runnable,
                }
            ]
        )
        runtime = ToolRuntime(
            state={},
            context={},
            config={
                "configurable": {
                    SUBAGENT_RESPONSE_FORMAT_CONFIG_KEY: response_format,
                }
            },
            stream_writer=lambda _chunk: None,
            tools=[task_tool],
            tool_call_id="call_worker",
            store=None,
        )

        with pytest.raises(
            ValueError,
            match='response_schema cannot be used with compiled subagent "worker"',
        ):
            task_tool.func(
                description="Do work.",
                subagent_type="worker",
                runtime=runtime,
            )

    def test_subagent_middleware_with_custom_subagent(self) -> None:
        """Test SubAgentMiddleware initialization with a custom subagent."""
        middleware = SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[
                {
                    "name": "weather",
                    "description": "Weather subagent",
                    "system_prompt": "Get weather.",
                    "model": "gpt-5.4-mini",
                    "tools": [get_weather],
                }
            ],
        )
        assert middleware is not None
        # Lean default: no usage prose; the subagent surfaces via the task tool.
        assert middleware.system_prompt is None
        task_tool = next(t for t in middleware.tools if t.name == "task")
        assert "weather" in (task_tool.description or "")

    def test_subagent_middleware_custom_system_prompt(self) -> None:
        """Test SubAgentMiddleware with a custom system prompt."""
        middleware = SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[
                {
                    "name": "weather",
                    "description": "Weather subagent",
                    "system_prompt": "Get weather.",
                    "model": "gpt-5.4-mini",
                    "tools": [],
                }
            ],
            system_prompt="Use the task tool to call a subagent.",
        )
        assert middleware is not None
        # Custom system prompt plus available subagent types
        assert middleware.system_prompt.startswith("Use the task tool to call a subagent.")

    def test_requires_subagents(self) -> None:
        """Test that at least one subagent is required."""
        with pytest.raises(ValueError, match="At least one subagent"):
            SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[],
            )

    def test_subagent_requires_model(self) -> None:
        """Test that subagents must specify model."""
        with pytest.raises(ValueError, match="must specify 'model'"):
            SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[
                    {
                        "name": "test",
                        "description": "Test",
                        "system_prompt": "Test.",
                        "tools": [],
                        # Missing "model"
                    }
                ],
            )

    def test_subagent_requires_tools(self) -> None:
        """Test that subagents must specify tools."""
        with pytest.raises(ValueError, match="must specify 'tools'"):
            SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[
                    {
                        "name": "test",
                        "description": "Test",
                        "system_prompt": "Test.",
                        "model": "gpt-5.4-mini",
                        # Missing "tools"
                    }
                ],
            )

    def _make_echo_graph(self) -> object:
        """Build a minimal MessagesState graph for use in CompiledSubAgent tests."""

        def echo_node(_state: MessagesState) -> dict:
            return {"messages": [AIMessage(content="hello")]}

        builder = StateGraph(MessagesState)
        builder.add_node("echo", echo_node)
        builder.add_edge(START, "echo")
        return builder.compile()

    def _task_runtime(self, task_tool: object, tool_call_id: str) -> ToolRuntime:
        return ToolRuntime(
            state={},
            context={},
            config={"configurable": {}},
            stream_writer=lambda _chunk: None,
            tools=[task_tool],
            tool_call_id=tool_call_id,
            store=None,
        )

    def test_forked_compiled_subagent_receives_effective_parent_history(self) -> None:
        captured: dict[str, object] = {}

        class _Runnable:
            def with_config(self, config: dict[str, object]) -> "_Runnable":
                del config
                return self

            def invoke(self, state: dict[str, object], config: object = None) -> dict[str, object]:
                del config
                captured.update(state)
                return {"messages": [AIMessage(content="done")]}

        summary = HumanMessage(content="summary")
        after_cutoff = HumanMessage(content="after cutoff")
        in_flight = AIMessage(
            content="must not leak",
            tool_calls=[
                {
                    "name": "task",
                    "args": {},
                    "id": "call_task",
                    "type": "tool_call",
                },
                {
                    "name": "get_weather",
                    "args": {},
                    "id": "call_weather",
                    "type": "tool_call",
                },
            ],
        )
        middleware = SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[
                {
                    "name": "worker",
                    "description": "Continues with context.",
                    "runnable": _Runnable(),
                    "mode": "fork",
                }
            ],
            private_state_keys=frozenset({"_summarization_event"}),
        )
        task_tool = middleware.tools[0]
        runtime = ToolRuntime(
            state={
                "messages": [HumanMessage(content="old"), after_cutoff, in_flight],
                "_summarization_event": {
                    "cutoff_index": 1,
                    "summary_message": summary,
                    "file_path": None,
                },
            },
            context={},
            config={"configurable": {}},
            stream_writer=lambda _chunk: None,
            tools=[task_tool],
            tool_call_id="call_worker",
            store=None,
        )

        task_tool.func(description="new task", subagent_type="worker", runtime=runtime)

        assert captured["messages"] == [summary, after_cutoff, HumanMessage(content="new task")]
        assert "_summarization_event" not in captured

    def test_compiled_fork_does_not_receive_parent_prompt_state(self) -> None:
        captured: dict[str, object] = {}

        class _Runnable:
            def with_config(self, config: dict[str, object]) -> "_Runnable":
                del config
                return self

            def invoke(self, state: dict[str, object], config: object = None) -> dict[str, object]:
                del config
                captured.update(state)
                return {"messages": [AIMessage(content="done")]}

        middleware = SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[
                {
                    "name": "worker",
                    "description": "Continues with context.",
                    "runnable": _Runnable(),
                    "mode": "fork",
                }
            ],
        )
        task_tool = middleware.tools[0]
        runtime = ToolRuntime(
            state={
                "messages": [HumanMessage(content="parent history")],
                _PARENT_SYSTEM_MESSAGE_KEY: SystemMessage(content="SESSION_PROMPT"),
            },
            context={},
            config={"configurable": {}},
            stream_writer=lambda _chunk: None,
            tools=[task_tool],
            tool_call_id="call_worker",
            store=None,
        )

        task_tool.func(description="new task", subagent_type="worker", runtime=runtime)

        assert _PARENT_SYSTEM_MESSAGE_KEY not in captured

    def test_rejects_duplicate_subagent_names(self) -> None:
        class _Runnable:
            def with_config(self, config: dict[str, object]) -> "_Runnable":
                del config
                return self

            def invoke(self, state: dict[str, object], config: object = None) -> dict[str, object]:
                del state, config
                return {"messages": [AIMessage(content="done")]}

        with pytest.raises(ValueError, match="Duplicate subagent name 'worker'"):
            SubAgentMiddleware(
                backend=StateBackend(),
                subagents=[
                    {
                        "name": "worker",
                        "description": "Forked registration.",
                        "runnable": _Runnable(),
                        "mode": "fork",
                    },
                    {
                        "name": "worker",
                        "description": "Second registration.",
                        "runnable": _Runnable(),
                    },
                ],
            )

    def test_handoff_subagent_receives_only_task_description(self) -> None:
        captured: dict[str, object] = {}

        class _Runnable:
            def with_config(self, config: dict[str, object]) -> "_Runnable":
                del config
                return self

            def invoke(self, state: dict[str, object], config: object = None) -> dict[str, object]:
                del config
                captured.update(state)
                return {"messages": [AIMessage(content="done")]}

        middleware = SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[
                {
                    "name": "worker",
                    "description": "Starts fresh.",
                    "runnable": _Runnable(),
                }
            ],
        )
        task_tool = middleware.tools[0]
        runtime = ToolRuntime(
            state={"messages": [HumanMessage(content="parent history")]},
            context={},
            config={"configurable": {}},
            stream_writer=lambda _chunk: None,
            tools=[task_tool],
            tool_call_id="call_worker",
            store=None,
        )

        task_tool.func(description="new task", subagent_type="worker", runtime=runtime)

        assert captured["messages"] == [HumanMessage(content="new task")]

    def test_compiled_subagent_name_propagated_via_config(self) -> None:
        """CompiledSubAgent.name is forwarded into metadata.lc_agent_name and run_name."""
        configs: list[dict[str, object]] = []

        class _Runnable:
            def with_config(self, config: dict[str, object]) -> "_Runnable":
                configs.append(config)
                return self

            def invoke(
                self,
                state: dict[str, object],
                config: object = None,
            ) -> dict[str, object]:
                del state, config
                return {"messages": [AIMessage(content="hello")]}

        SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[
                {
                    "name": "my-subagent",
                    "description": "A custom subagent",
                    "runnable": _Runnable(),
                }
            ],
        )

        assert configs == [
            {
                "metadata": {"lc_agent_name": "my-subagent"},
                "run_name": "my-subagent",
            }
        ]

    def test_compiled_subagent_does_not_mutate_original_runnable(self) -> None:
        """Task-tool setup must not mutate the original runnable."""
        graph = self._make_echo_graph()
        original_config = getattr(graph, "config", None)

        SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[
                {
                    "name": "my-subagent",
                    "description": "A custom subagent",
                    "runnable": graph,
                }
            ],
        )

        assert graph.config == original_config, "Original runnable was mutated; use with_config instead of attribute assignment"

    def test_same_runnable_reused_across_multiple_subagents(self) -> None:
        """Same runnable registered under two different names must not cross-contaminate configs."""

        class _Runnable:
            def __init__(self, config: dict[str, object] | None = None) -> None:
                self.config = config

            def with_config(self, config: dict[str, object]) -> "_Runnable":
                return _Runnable(config)

            def invoke(
                self,
                state: dict[str, object],
                config: object = None,
            ) -> dict[str, object]:
                del state, config
                if self.config is None:
                    msg = "Expected configured runnable clone"
                    raise AssertionError(msg)
                return {"messages": [AIMessage(content=str(self.config["run_name"]))]}

        graph = _Runnable()

        middleware = SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[
                {
                    "name": "agent-alpha",
                    "description": "First binding",
                    "runnable": graph,
                },
                {
                    "name": "agent-beta",
                    "description": "Second binding",
                    "runnable": graph,
                },
            ],
        )

        task_tool = middleware.tools[0]
        alpha = task_tool.func(
            description="Do alpha work.",
            subagent_type="agent-alpha",
            runtime=self._task_runtime(task_tool, "call_alpha"),
        )
        beta = task_tool.func(
            description="Do beta work.",
            subagent_type="agent-beta",
            runtime=self._task_runtime(task_tool, "call_beta"),
        )

        assert alpha.update["messages"][0].content == "agent-alpha"
        assert beta.update["messages"][0].content == "agent-beta"
        assert graph.config is None

    def test_middleware_delegates_to_create_sub_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Middleware should use the shared entrypoint for declarative subagents."""
        graph = self._make_echo_graph()
        calls: list[tuple[object, type | None]] = []

        class CustomState(MessagesState):
            pass

        def fake_create_sub_agent(
            spec: object,
            *,
            state_schema: type | None = None,
            response_format: object = None,
        ) -> object:
            del response_format
            calls.append((spec, state_schema))
            return graph

        monkeypatch.setattr("deepagents.middleware.subagents.create_sub_agent", fake_create_sub_agent)

        SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[
                {
                    "name": "agent-alpha",
                    "description": "First binding",
                    "system_prompt": "Work on the task.",
                    "model": "test-model",
                    "tools": [],
                },
            ],
            state_schema=CustomState,
        )

        assert len(calls) == 1
        assert calls[0][1] is CustomState

    def test_multiple_subagents_with_interrupt_on(self) -> None:
        """Test creating agent with multiple subagents that have interrupt_on configured."""
        agent = create_agent(
            model="claude-sonnet-4-6",
            system_prompt="Use the task tool to call subagents.",
            middleware=[
                SubAgentMiddleware(
                    backend=StateBackend(),
                    subagents=[
                        {
                            "name": "subagent1",
                            "description": "First subagent.",
                            "system_prompt": "You are subagent 1.",
                            "model": "claude-sonnet-4-6",
                            "tools": [get_weather],
                            "interrupt_on": {"get_weather": True},
                        },
                        {
                            "name": "subagent2",
                            "description": "Second subagent.",
                            "system_prompt": "You are subagent 2.",
                            "model": "claude-sonnet-4-6",
                            "tools": [get_weather],
                            "interrupt_on": {"get_weather": True},
                        },
                    ],
                )
            ],
        )
        # This would error if the middleware was accumulated incorrectly
        assert agent is not None
