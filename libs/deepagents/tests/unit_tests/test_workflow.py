"""Tests for workflow-mode middleware (declarative multi-agent orchestration).

Covers: tool gating on `workflow_mode`, spec validation, `{{id}}` templating,
sequential phases with data passing, parallel fan-out within a phase, error
isolation, and the async execution path.
"""

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver

from deepagents import create_deep_agent
from deepagents.middleware.workflow import (
    DEFAULT_MAX_STEPS,
    WorkflowMiddleware,
    WorkflowSpec,
    _render_prompt,
    validate_workflow,
)
from tests.unit_tests.chat_model import GenericFakeChatModel

_AVAIL = frozenset({"researcher", "synthesizer", "general-purpose"})


class _ConstantChatModel(BaseChatModel):
    """Stateless fake model that returns a fixed message on every call.

    Thread-safe (no iterator/mutable state), so it is safe to share across the
    parallel subagent invocations a workflow phase performs.
    """

    content: str = ""

    @property
    def _llm_type(self) -> str:
        return "constant"

    def _generate(
        self,
        messages: Sequence[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.content))])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self


class _EchoChatModel(BaseChatModel):
    """Stateless fake model that echoes the last human message back verbatim.

    Lets a test assert exactly what prompt a subagent received (proving
    `{{id}}` templating and cross-phase data passing).
    """

    @property
    def _llm_type(self) -> str:
        return "echo"

    def _generate(
        self,
        messages: Sequence[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = ""
        for msg in reversed(list(messages)):
            if isinstance(msg, HumanMessage):
                text = msg.text if hasattr(msg, "text") else str(msg.content)
                break
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"ECHO:{text}"))])

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self


def _parent_model_calling_workflow(phases: list[dict]) -> GenericFakeChatModel:
    """Parent model that calls the `workflow` tool once then finishes."""
    return GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "workflow", "id": "wf1", "type": "tool_call", "args": {"phases": phases}}],
                ),
                AIMessage(content="done"),
            ]
        )
    )


class TestWorkflowGating:
    """The `workflow` tool is exposed only when `workflow_mode=True`."""

    def test_tool_absent_by_default(self) -> None:
        agent = create_deep_agent(model=GenericFakeChatModel(messages=iter([AIMessage(content="x")])))
        assert "workflow" not in agent.nodes["tools"].bound._tools_by_name

    def test_tool_present_when_enabled(self) -> None:
        agent = create_deep_agent(
            model=GenericFakeChatModel(messages=iter([AIMessage(content="x")])),
            workflow_mode=True,
        )
        tools = agent.nodes["tools"].bound._tools_by_name
        assert "workflow" in tools
        # The task tool still coexists with the workflow tool.
        assert "task" in tools

    def test_middleware_requires_subagents(self) -> None:
        with pytest.raises(ValueError, match="At least one subagent"):
            WorkflowMiddleware(subagents=[])


class TestWorkflowValidation:
    """`validate_workflow` rejects malformed specs with actionable messages."""

    def test_valid_spec_returns_none(self) -> None:
        spec = WorkflowSpec(
            phases=[
                {"title": "p", "steps": [{"id": "a", "subagent_type": "researcher", "prompt": "x"}]},
                {"title": "q", "steps": [{"id": "b", "subagent_type": "synthesizer", "prompt": "use {{a}}", "depends_on": ["a"]}]},
            ]
        )
        assert validate_workflow(spec, available_subagents=_AVAIL) is None

    def test_duplicate_ids_rejected(self) -> None:
        spec = WorkflowSpec(
            phases=[
                {"title": "p", "steps": [{"id": "a", "subagent_type": "researcher", "prompt": "x"}]},
                {"title": "q", "steps": [{"id": "a", "subagent_type": "researcher", "prompt": "y"}]},
            ]
        )
        assert "Duplicate step id 'a'" in (validate_workflow(spec, available_subagents=_AVAIL) or "")

    def test_same_phase_dependency_rejected(self) -> None:
        spec = WorkflowSpec(
            phases=[
                {
                    "title": "p",
                    "steps": [
                        {"id": "x", "subagent_type": "researcher", "prompt": "{{y}}", "depends_on": ["y"]},
                        {"id": "y", "subagent_type": "researcher", "prompt": "hi"},
                    ],
                }
            ]
        )
        assert "same or a later phase" in (validate_workflow(spec, available_subagents=_AVAIL) or "")

    def test_template_ref_without_depends_on_rejected(self) -> None:
        spec = WorkflowSpec(
            phases=[
                {"title": "p", "steps": [{"id": "a", "subagent_type": "researcher", "prompt": "hi"}]},
                {"title": "q", "steps": [{"id": "b", "subagent_type": "researcher", "prompt": "use {{a}}"}]},
            ]
        )
        assert "does not list them in `depends_on`" in (validate_workflow(spec, available_subagents=_AVAIL) or "")

    def test_unknown_subagent_rejected(self) -> None:
        spec = WorkflowSpec(phases=[{"title": "p", "steps": [{"id": "a", "subagent_type": "nope", "prompt": "hi"}]}])
        assert "unknown subagent 'nope'" in (validate_workflow(spec, available_subagents=_AVAIL) or "")

    def test_max_steps_enforced(self) -> None:
        steps = [{"id": f"s{i}", "subagent_type": "researcher", "prompt": "x"} for i in range(DEFAULT_MAX_STEPS + 1)]
        spec = WorkflowSpec(phases=[{"title": "p", "steps": steps}])
        assert "exceeds the limit" in (validate_workflow(spec, available_subagents=_AVAIL) or "")

    def test_render_prompt_substitutes_refs(self) -> None:
        assert _render_prompt("a={{a}} b={{ b.output }}", {"a": "1", "b": "2"}) == "a=1 b=2"

    def test_render_prompt_leaves_unknown_refs(self) -> None:
        assert _render_prompt("x={{missing}}", {}) == "x={{missing}}"


class TestWorkflowExecution:
    """End-to-end workflow execution through `create_deep_agent`."""

    def _make_agent(self, phases: list[dict], synth_model: BaseChatModel | None = None):
        researcher = create_agent(model=_ConstantChatModel(content="FINDING"))
        synth = create_agent(model=synth_model or _ConstantChatModel(content="SYNTHESIS"))
        return create_deep_agent(
            model=_parent_model_calling_workflow(phases),
            workflow_mode=True,
            checkpointer=InMemorySaver(),
            subagents=[
                {"name": "researcher", "description": "researches", "runnable": researcher},
                {"name": "synthesizer", "description": "synthesizes", "runnable": synth},
            ],
        )

    def test_sequential_phases_and_data_passing(self) -> None:
        # The synth echoes its prompt, proving templated upstream outputs arrive.
        agent = self._make_agent(
            phases=[
                {"title": "Research", "steps": [{"id": "a", "subagent_type": "researcher", "prompt": "research A"}]},
                {
                    "title": "Synth",
                    "steps": [{"id": "s", "subagent_type": "synthesizer", "depends_on": ["a"], "prompt": "Got: {{a}}"}],
                },
            ],
            synth_model=_EchoChatModel(),
        )
        result = agent.invoke({"messages": [HumanMessage(content="go")]}, config={"configurable": {"thread_id": "seq"}})
        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        assert len(tool_messages) == 1
        # Final result is the last phase's only step output: the echoed, templated prompt.
        assert tool_messages[0].content == "ECHO:Got: FINDING"

    def test_parallel_fan_out_then_fan_in(self) -> None:
        agent = self._make_agent(
            phases=[
                {
                    "title": "Research",
                    "steps": [
                        {"id": "a", "subagent_type": "researcher", "prompt": "research A"},
                        {"id": "b", "subagent_type": "researcher", "prompt": "research B"},
                    ],
                },
                {
                    "title": "Synth",
                    "steps": [
                        {"id": "s", "subagent_type": "synthesizer", "depends_on": ["a", "b"], "prompt": "{{a}} + {{b}}"},
                    ],
                },
            ],
            synth_model=_EchoChatModel(),
        )
        result = agent.invoke({"messages": [HumanMessage(content="go")]}, config={"configurable": {"thread_id": "par"}})
        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        # Both parallel researchers ran and both outputs were templated into the synth prompt.
        assert tool_messages[0].content == "ECHO:FINDING + FINDING"

    def test_invalid_spec_returns_error_to_model(self) -> None:
        # Forward dependency: validation should fail and surface a ToolMessage, not crash.
        agent = self._make_agent(
            phases=[
                {"title": "p", "steps": [{"id": "x", "subagent_type": "researcher", "prompt": "{{y}}", "depends_on": ["y"]}]},
                {"title": "q", "steps": [{"id": "y", "subagent_type": "researcher", "prompt": "hi"}]},
            ]
        )
        result = agent.invoke({"messages": [HumanMessage(content="go")]}, config={"configurable": {"thread_id": "bad"}})
        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        assert len(tool_messages) == 1
        assert "earlier phase" in tool_messages[0].content

    def test_malformed_spec_returns_friendly_error(self) -> None:
        # A step missing the required `prompt` field fails Pydantic validation.
        # Because the tool advertises a loose arg schema, this reaches the engine
        # and comes back as an actionable message instead of crashing the run.
        agent = self._make_agent(
            phases=[{"title": "p", "steps": [{"id": "a", "subagent_type": "researcher"}]}],
        )
        result = agent.invoke({"messages": [HumanMessage(content="go")]}, config={"configurable": {"thread_id": "malformed"}})
        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        assert len(tool_messages) == 1
        assert "Invalid workflow spec" in tool_messages[0].content

    def test_non_dict_phase_returns_friendly_error(self) -> None:
        # A phase element that isn't an object (here a bare string) must not be
        # rejected at the tool boundary with an opaque error — it should reach
        # the engine and come back as a correctable message.
        agent = self._make_agent(phases=["Summarize", {"title": "p", "steps": [{"id": "a", "subagent_type": "researcher", "prompt": "x"}]}])
        result = agent.invoke({"messages": [HumanMessage(content="go")]}, config={"configurable": {"thread_id": "nondict"}})
        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        assert len(tool_messages) == 1
        assert "Invalid workflow spec" in tool_messages[0].content

    def test_emits_plan_event_before_phases(self) -> None:
        # The `plan` event must arrive (with step descriptions) before any
        # `phase_start`, so a UI can preview the workflow before it runs.
        agent = self._make_agent(
            phases=[
                {
                    "title": "Research",
                    "steps": [{"id": "a", "subagent_type": "researcher", "description": "Look into A", "prompt": "research A"}],
                },
                {
                    "title": "Synth",
                    "steps": [{"id": "s", "subagent_type": "synthesizer", "description": "Combine", "depends_on": ["a"], "prompt": "{{a}}"}],
                },
            ],
        )
        events = []
        for mode, chunk in agent.stream(
            {"messages": [HumanMessage(content="go")]},
            config={"configurable": {"thread_id": "plan"}},
            stream_mode=["custom"],
        ):
            if mode == "custom" and isinstance(chunk, dict) and "workflow" in chunk:
                events.append(chunk["workflow"])

        kinds = [e.get("event") for e in events]
        assert "plan" in kinds
        assert kinds.index("plan") < kinds.index("phase_start")
        plan = next(e for e in events if e.get("event") == "plan")
        assert plan["phase_count"] == 2
        assert plan["step_count"] == 2
        assert plan["phases"][0]["steps"][0]["description"] == "Look into A"
        assert plan["phases"][1]["steps"][0]["depends_on"] == ["a"]

    def test_workflow_model_overrides_step_runner(self) -> None:
        # With `workflow_model` set, the (raw) general-purpose step runs on it,
        # not the main model. The worker returns a sentinel the main model never
        # emits, so the workflow result proves which model executed the step.
        main = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "workflow",
                                "id": "w",
                                "type": "tool_call",
                                "args": {"phases": [{"title": "P", "steps": [{"id": "a", "subagent_type": "general-purpose", "prompt": "do it"}]}]},
                            }
                        ],
                    ),
                    AIMessage(content="done"),
                ]
            )
        )
        worker = GenericFakeChatModel(messages=iter([AIMessage(content="FROM_WORKFLOW_MODEL")]))
        agent = create_deep_agent(model=main, workflow_mode=True, workflow_model=worker, checkpointer=InMemorySaver())
        result = agent.invoke({"messages": [HumanMessage(content="go")]}, config={"configurable": {"thread_id": "wm"}})
        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        assert tool_messages[0].content == "FROM_WORKFLOW_MODEL"

    def test_async_execution(self) -> None:
        agent = self._make_agent(
            phases=[{"title": "P", "steps": [{"id": "only", "subagent_type": "researcher", "prompt": "x"}]}],
        )
        result = asyncio.run(agent.ainvoke({"messages": [HumanMessage(content="go")]}, config={"configurable": {"thread_id": "async"}}))
        tool_messages = [m for m in result["messages"] if m.type == "tool"]
        assert tool_messages[0].content == "FINDING"
