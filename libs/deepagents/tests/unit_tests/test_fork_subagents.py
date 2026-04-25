"""Fork-mode subagent tests — imported into test_subagents.py.

Split out to keep test_subagents.py readable; the fork path touches seeding,
prompt composition, model inheritance, and telemetry.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

import pytest
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from langchain.tools import ToolRuntime
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.graph import create_deep_agent
from deepagents.middleware._utils import append_to_system_message
from deepagents.middleware.subagents import (
    ALL_FORKED_USAGE_GUIDANCE,
    FORK_USAGE_GUIDANCE,
    FORKED_SUBAGENT_MARKER,
    CompiledSubAgent,
    _build_task_tool,
)
from tests.unit_tests.chat_model import GenericFakeChatModel

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.runnables import RunnableConfig


class _RecordingChatModel(BaseChatModel):
    """Fake chat model that records every call's input messages.

    Returns a tool call invoking the configured subagent on the first call and
    plain final assistant messages afterwards. One instance serves both parent
    and fork (fork inherits parent's model by design), so the recorded_calls
    list lets tests inspect the messages seen by each.
    """

    recorded_calls: list[list[Any]] = []  # noqa: RUF012  # pydantic field, per-instance
    bound_tools_calls: list[list[Any]] = []  # noqa: RUF012  # pydantic field, per-instance
    scripted_subagent: str = "forked"
    scripted_task_description: str = "do it"
    _call_idx: int = 0

    @property
    def _llm_type(self) -> str:
        return "recording"

    def _generate(
        self,
        messages: Sequence[Any],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.recorded_calls.append(list(messages))
        idx = self._call_idx
        self._call_idx += 1
        if idx == 0:
            response: AIMessage = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {
                            "description": self.scripted_task_description,
                            "subagent_type": self.scripted_subagent,
                        },
                        "id": "call-fork-1",
                        "type": "tool_call",
                    }
                ],
            )
        else:
            response = AIMessage(content="fork final response")
        return ChatResult(generations=[ChatGeneration(message=response)])

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:  # type: ignore[override]  # noqa: ANN401
        try:
            self.bound_tools_calls.append(list(tools))
        except TypeError:
            self.bound_tools_calls.append([tools])
        return self


class _SystemAppendingMiddleware(AgentMiddleware[Any, Any, Any]):
    """Test middleware that appends deterministic text to the system prompt."""

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        return handler(request.override(system_message=append_to_system_message(request.system_message, self._text)))


class _RecordingRunnable(Runnable):
    """Stub subagent runnable that records state and config on each invoke."""

    def __init__(self, response_text: str = "done") -> None:
        self.state_inputs: list[dict[str, Any]] = []
        self.configs: list[RunnableConfig | None] = []
        self._response_text = response_text

    def invoke(self, input: dict[str, Any], config: RunnableConfig | None = None, **_: Any) -> dict[str, Any]:  # noqa: A002
        self.state_inputs.append(input)
        self.configs.append(config)
        return {"messages": [AIMessage(content=self._response_text)]}

    async def ainvoke(self, input: dict[str, Any], config: RunnableConfig | None = None, **_: Any) -> dict[str, Any]:  # noqa: A002
        return self.invoke(input, config)


def _make_tool_runtime(*, state: dict[str, Any], tool_call_id: str = "tc-1") -> ToolRuntime:
    return ToolRuntime(
        state=state,
        context=None,
        tool_call_id=tool_call_id,
        store=None,
        stream_writer=lambda _: None,
        config={},
    )


def _system_text(messages: list[Any]) -> str:
    return next(str(m.content) for m in messages if isinstance(m, SystemMessage))


def _tool_names(tools: list[Any]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
        if isinstance(name, str):
            names.add(name)
    return names


def _tool_description(tools: list[Any], name: str) -> str:
    for tool in tools:
        tool_name = tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)
        if tool_name != name:
            continue
        description = tool.get("description") if isinstance(tool, dict) else getattr(tool, "description", "")
        return str(description)
    msg = f"Tool {name!r} not found"
    raise AssertionError(msg)


def _make_fork_agent(model: _RecordingChatModel, **kwargs: Any) -> Runnable:
    return create_deep_agent(
        model=model,
        system_prompt="PARENT_PROMPT_PREFIX",
        subagents=[
            {
                "name": "forked",
                "description": "Fork worker.",
                "system_prompt": "FORK_SUFFIX",
                "tools": [],
                "fork": True,
            }
        ],
        **kwargs,
    )


def _invoke_and_capture_parent_fork(model: _RecordingChatModel, **kwargs: Any) -> tuple[list[Any], list[Any]]:
    agent = _make_fork_agent(model, **kwargs)
    agent.invoke({"messages": [HumanMessage(content="MAIN_USER_MSG")]})

    assert len(model.recorded_calls) >= 2
    return model.recorded_calls[0], model.recorded_calls[1]


def _write_skill(root: Path, *, name: str = "parity-skill") -> str:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: Skill for fork prompt parity.
---

# {name}

Keep prompt cache prefixes stable.
""",
        encoding="utf-8",
    )
    return str(root)


class TestForkSubagents:
    """Fork-mode subagent behavior — context inheritance, prompt composition, telemetry."""

    def test_fork_prepends_parent_messages_when_seeding_state(self) -> None:
        fork_runnable = _RecordingRunnable()
        plain_runnable = _RecordingRunnable()
        task_tool = _build_task_tool(
            [
                {"name": "forked", "description": "Fork worker.", "runnable": fork_runnable, "fork": True},
                {"name": "plain", "description": "Stateless worker.", "runnable": plain_runnable, "fork": False},
            ]
        )

        parent_msgs = [HumanMessage(content="parent Q"), AIMessage(content="parent A")]
        runtime = _make_tool_runtime(state={"messages": parent_msgs})
        task_tool.func(description="do thing", subagent_type="forked", runtime=runtime)

        seeded = fork_runnable.state_inputs[0]["messages"]
        assert [m.content for m in seeded] == ["parent Q", "parent A", "do thing"]
        assert isinstance(seeded[-1], HumanMessage)

        runtime2 = _make_tool_runtime(state={"messages": parent_msgs}, tool_call_id="tc-2")
        task_tool.func(description="do thing 2", subagent_type="plain", runtime=runtime2)
        seeded_plain = plain_runnable.state_inputs[0]["messages"]
        assert len(seeded_plain) == 1
        assert isinstance(seeded_plain[0], HumanMessage)
        assert seeded_plain[0].content == "do thing 2"

    def test_subagent_state_excludes_parent_private_context(self) -> None:
        fork_runnable = _RecordingRunnable()
        task_tool = _build_task_tool(
            [
                {"name": "forked", "description": "Fork worker.", "runnable": fork_runnable, "fork": True},
            ]
        )

        task_tool.func(
            description="do thing",
            subagent_type="forked",
            runtime=_make_tool_runtime(
                state={
                    "messages": [HumanMessage(content="parent Q")],
                    "local_context": "cli local context",
                    "skills_metadata": [{"name": "skill"}],
                    "memory_contents": "memory",
                    "custom_state": "should not leak",
                }
            ),
        )

        seeded = fork_runnable.state_inputs[0]
        assert "local_context" not in seeded
        assert "skills_metadata" not in seeded
        assert "memory_contents" not in seeded
        assert "custom_state" not in seeded
        assert [m.content for m in seeded["messages"]] == ["parent Q", "do thing"]

    def test_nonfork_subagent_keeps_parent_custom_state(self) -> None:
        plain_runnable = _RecordingRunnable()
        task_tool = _build_task_tool(
            [
                {"name": "plain", "description": "Plain worker.", "runnable": plain_runnable, "fork": False},
            ]
        )

        task_tool.func(
            description="do thing",
            subagent_type="plain",
            runtime=_make_tool_runtime(
                state={
                    "messages": [HumanMessage(content="parent Q")],
                    "local_context": "cli local context",
                    "custom_state": "kept",
                }
            ),
        )

        seeded = plain_runnable.state_inputs[0]
        assert seeded["custom_state"] == "kept"
        assert "local_context" not in seeded
        assert [m.content for m in seeded["messages"]] == ["do thing"]

    def test_fork_drops_current_task_tool_call_from_seeded_state(self) -> None:
        fork_runnable = _RecordingRunnable()
        task_tool = _build_task_tool(
            [
                {"name": "forked", "description": "Fork worker.", "runnable": fork_runnable, "fork": True},
            ]
        )

        parent_msgs = [
            HumanMessage(content="cached parent context"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {"description": "do thing", "subagent_type": "forked"},
                        "id": "tc-current",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content="current tool bookkeeping", tool_call_id="tc-current"),
        ]

        task_tool.func(
            description="do thing",
            subagent_type="forked",
            runtime=_make_tool_runtime(state={"messages": parent_msgs}, tool_call_id="tc-current"),
        )

        seeded = fork_runnable.state_inputs[0]["messages"]
        assert [m.content for m in seeded] == ["cached parent context", "do thing"]
        assert all(not isinstance(m, AIMessage) for m in seeded)
        assert all(not isinstance(m, ToolMessage) for m in seeded)

    def test_fork_drops_entire_current_ai_turn_with_parallel_task_calls(self) -> None:
        fork_runnable = _RecordingRunnable()
        task_tool = _build_task_tool(
            [
                {"name": "forked", "description": "Fork worker.", "runnable": fork_runnable, "fork": True},
            ]
        )

        parent_msgs = [
            HumanMessage(content="cached parent context"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {"description": "first task", "subagent_type": "forked"},
                        "id": "tc-current",
                        "type": "tool_call",
                    },
                    {
                        "name": "task",
                        "args": {"description": "parallel sibling", "subagent_type": "forked"},
                        "id": "tc-sibling",
                        "type": "tool_call",
                    },
                ],
            ),
        ]

        task_tool.func(
            description="first task",
            subagent_type="forked",
            runtime=_make_tool_runtime(state={"messages": parent_msgs}, tool_call_id="tc-current"),
        )

        seeded = fork_runnable.state_inputs[0]["messages"]
        assert [m.content for m in seeded] == ["cached parent context", "first task"]
        assert all(not isinstance(m, AIMessage) for m in seeded)

    def test_fork_preserves_prior_completed_tool_history_when_trimming_current_task(self) -> None:
        fork_runnable = _RecordingRunnable()
        task_tool = _build_task_tool(
            [
                {"name": "forked", "description": "Fork worker.", "runnable": fork_runnable, "fork": True},
            ]
        )

        prior_tool_call = AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "search",
                    "args": {"query": "cacheable context"},
                    "id": "tc-prior",
                    "type": "tool_call",
                }
            ],
        )
        prior_tool_result = ToolMessage(content="prior result", tool_call_id="tc-prior")
        parent_msgs = [
            HumanMessage(content="earlier user context"),
            prior_tool_call,
            prior_tool_result,
            AIMessage(content="prior answer"),
            HumanMessage(content="current cached request"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "task",
                        "args": {"description": "fork now", "subagent_type": "forked"},
                        "id": "tc-current",
                        "type": "tool_call",
                    }
                ],
            ),
        ]

        task_tool.func(
            description="fork now",
            subagent_type="forked",
            runtime=_make_tool_runtime(state={"messages": parent_msgs}, tool_call_id="tc-current"),
        )

        seeded = fork_runnable.state_inputs[0]["messages"]
        assert seeded[:-1] == parent_msgs[:-1]
        assert seeded[-1].content == "fork now"
        assert prior_tool_call in seeded
        assert prior_tool_result in seeded

    def test_fork_sets_ls_agent_type_to_fork_subagent(self) -> None:
        fork_runnable = _RecordingRunnable()
        plain_runnable = _RecordingRunnable()
        task_tool = _build_task_tool(
            [
                {"name": "forked", "description": "Fork worker.", "runnable": fork_runnable, "fork": True},
                {"name": "plain", "description": "Stateless worker.", "runnable": plain_runnable, "fork": False},
            ]
        )

        task_tool.func(
            description="x",
            subagent_type="forked",
            runtime=_make_tool_runtime(state={"messages": []}),
        )
        task_tool.func(
            description="y",
            subagent_type="plain",
            runtime=_make_tool_runtime(state={"messages": []}, tool_call_id="tc-2"),
        )

        assert fork_runnable.configs[0]["configurable"]["ls_agent_type"] == "fork-subagent"
        assert plain_runnable.configs[0]["configurable"]["ls_agent_type"] == "subagent"

    def test_forked_subagent_rendered_with_marker_and_guidance(self) -> None:
        fork_runnable = _RecordingRunnable()
        plain_runnable = _RecordingRunnable()
        task_tool = _build_task_tool(
            [
                {"name": "forked", "description": "Fork worker.", "runnable": fork_runnable, "fork": True},
                {"name": "plain", "description": "Stateless worker.", "runnable": plain_runnable, "fork": False},
            ]
        )

        description = task_tool.description
        assert f"forked {FORKED_SUBAGENT_MARKER}: Fork worker." in description
        assert "- plain: Stateless worker." in description
        plain_line = description.split("- plain:")[1].split("\n")[0]
        assert FORKED_SUBAGENT_MARKER not in plain_line
        assert FORK_USAGE_GUIDANCE in description

    def test_no_fork_no_marker_or_guidance(self) -> None:
        task_tool = _build_task_tool(
            [
                {"name": "plain", "description": "Stateless.", "runnable": _RecordingRunnable(), "fork": False},
            ]
        )
        assert FORKED_SUBAGENT_MARKER not in task_tool.description
        assert FORK_USAGE_GUIDANCE not in task_tool.description
        assert ALL_FORKED_USAGE_GUIDANCE not in task_tool.description

    def test_all_forked_subagents_use_default_context_guidance(self) -> None:
        task_tool = _build_task_tool(
            [
                {"name": "alpha", "description": "First fork.", "runnable": _RecordingRunnable(), "fork": True},
                {"name": "beta", "description": "Second fork.", "runnable": _RecordingRunnable(), "fork": True},
            ]
        )

        assert "- alpha: First fork." in task_tool.description
        assert "- beta: Second fork." in task_tool.description
        assert FORKED_SUBAGENT_MARKER not in task_tool.description
        assert FORK_USAGE_GUIDANCE not in task_tool.description
        assert ALL_FORKED_USAGE_GUIDANCE in task_tool.description

    def test_fork_composes_parent_prefix_and_inherits_message_history(self) -> None:
        """End-to-end: fork's model call receives parent's composed system prompt and messages.

        This is the test that actually proves cache alignment is possible —
        the fork's input prefix (system message + history) matches the
        parent's. Providers cache off that prefix.
        """
        model = _RecordingChatModel()
        agent = create_deep_agent(
            model=model,
            system_prompt="PARENT_PROMPT_PREFIX",
            subagents=[
                {
                    "name": "forked",
                    "description": "Fork worker.",
                    "system_prompt": "FORK_SUFFIX",
                    "tools": [],
                    "fork": True,
                }
            ],
        )
        agent.invoke({"messages": [HumanMessage(content="MAIN_USER_MSG")]})

        # First call is the parent's, second is the fork's.
        assert len(model.recorded_calls) >= 2
        parent_input = model.recorded_calls[0]
        fork_input = model.recorded_calls[1]

        # Fork's system message must match the parent's byte-for-byte (no
        # suffix appended) so the cached prefix can be reused.
        parent_system_text = next(str(m.content) for m in parent_input if isinstance(m, SystemMessage))
        fork_system_text = next(str(m.content) for m in fork_input if isinstance(m, SystemMessage))
        assert fork_system_text == parent_system_text, (
            "Fork's system message diverges from the parent's. That breaks prompt-cache alignment for every downstream message block."
        )

        # Parent's HumanMessage is inherited by the fork.
        fork_human_contents = [str(m.content) for m in fork_input if isinstance(m, HumanMessage)]
        assert any("MAIN_USER_MSG" in c for c in fork_human_contents), (
            f"Fork did not inherit parent's HumanMessage. HumanMessages seen: {fork_human_contents}"
        )

        # The fork's own `system_prompt` ("FORK_SUFFIX") rides inside the
        # trailing HumanMessage as a preamble, not in the system slot.
        assert "FORK_SUFFIX" not in fork_system_text, "Fork's own system_prompt leaked into the system slot — cache prefix will diverge."
        last_human = next(m for m in reversed(fork_input) if isinstance(m, HumanMessage))
        assert "FORK_SUFFIX" in str(last_human.content), (
            f"Fork's own instructions were not injected as a preamble into the "
            f"trailing HumanMessage. Last HumanMessage content: {last_human.content!r}"
        )

    def test_fork_system_prompt_matches_parent_with_memory(self, tmp_path: Path) -> None:
        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        memory_path = str(tmp_path / "AGENTS.md")
        backend.upload_files([(memory_path, b"# Memory\nPreserve this cached memory block.")])
        model = _RecordingChatModel()

        parent_input, fork_input = _invoke_and_capture_parent_fork(
            model,
            backend=backend,
            memory=[memory_path],
        )

        parent_system = _system_text(parent_input)
        fork_system = _system_text(fork_input)
        assert parent_system == fork_system
        assert "Preserve this cached memory block." in parent_system

    def test_fork_system_prompt_matches_parent_with_async_subagents(self) -> None:
        model = _RecordingChatModel()
        agent = create_deep_agent(
            model=model,
            system_prompt="PARENT_PROMPT_PREFIX",
            subagents=[
                {
                    "name": "remote-researcher",
                    "description": "Remote async worker.",
                    "graph_id": "remote_graph",
                },
                {
                    "name": "forked",
                    "description": "Fork worker.",
                    "system_prompt": "FORK_SUFFIX",
                    "tools": [],
                    "fork": True,
                },
            ],
        )
        agent.invoke({"messages": [HumanMessage(content="MAIN_USER_MSG")]})

        parent_system = _system_text(model.recorded_calls[0])
        fork_system = _system_text(model.recorded_calls[1])
        assert parent_system == fork_system
        assert "Available async subagent types:" in parent_system
        assert "- remote-researcher: Remote async worker." in parent_system

    def test_fork_system_prompt_matches_parent_with_top_level_skills(self, tmp_path: Path) -> None:
        skills_root = _write_skill(tmp_path / "skills")
        backend = FilesystemBackend(root_dir=str(tmp_path), virtual_mode=False)
        model = _RecordingChatModel()

        parent_input, fork_input = _invoke_and_capture_parent_fork(
            model,
            backend=backend,
            skills=[skills_root],
        )

        parent_system = _system_text(parent_input)
        fork_system = _system_text(fork_input)
        assert parent_system == fork_system
        assert "Skill for fork prompt parity." in parent_system

    def test_fork_system_prompt_matches_parent_with_top_level_system_middleware(self) -> None:
        model = _RecordingChatModel()

        parent_input, fork_input = _invoke_and_capture_parent_fork(
            model,
            middleware=[_SystemAppendingMiddleware("CUSTOM_PARENT_SYSTEM_APPEND")],
        )

        parent_system = _system_text(parent_input)
        fork_system = _system_text(fork_input)
        assert parent_system == fork_system
        assert "CUSTOM_PARENT_SYSTEM_APPEND" in parent_system

    def test_fork_available_agents_block_matches_parent_with_multiple_subagents(self) -> None:
        model = _RecordingChatModel()
        agent = create_deep_agent(
            model=model,
            system_prompt="PARENT_PROMPT_PREFIX",
            subagents=[
                {
                    "name": "plain",
                    "description": "Plain worker.",
                    "system_prompt": "PLAIN",
                    "tools": [],
                },
                {
                    "name": "forked",
                    "description": "Fork worker.",
                    "system_prompt": "FORK_SUFFIX",
                    "tools": [],
                    "fork": True,
                },
            ],
        )
        agent.invoke({"messages": [HumanMessage(content="MAIN_USER_MSG")]})

        parent_system = _system_text(model.recorded_calls[0])
        fork_system = _system_text(model.recorded_calls[1])
        assert parent_system == fork_system
        assert "- general-purpose: General-purpose agent for researching complex questions" in parent_system
        assert "- plain: Plain worker." in parent_system
        assert "- forked: Fork worker." in parent_system

    def test_fork_does_not_receive_recursive_task_tool(self) -> None:
        model = _RecordingChatModel()
        agent = _make_fork_agent(model)
        agent.invoke({"messages": [HumanMessage(content="MAIN_USER_MSG")]})

        assert len(model.bound_tools_calls) >= 2
        parent_tool_names = _tool_names(model.bound_tools_calls[0])
        fork_tool_names = _tool_names(model.bound_tools_calls[1])
        assert "task" in parent_tool_names
        assert "task" not in fork_tool_names

    def test_fork_preserves_system_message_content_blocks(self) -> None:
        model = _RecordingChatModel()
        system_prompt = SystemMessage(
            content=[
                {"type": "text", "text": "PARENT_BLOCK_A"},
                {"type": "text", "text": "\nPARENT_BLOCK_B"},
            ]
        )
        agent = create_deep_agent(
            model=model,
            system_prompt=system_prompt,
            subagents=[
                {
                    "name": "forked",
                    "description": "Fork worker.",
                    "system_prompt": "FORK_SUFFIX",
                    "tools": [],
                    "fork": True,
                }
            ],
        )
        agent.invoke({"messages": [HumanMessage(content="MAIN_USER_MSG")]})

        parent_system_message = next(m for m in model.recorded_calls[0] if isinstance(m, SystemMessage))
        fork_system_message = next(m for m in model.recorded_calls[1] if isinstance(m, SystemMessage))
        assert fork_system_message.content == parent_system_message.content
        assert "PARENT_BLOCK_A" in str(parent_system_message.content)
        assert "PARENT_BLOCK_B" in str(parent_system_message.content)

    def test_fork_without_model_inherits_parent_model(self) -> None:
        parent_model = GenericFakeChatModel(messages=iter([AIMessage(content="noop")]))
        # Should build without raising; fork inherits the parent instance.
        create_deep_agent(
            model=parent_model,
            subagents=[
                {
                    "name": "forked",
                    "description": "Fork worker.",
                    "system_prompt": "FORK",
                    "tools": [],
                    "fork": True,
                }
            ],
        )

    def test_fork_with_explicit_model_raises(self) -> None:
        parent_model = GenericFakeChatModel(messages=iter([AIMessage(content="noop")]))
        explicit_model = _RecordingChatModel()
        with pytest.raises(ValueError, match="Forked subagent 'forked' cannot declare a model"):
            create_deep_agent(
                model=parent_model,
                subagents=[
                    {
                        "name": "forked",
                        "description": "Fork worker.",
                        "system_prompt": "FORK",
                        "model": explicit_model,
                        "tools": [],
                        "fork": True,
                    }
                ],
            )

    def test_compiled_subagent_with_fork_raises(self) -> None:
        parent_model = GenericFakeChatModel(messages=iter([AIMessage(content="noop")]))
        with pytest.raises(ValueError, match="CompiledSubAgent 'compiled-one' cannot set fork=True"):
            create_deep_agent(
                model=parent_model,
                subagents=[
                    {
                        "name": "compiled-one",
                        "description": "Compiled.",
                        "runnable": RunnableLambda(lambda _: {"messages": [AIMessage(content="ok")]}),
                        "fork": True,  # type: ignore[typeddict-unknown-key]
                    }
                ],
            )

    def test_subagent_intermediate_messages_do_not_leak_to_parent(self) -> None:
        """Isolation guarantee — fork and non-fork alike return only the final message to the parent."""
        intermediate = AIMessage(content="SUBAGENT_INTERMEDIATE_SHOULD_NOT_LEAK")
        final = AIMessage(content="subagent final")
        mock_subagent = RunnableLambda(lambda _: {"messages": [intermediate, final]})

        parent_chat_model = GenericFakeChatModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "task",
                                "args": {"description": "do it", "subagent_type": "compiled-spy"},
                                "id": "call_spy",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    AIMessage(content="Done"),
                ]
            )
        )
        agent = create_deep_agent(
            model=parent_chat_model,
            subagents=[
                CompiledSubAgent(
                    name="compiled-spy",
                    description="spy",
                    runnable=mock_subagent,
                ),
            ],
        )
        result = agent.invoke(
            {"messages": [HumanMessage(content="start")]},
            config={"configurable": {"thread_id": f"fork-leak-{uuid.uuid4().hex}"}},
        )
        contents = [getattr(m, "content", "") for m in result["messages"]]
        assert "SUBAGENT_INTERMEDIATE_SHOULD_NOT_LEAK" not in contents
