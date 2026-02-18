from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, ToolMessage
from langsmith import testing as t

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

from deepagents.backends.utils import create_file_data, file_data_to_string


def _coerce_result_files_to_strings(raw_files: object) -> dict[str, str]:
    if raw_files is None:
        return {}
    if not isinstance(raw_files, Mapping):
        msg = f"Expected files to be dict, got {type(raw_files)}"
        raise TypeError(msg)

    files: dict[str, str] = {}
    for path, file_data in raw_files.items():
        if not isinstance(path, str):
            msg = f"Expected file path to be str, got {type(path)}"
            raise TypeError(msg)

        if isinstance(file_data, str):
            files[path] = file_data
            continue

        if isinstance(file_data, Mapping) and "content" in file_data:
            files[path] = file_data_to_string(dict(file_data))
            continue

        msg = f"Unexpected file representation for {path}: {type(file_data)}"
        raise TypeError(msg)

    return files


@dataclass(frozen=True)
class TrajectoryExpectations:
    """Optional assertions for an `AgentTrajectory`.

    Any expectation left as `None` is not enforced.

    Attributes:
        num_agent_steps: Exact number of model/action steps.
            This counts the number of `AIMessage` actions captured in the trajectory.
        num_tool_call_requests: Exact number of tool call requests.
            This is computed as the sum of `len(step.action.tool_calls)` across all steps.
    """

    num_agent_steps: int | None = None
    num_tool_call_requests: int | None = None


@dataclass(frozen=True)
class AgentStep:
    """A step of the agent."""

    index: int
    """Start counting from 1"""
    action: AIMessage
    """AI message output from the agent. May or may not contain tool calls."""
    observations: list[ToolMessage]
    """Any observations made through tool calls."""

    def __post_init__(self) -> None:
        if self.index <= 0:
            msg = "index must be positive"
            raise ValueError(msg)


@dataclass(frozen=True)
class AgentTrajectory:
    """A trajectory of the agent."""

    steps: list[AgentStep]
    files: dict[str, str]


def _trajectory_from_result(result: Mapping[str, object]) -> AgentTrajectory:
    steps: list[AgentStep] = []
    current_step: AgentStep | None = None

    messages_obj = result.get("messages")
    if not isinstance(messages_obj, list):
        msg = f"Expected result['messages'] to be list, got {type(messages_obj)}"
        raise TypeError(msg)

    for msg_obj in messages_obj[1:]:
        if isinstance(msg_obj, AIMessage):
            if current_step is not None:
                steps.append(current_step)
            current_step = AgentStep(index=len(steps) + 1, action=msg_obj, observations=[])
        elif isinstance(msg_obj, ToolMessage):
            if current_step is not None:
                current_step.observations.append(msg_obj)

    if current_step is not None:
        steps.append(current_step)

    return AgentTrajectory(
        steps=steps,
        files=_coerce_result_files_to_strings(result.get("files")),
    )


def _assert_expectations(trajectory: AgentTrajectory, expect: TrajectoryExpectations) -> None:
    agent_steps = len(trajectory.steps)
    tool_call_requests = sum(len(step.action.tool_calls) for step in trajectory.steps)

    t.log_feedback(key="agent_steps", value=agent_steps)
    t.log_feedback(key="tool_call_requests", value=tool_call_requests)

    if expect.num_agent_steps is not None:
        t.log_feedback(
            key="match_num_agent_steps",
            value=int(agent_steps == expect.num_agent_steps),
        )
        t.log_feedback(key="expected_num_agent_steps", value=expect.num_agent_steps)
        assert agent_steps == expect.num_agent_steps
    if expect.num_tool_call_requests is not None:
        t.log_feedback(
            key="match_num_tool_call_requests",
            value=int(tool_call_requests == expect.num_tool_call_requests),
        )
        t.log_feedback(key="expected_num_tool_call_requests", value=expect.num_tool_call_requests)
        assert tool_call_requests == expect.num_tool_call_requests


def run_agent(
    agent: CompiledStateGraph[Any, Any],
    *,
    query: str,
    initial_files: dict[str, str] | None = None,
    expect: TrajectoryExpectations | None = None,
) -> AgentTrajectory:
    """Run agent eval against the given query."""
    inputs: dict[str, object] = {"messages": [{"role": "user", "content": query}]}
    if initial_files is not None:
        inputs["files"] = {path: create_file_data(content) for path, content in initial_files.items()}
    thread_id = uuid.uuid4()
    config = {"configurable": {"thread_id": thread_id}}
    t.log_inputs(inputs)
    result = agent.invoke(inputs, config)
    t.log_outputs(result)

    if not isinstance(result, Mapping):
        msg = f"Expected invoke result to be Mapping, got {type(result)}"
        raise TypeError(msg)

    trajectory = _trajectory_from_result(result)
    if expect is not None:
        _assert_expectations(trajectory, expect)
    return trajectory
