from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langsmith import testing as t

from deepagents.backends.utils import create_file_data, file_data_to_string


@dataclass(frozen=True)
class ToolCallExpectation:
    step: int
    name: str
    args_contains: dict[str, object] | None = None
    args_equals: dict[str, object] | None = None


if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


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
    tool_calls: tuple[ToolCallExpectation, ...] = ()
    final_text_contains: tuple[tuple[str, bool], ...] = ()

    def require_tool_call(
        self,
        *,
        step: int,
        name: str,
        args_contains: dict[str, object] | None = None,
        args_equals: dict[str, object] | None = None,
    ) -> TrajectoryExpectations:
        if step <= 0:
            msg = "step must be positive"
            raise ValueError(msg)
        if args_contains is not None and args_equals is not None:
            msg = "Only one of args_contains or args_equals may be set"
            raise ValueError(msg)
        return TrajectoryExpectations(
            num_agent_steps=self.num_agent_steps,
            num_tool_call_requests=self.num_tool_call_requests,
            final_text_contains=self.final_text_contains,
            tool_calls=(
                *self.tool_calls,
                ToolCallExpectation(
                    step=step,
                    name=name,
                    args_contains=args_contains,
                    args_equals=args_equals,
                ),
            ),
        )

    def require_final_text_contains(
        self,
        text: str,
        *,
        case_insensitive: bool = False,
    ) -> TrajectoryExpectations:
        return TrajectoryExpectations(
            num_agent_steps=self.num_agent_steps,
            num_tool_call_requests=self.num_tool_call_requests,
            tool_calls=self.tool_calls,
            final_text_contains=(*self.final_text_contains, (text, case_insensitive)),
        )


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

    @property
    def answer(self) -> str:
        return self.steps[-1].action.text

    def pretty(self) -> str:
        lines: list[str] = []
        for step in self.steps:
            lines.append(f"step {step.index}:")
            tool_calls = step.action.tool_calls
            if tool_calls:
                for tc in tool_calls:
                    name = tc.get("name")
                    args = tc.get("args")
                    lines.append(f"  - {name} {args}")
            else:
                text = step.action.text
                text_preview = text.strip().replace("\n", "\\n")
                lines.append(f"  text: {text_preview}")
        return "\n".join(lines)


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


def _assert_counts(trajectory: AgentTrajectory, expect: TrajectoryExpectations) -> None:
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
        if agent_steps != expect.num_agent_steps:
            pytest.fail(
                f"num_agent_steps mismatch: expected={expect.num_agent_steps}, actual={agent_steps}\n\ntrajectory:\n{trajectory.pretty()}",
                pytrace=False,
            )

    if expect.num_tool_call_requests is not None:
        t.log_feedback(
            key="match_num_tool_call_requests",
            value=int(tool_call_requests == expect.num_tool_call_requests),
        )
        t.log_feedback(key="expected_num_tool_call_requests", value=expect.num_tool_call_requests)
        if tool_call_requests != expect.num_tool_call_requests:
            pytest.fail(
                "num_tool_call_requests mismatch: "
                f"expected={expect.num_tool_call_requests}, actual={tool_call_requests}\n\n"
                f"trajectory:\n{trajectory.pretty()}",
                pytrace=False,
            )


def _assert_final_text(trajectory: AgentTrajectory, expect: TrajectoryExpectations) -> None:
    final_text = trajectory.steps[-1].action.text
    for text, case_insensitive in expect.final_text_contains:
        haystack = final_text.lower() if case_insensitive else final_text
        needle = text.lower() if case_insensitive else text
        if needle not in haystack:
            msg = f"Expected final text to contain {text!r} (case_insensitive={case_insensitive}), got: {final_text!r}"
            raise AssertionError(msg)


def _assert_tool_calls(trajectory: AgentTrajectory, expect: TrajectoryExpectations) -> None:
    for requirement in expect.tool_calls:
        if requirement.step > len(trajectory.steps):
            msg = f"Expected at least {requirement.step} steps to validate tool call requirement, got {len(trajectory.steps)}"
            raise AssertionError(msg)

        step = trajectory.steps[requirement.step - 1]
        step_tool_calls = step.action.tool_calls

        matches: list[dict[str, object]] = [tc for tc in step_tool_calls if tc.get("name") == requirement.name]
        if requirement.args_contains is not None:
            matches = [
                tc for tc in matches if isinstance(tc.get("args"), dict) and all(tc["args"].get(k) == v for k, v in requirement.args_contains.items())
            ]
        if requirement.args_equals is not None:
            matches = [tc for tc in matches if tc.get("args") == requirement.args_equals]

        if not matches:
            msg = (
                "Missing expected tool call in step "
                f"{requirement.step}: name={requirement.name!r}, "
                f"args_contains={requirement.args_contains!r}, args_equals={requirement.args_equals!r}. "
                f"Actual tool calls: {step_tool_calls!r}"
            )
            raise AssertionError(msg)


def _assert_expectations(trajectory: AgentTrajectory, expect: TrajectoryExpectations) -> None:
    try:
        _assert_counts(trajectory, expect)
        _assert_final_text(trajectory, expect)
        _assert_tool_calls(trajectory, expect)
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"expectations failed: {e}\n\ntrajectory:\n{trajectory.pretty()}", pytrace=False)


def run_agent(
    agent: CompiledStateGraph[Any, Any],
    *,
    query: str,
    model: str,
    initial_files: dict[str, str] | None = None,
    expect: TrajectoryExpectations | None = None,
    thread_id: str | None = None,
) -> AgentTrajectory:
    """Run agent eval against the given query."""
    invoke_inputs: dict[str, object] = {"messages": [{"role": "user", "content": query}]}
    if initial_files is not None:
        invoke_inputs["files"] = {path: create_file_data(content) for path, content in initial_files.items()}

    if thread_id is None:
        thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    logged_inputs = dict(invoke_inputs)
    logged_inputs["model"] = model

    t.log_inputs(logged_inputs)
    result = agent.invoke(invoke_inputs, config)
    t.log_outputs(result)

    if not isinstance(result, Mapping):
        msg = f"Expected invoke result to be Mapping, got {type(result)}"
        raise TypeError(msg)

    trajectory = _trajectory_from_result(result)
    if expect is not None:
        _assert_expectations(trajectory, expect)
    return trajectory
