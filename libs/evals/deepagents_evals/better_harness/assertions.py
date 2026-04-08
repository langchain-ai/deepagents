"""Repo-local assertion and trajectory helpers for the better-harness runner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from deepagents.backends.utils import file_data_to_string
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class AgentStep:
    """A single agent step consisting of one model action and tool observations."""

    index: int
    action: AIMessage
    observations: list[ToolMessage]


@dataclass(frozen=True)
class AgentTrajectory:
    """A compact representation of an agent run."""

    steps: list[AgentStep]
    files: dict[str, str]

    @property
    def answer(self) -> str:
        """Return the final agent text."""
        return self.steps[-1].action.text if self.steps else ""

    def pretty(self) -> str:
        """Return a human-readable trace summary."""
        lines: list[str] = []
        for step in self.steps:
            lines.append(f"step {step.index}:")
            lines.extend(
                [
                    f"  - {tool_call.get('name')} {tool_call.get('args')}"
                    for tool_call in step.action.tool_calls
                ]
            )
            if step.action.text.strip():
                text_preview = step.action.text.strip().replace("\n", "\\n")
                lines.append(f"  text: {text_preview}")
        return "\n".join(lines)


class SuccessAssertion:
    """Base class for correctness assertions."""

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Return whether the assertion passed."""
        raise NotImplementedError

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Return a human-readable failure message."""
        raise NotImplementedError


class EfficiencyAssertion:
    """Base class for non-failing trajectory expectations."""

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Return whether the expectation was met."""
        raise NotImplementedError

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Return a human-readable expectation failure."""
        raise NotImplementedError


def _strip_common_zero_width(text: str) -> str:
    """Remove common zero-width characters from text comparisons."""
    return text.translate(
        {
            ord("\u200b"): None,
            ord("\u200c"): None,
            ord("\u200d"): None,
            ord("\ufeff"): None,
        }
    )


def _coerce_result_files_to_strings(raw_files: object) -> dict[str, str]:
    """Normalize agent result files into plain strings."""
    if raw_files is None:
        return {}
    if not isinstance(raw_files, Mapping):
        msg = f"Expected files to be dict-like, got {type(raw_files)!r}"
        raise TypeError(msg)

    files: dict[str, str] = {}
    for path, file_data in raw_files.items():
        if not isinstance(path, str):
            msg = f"Expected file path to be str, got {type(path)!r}"
            raise TypeError(msg)
        if isinstance(file_data, str):
            files[path] = file_data
            continue
        if isinstance(file_data, Mapping) and "content" in file_data:
            files[path] = file_data_to_string(dict(file_data))
            continue
        msg = f"Unexpected file representation for {path}: {type(file_data)!r}"
        raise TypeError(msg)
    return files


@dataclass(frozen=True)
class FinalTextContains(SuccessAssertion):
    """Assert that the final text contains a substring."""

    text: str
    case_insensitive: bool = False

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Return whether the final text contains `self.text`."""
        haystack = _strip_common_zero_width(trajectory.answer)
        needle = _strip_common_zero_width(self.text)
        if self.case_insensitive:
            haystack = haystack.lower()
            needle = needle.lower()
        return needle in haystack

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Describe the mismatch."""
        return (
            f"Expected final text to contain {self.text!r} "
            f"(case_insensitive={self.case_insensitive}), got {trajectory.answer!r}"
        )


@dataclass(frozen=True)
class ToolCallExpectation(EfficiencyAssertion):
    """Expect a tool call to appear in the trajectory."""

    name: str
    step: int | None = None
    args_contains: dict[str, object] | None = None
    args_equals: dict[str, object] | None = None

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Return whether a matching tool call exists."""
        return bool(self._find_matches(trajectory))

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Describe the missing tool call."""
        _ = trajectory
        return (
            f"Missing expected tool call: name={self.name!r}, step={self.step!r}, "
            f"args_contains={self.args_contains!r}, args_equals={self.args_equals!r}"
        )

    def _matches_tool_call(self, tool_call: dict[str, object]) -> bool:
        if tool_call.get("name") != self.name:
            return False
        if self.args_contains is not None:
            args = tool_call.get("args")
            if not isinstance(args, dict):
                return False
            if not all(args.get(key) == value for key, value in self.args_contains.items()):
                return False
        return self.args_equals is None or tool_call.get("args") == self.args_equals

    def _find_matches(self, trajectory: AgentTrajectory) -> list[dict[str, object]]:
        steps = trajectory.steps
        if self.step is not None:
            if self.step > len(steps):
                return []
            steps = [steps[self.step - 1]]
        return [
            tool_call
            for step in steps
            for tool_call in step.action.tool_calls
            if self._matches_tool_call(tool_call)
        ]


@dataclass(frozen=True)
class AgentSteps(EfficiencyAssertion):
    """Expect a specific number of agent steps."""

    n: int

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Return whether the step count matches."""
        return len(trajectory.steps) == self.n

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Describe the mismatch."""
        return f"Expected {self.n} steps, got {len(trajectory.steps)}"


@dataclass(frozen=True)
class ToolCallRequests(EfficiencyAssertion):
    """Expect a specific number of tool-call requests."""

    n: int

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Return whether the tool-call count matches."""
        return sum(len(step.action.tool_calls) for step in trajectory.steps) == self.n

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Describe the mismatch."""
        actual = sum(len(step.action.tool_calls) for step in trajectory.steps)
        return f"Expected {self.n} tool-call requests, got {actual}"


@dataclass(frozen=True)
class TrajectoryScorer:
    """Two-tier assertion container for benchmark cases."""

    _success: tuple[SuccessAssertion, ...] = ()
    _expectations: tuple[EfficiencyAssertion, ...] = ()

    def success(self, *assertions: SuccessAssertion) -> TrajectoryScorer:
        """Return a new scorer with additional correctness assertions."""
        return TrajectoryScorer(
            _success=(*self._success, *assertions),
            _expectations=self._expectations,
        )

    def expect(
        self,
        *,
        agent_steps: int | None = None,
        tool_call_requests: int | None = None,
        tool_calls: list[ToolCallExpectation] | None = None,
    ) -> TrajectoryScorer:
        """Return a new scorer with additional non-failing expectations."""
        expectations: list[EfficiencyAssertion] = list(self._expectations)
        if agent_steps is not None:
            expectations.append(AgentSteps(agent_steps))
        if tool_call_requests is not None:
            expectations.append(ToolCallRequests(tool_call_requests))
        if tool_calls is not None:
            expectations.extend(tool_calls)
        return TrajectoryScorer(_success=self._success, _expectations=tuple(expectations))


@dataclass
class LLMJudge(SuccessAssertion):
    """Grade a trajectory against one or more natural-language criteria."""

    criteria: tuple[str, ...]
    judge_model: str = "claude-sonnet-4-6"
    include_tool_calls: bool = False
    _last_results: list[dict[str, Any]] | None = field(
        default=None,
        repr=False,
        compare=False,
        hash=False,
    )

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Return whether all criteria passed."""
        results = self._grade(trajectory)
        self._last_results = results
        return all(result["score"] for result in results)

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Describe which criteria failed."""
        results = self._last_results if self._last_results is not None else self._grade(trajectory)
        failed = [(index, result) for index, result in enumerate(results, 1) if not result["score"]]
        reasons = [
            f"Criteria {index}: {result.get('comment') or 'no reason'}"
            for index, result in failed
        ]
        return f"{len(failed)}/{len(results)} criteria failed — " + "; ".join(reasons)

    def _serialize(self, trajectory: AgentTrajectory) -> str:
        if self.include_tool_calls:
            return trajectory.pretty()
        return "\n\n".join(
            f"[Agent]: {step.action.text}" for step in trajectory.steps if step.action.text
        )

    def _grade(self, trajectory: AgentTrajectory) -> list[dict[str, Any]]:
        conversation = self._serialize(trajectory)
        if not conversation.strip():
            msg = "Cannot grade trajectory with no text content"
            raise ValueError(msg)

        prompt = (
            """You are a strict grading assistant. You will receive an
agent trajectory (a sequence of steps) and a single criterion. Decide whether
the agent's trajectory satisfies the criterion.

<criterion>
{criterion}
</criterion>

<agent_trajectory>
{outputs}
</agent_trajectory>"""
            if self.include_tool_calls
            else """You are a strict grading assistant. You will receive a
series of agent responses and a single criterion. Decide whether the agent's
responses satisfy the criterion.

<criterion>
{criterion}
</criterion>

<agent_responses>
{outputs}
</agent_responses>"""
        )
        judge = init_chat_model(
            self.judge_model,
            temperature=0,
            timeout=90,
        ).with_structured_output(JudgeVerdict)
        results: list[dict[str, Any]] = []
        for criterion in self.criteria:
            verdict = judge.invoke(
                prompt.format(
                    criterion=criterion,
                    outputs=conversation,
                )
            )
            results.append({"score": verdict.passed, "comment": verdict.comment})
        return results


def final_text_contains(
    text: str,
    *,
    case_insensitive: bool = False,
) -> FinalTextContains:
    """Create a final-text-contains assertion."""
    return FinalTextContains(text=text, case_insensitive=case_insensitive)


def tool_call(
    name: str,
    *,
    step: int | None = None,
    args_contains: dict[str, object] | None = None,
    args_equals: dict[str, object] | None = None,
) -> ToolCallExpectation:
    """Create a tool-call expectation."""
    return ToolCallExpectation(
        name=name,
        step=step,
        args_contains=args_contains,
        args_equals=args_equals,
    )


def llm_judge(
    *criteria: str,
    judge_model: str = "claude-sonnet-4-6",
    include_tool_calls: bool = False,
) -> LLMJudge:
    """Create an LLM judge assertion."""
    return LLMJudge(
        criteria=tuple(criteria),
        judge_model=judge_model,
        include_tool_calls=include_tool_calls,
    )


def trajectory_from_result(result: Mapping[str, object]) -> AgentTrajectory:
    """Build an `AgentTrajectory` from an agent invoke result."""
    messages_obj = result.get("messages")
    if not isinstance(messages_obj, list):
        msg = f"Expected result['messages'] to be list, got {type(messages_obj)!r}"
        raise TypeError(msg)

    steps: list[AgentStep] = []
    current_step: AgentStep | None = None
    for message in messages_obj[1:]:
        if isinstance(message, AIMessage):
            if current_step is not None:
                steps.append(current_step)
            current_step = AgentStep(index=len(steps) + 1, action=message, observations=[])
        elif isinstance(message, ToolMessage) and current_step is not None:
            current_step.observations.append(message)

    if current_step is not None:
        steps.append(current_step)

    return AgentTrajectory(
        steps=steps,
        files=_coerce_result_files_to_strings(result.get("files")),
    )
class JudgeVerdict(BaseModel):
    """Structured output schema for the built-in LLM judge."""

    passed: bool = Field(description="Whether the criterion is satisfied.")
    comment: str = Field(description="Brief explanation of the decision.")

