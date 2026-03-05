from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langsmith import testing as t

from deepagents.backends.utils import create_file_data, file_data_to_string

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


# ---------------------------------------------------------------------------
# Core trajectory data structures
# ---------------------------------------------------------------------------


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
        """Return the text content of the last agent step."""
        return self.steps[-1].action.text

    def pretty(self) -> str:
        """Return a human-readable summary of the trajectory."""
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


# ---------------------------------------------------------------------------
# Assertion base classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuccessAssertion:
    """Base for correctness assertions that fail the test when violated."""

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Return ``True`` when the assertion holds.

        Args:
            trajectory: The agent trajectory to check.

        Returns:
            Whether the assertion passed.

        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Return a human-readable explanation of why the check failed.

        Args:
            trajectory: The agent trajectory that failed the check.

        Returns:
            A description of the failure.

        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError

    @property
    def feedback_key(self) -> str:
        """Return a short key used when logging feedback to LangSmith.

        Returns:
            The feedback key string.

        Raises:
            NotImplementedError: Subclasses must override this property.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class EfficiencyAssertion:
    """Base for trajectory-shape assertions that are logged but never fail."""

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Return ``True`` when the assertion holds.

        Args:
            trajectory: The agent trajectory to check.

        Returns:
            Whether the assertion passed.

        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Return a human-readable explanation of why the check failed.

        Args:
            trajectory: The agent trajectory that failed the check.

        Returns:
            A description of the failure.

        Raises:
            NotImplementedError: Subclasses must override this method.
        """
        raise NotImplementedError

    @property
    def feedback_key(self) -> str:
        """Return a short key used when logging feedback to LangSmith.

        Returns:
            The feedback key string.

        Raises:
            NotImplementedError: Subclasses must override this property.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_common_zero_width(text: str) -> str:
    """Remove common zero-width characters that can break string comparisons.

    Some models insert invisible Unicode characters (e.g., zero-width spaces) into
    strings that look like paths or URLs. This is typically a formatting heuristic
    to prevent auto-linking or to stabilize rendering in downstream UIs.

    Our eval expectations are literal substring checks. Stripping these characters
    makes the checks robust to formatting-only differences while preserving the
    semantic content of the answer.

    Args:
        text: The input string.

    Returns:
        The string with zero-width characters removed.
    """
    return text.translate(
        {
            ord("\u200b"): None,  # zero-width space
            ord("\u200c"): None,  # zero-width non-joiner
            ord("\u200d"): None,  # zero-width joiner
            ord("\ufeff"): None,  # zero-width no-break space / BOM
        }
    )


def _coerce_result_files_to_strings(raw_files: object) -> dict[str, str]:
    """Coerce the ``files`` value from an agent result into ``dict[str, str]``.

    Args:
        raw_files: The raw files object from the agent result.

    Returns:
        A mapping of file paths to their string contents.

    Raises:
        TypeError: If the files object has an unexpected type.
    """
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


# ---------------------------------------------------------------------------
# Concrete success assertions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FinalTextContains(SuccessAssertion):
    """Assert that the final agent text contains a given substring.

    Attributes:
        text: The substring to look for.
        case_insensitive: Whether the comparison should ignore case.
    """

    text: str
    case_insensitive: bool = False

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Check that the final step text contains ``self.text``.

        Args:
            trajectory: The agent trajectory to check.

        Returns:
            Whether the final text contains the expected substring.
        """
        haystack = _strip_common_zero_width(trajectory.steps[-1].action.text)
        needle = _strip_common_zero_width(self.text)
        if self.case_insensitive:
            haystack = haystack.lower()
            needle = needle.lower()
        return needle in haystack

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Describe why the final-text-contains check failed.

        Args:
            trajectory: The agent trajectory that failed the check.

        Returns:
            A human-readable failure description.
        """
        final_text = _strip_common_zero_width(trajectory.steps[-1].action.text)
        return (
            f"Expected final text to contain {self.text!r} "
            f"(case_insensitive={self.case_insensitive}), got: {final_text!r}"
        )

    @property
    def feedback_key(self) -> str:
        """Return the feedback key for this assertion.

        Returns:
            A feedback key string.
        """
        return f"success:final_text_contains:{self.text}"


@dataclass(frozen=True)
class FinalTextExcludes(SuccessAssertion):
    """Assert that the final agent text does NOT contain a given substring.

    Attributes:
        text: The substring that must be absent.
        case_insensitive: Whether the comparison should ignore case.
    """

    text: str
    case_insensitive: bool = False

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Check that the final step text does not contain ``self.text``.

        Args:
            trajectory: The agent trajectory to check.

        Returns:
            Whether the final text excludes the expected substring.
        """
        haystack = _strip_common_zero_width(trajectory.steps[-1].action.text)
        needle = _strip_common_zero_width(self.text)
        if self.case_insensitive:
            haystack = haystack.lower()
            needle = needle.lower()
        return needle not in haystack

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Describe why the final-text-excludes check failed.

        Args:
            trajectory: The agent trajectory that failed the check.

        Returns:
            A human-readable failure description.
        """
        final_text = _strip_common_zero_width(trajectory.steps[-1].action.text)
        return (
            f"Expected final text NOT to contain {self.text!r} "
            f"(case_insensitive={self.case_insensitive}), got: {final_text!r}"
        )

    @property
    def feedback_key(self) -> str:
        """Return the feedback key for this assertion.

        Returns:
            A feedback key string.
        """
        return f"success:final_text_excludes:{self.text}"


@dataclass(frozen=True)
class FileEquals(SuccessAssertion):
    """Assert that a file in the trajectory has exactly the expected content.

    Attributes:
        path: The file path to check.
        content: The expected full content of the file.
    """

    path: str
    content: str

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Check that the file at ``self.path`` equals ``self.content``.

        Args:
            trajectory: The agent trajectory to check.

        Returns:
            Whether the file content matches exactly.
        """
        return trajectory.files.get(self.path) == self.content

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Describe why the file-equals check failed.

        Args:
            trajectory: The agent trajectory that failed the check.

        Returns:
            A human-readable failure description.
        """
        actual = trajectory.files.get(self.path)
        if actual is None:
            return f"File {self.path!r} not found in trajectory files"
        return (
            f"File {self.path!r} content mismatch.\n"
            f"Expected:\n{self.content!r}\n"
            f"Actual:\n{actual!r}"
        )

    @property
    def feedback_key(self) -> str:
        """Return the feedback key for this assertion.

        Returns:
            A feedback key string.
        """
        return f"success:file_equals:{self.path}"


@dataclass(frozen=True)
class FileContains(SuccessAssertion):
    """Assert that a file in the trajectory contains a given substring.

    Attributes:
        path: The file path to check.
        substring: The substring to look for.
    """

    path: str
    substring: str

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Check that the file at ``self.path`` contains ``self.substring``.

        Args:
            trajectory: The agent trajectory to check.

        Returns:
            Whether the file content contains the substring.
        """
        file_content = trajectory.files.get(self.path)
        if file_content is None:
            return False
        return self.substring in file_content

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Describe why the file-contains check failed.

        Args:
            trajectory: The agent trajectory that failed the check.

        Returns:
            A human-readable failure description.
        """
        actual = trajectory.files.get(self.path)
        if actual is None:
            return f"File {self.path!r} not found in trajectory files"
        return (
            f"File {self.path!r} does not contain {self.substring!r}.\n"
            f"Actual content:\n{actual!r}"
        )

    @property
    def feedback_key(self) -> str:
        """Return the feedback key for this assertion.

        Returns:
            A feedback key string.
        """
        return f"success:file_contains:{self.path}"


@dataclass(frozen=True)
class FileExcludes(SuccessAssertion):
    """Assert that a file in the trajectory does NOT contain a given substring.

    Attributes:
        path: The file path to check.
        substring: The substring that must be absent.
    """

    path: str
    substring: str

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Check that the file at ``self.path`` does not contain ``self.substring``.

        Args:
            trajectory: The agent trajectory to check.

        Returns:
            Whether the file content excludes the substring.
        """
        return self.substring not in trajectory.files.get(self.path, "")

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Describe why the file-excludes check failed.

        Args:
            trajectory: The agent trajectory that failed the check.

        Returns:
            A human-readable failure description.
        """
        actual = trajectory.files.get(self.path, "")
        return (
            f"File {self.path!r} unexpectedly contains {self.substring!r}.\n"
            f"Actual content:\n{actual!r}"
        )

    @property
    def feedback_key(self) -> str:
        """Return the feedback key for this assertion.

        Returns:
            A feedback key string.
        """
        return f"success:file_excludes:{self.path}"


# ---------------------------------------------------------------------------
# Concrete efficiency assertions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentSteps(EfficiencyAssertion):
    """Assert that the trajectory has exactly ``n`` agent steps.

    Attributes:
        n: Expected number of agent steps.
    """

    n: int

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Check that the trajectory has exactly ``self.n`` steps.

        Args:
            trajectory: The agent trajectory to check.

        Returns:
            Whether the step count matches.
        """
        return len(trajectory.steps) == self.n

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Describe why the agent-steps check failed.

        Args:
            trajectory: The agent trajectory that failed the check.

        Returns:
            A human-readable failure description.
        """
        return (
            f"Expected {self.n} agent steps, got {len(trajectory.steps)}"
        )

    @property
    def feedback_key(self) -> str:
        """Return the feedback key for this assertion.

        Returns:
            A feedback key string.
        """
        return "expect:agent_steps"


@dataclass(frozen=True)
class ToolCallRequests(EfficiencyAssertion):
    """Assert that the trajectory has exactly ``n`` total tool call requests.

    Attributes:
        n: Expected total number of tool call requests.
    """

    n: int

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Check that total tool call requests equal ``self.n``.

        Args:
            trajectory: The agent trajectory to check.

        Returns:
            Whether the tool call count matches.
        """
        actual = sum(len(s.action.tool_calls) for s in trajectory.steps)
        return actual == self.n

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Describe why the tool-call-requests check failed.

        Args:
            trajectory: The agent trajectory that failed the check.

        Returns:
            A human-readable failure description.
        """
        actual = sum(len(s.action.tool_calls) for s in trajectory.steps)
        return (
            f"Expected {self.n} tool call requests, got {actual}"
        )

    @property
    def feedback_key(self) -> str:
        """Return the feedback key for this assertion.

        Returns:
            A feedback key string.
        """
        return "expect:tool_call_requests"


@dataclass(frozen=True)
class ToolCall(EfficiencyAssertion):
    """Assert that a specific tool call occurred in the trajectory.

    When ``step`` is ``None``, all steps are searched. When ``step`` is given,
    only that step (1-indexed) is checked.

    Attributes:
        name: Expected tool name.
        step: Optional 1-indexed step to restrict the search to.
        args_contains: If set, the tool call args must contain these key-value pairs.
        args_equals: If set, the tool call args must equal this dict exactly.
    """

    name: str
    step: int | None = None
    args_contains: dict[str, object] | None = None
    args_equals: dict[str, object] | None = None

    def check(self, trajectory: AgentTrajectory) -> bool:
        """Check that a matching tool call exists in the trajectory.

        Args:
            trajectory: The agent trajectory to check.

        Returns:
            Whether a matching tool call was found.
        """
        return bool(self._find_matches(trajectory))

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        """Describe why the tool-call check failed.

        Args:
            trajectory: The agent trajectory that failed the check.

        Returns:
            A human-readable failure description.
        """
        step_desc = f" in step {self.step}" if self.step is not None else ""
        return (
            f"Missing expected tool call{step_desc}: "
            f"name={self.name!r}, "
            f"args_contains={self.args_contains!r}, "
            f"args_equals={self.args_equals!r}"
        )

    @property
    def feedback_key(self) -> str:
        """Return the feedback key for this assertion.

        Returns:
            A feedback key string.
        """
        return f"expect:tool_call:{self.name}"

    def _matches_tool_call(self, tc: dict[str, object]) -> bool:
        """Check whether a single tool call dict matches this expectation.

        Args:
            tc: A tool call dictionary with ``name`` and ``args`` keys.

        Returns:
            Whether the tool call matches.
        """
        if tc.get("name") != self.name:
            return False
        if self.args_contains is not None:
            args = tc.get("args")
            if not isinstance(args, dict):
                return False
            if not all(args.get(k) == v for k, v in self.args_contains.items()):
                return False
        return self.args_equals is None or tc.get("args") == self.args_equals

    def _find_matches(self, trajectory: AgentTrajectory) -> list[dict[str, object]]:
        """Find tool calls matching this expectation.

        Args:
            trajectory: The agent trajectory to search.

        Returns:
            A list of matching tool call dicts.
        """
        if self.step is not None:
            if self.step > len(trajectory.steps):
                return []
            steps_to_search = [trajectory.steps[self.step - 1]]
        else:
            steps_to_search = trajectory.steps

        return [
            tc
            for s in steps_to_search
            for tc in s.action.tool_calls
            if self._matches_tool_call(tc)
        ]


# ---------------------------------------------------------------------------
# Factory functions (public API)
# ---------------------------------------------------------------------------


def final_text_contains(
    text: str,
    *,
    case_insensitive: bool = False,
) -> FinalTextContains:
    """Create a ``FinalTextContains`` success assertion.

    Args:
        text: The substring to look for in the final agent text.
        case_insensitive: Whether the comparison should ignore case.

    Returns:
        A ``FinalTextContains`` assertion instance.
    """
    return FinalTextContains(text=text, case_insensitive=case_insensitive)


def final_text_excludes(
    text: str,
    *,
    case_insensitive: bool = False,
) -> FinalTextExcludes:
    """Create a ``FinalTextExcludes`` success assertion.

    Args:
        text: The substring that must be absent from the final agent text.
        case_insensitive: Whether the comparison should ignore case.

    Returns:
        A ``FinalTextExcludes`` assertion instance.
    """
    return FinalTextExcludes(text=text, case_insensitive=case_insensitive)


def file_equals(path: str, content: str) -> FileEquals:
    """Create a ``FileEquals`` success assertion.

    Args:
        path: The file path to check.
        content: The expected full content of the file.

    Returns:
        A ``FileEquals`` assertion instance.
    """
    return FileEquals(path=path, content=content)


def file_contains(path: str, substring: str) -> FileContains:
    """Create a ``FileContains`` success assertion.

    Args:
        path: The file path to check.
        substring: The substring to look for.

    Returns:
        A ``FileContains`` assertion instance.
    """
    return FileContains(path=path, substring=substring)


def file_excludes(path: str, substring: str) -> FileExcludes:
    """Create a ``FileExcludes`` success assertion.

    Args:
        path: The file path to check.
        substring: The substring that must be absent.

    Returns:
        A ``FileExcludes`` assertion instance.
    """
    return FileExcludes(path=path, substring=substring)


def agent_steps(n: int) -> AgentSteps:
    """Create an ``AgentSteps`` efficiency assertion.

    Args:
        n: Expected number of agent steps.

    Returns:
        An ``AgentSteps`` assertion instance.
    """
    return AgentSteps(n=n)


def tool_call_requests(n: int) -> ToolCallRequests:
    """Create a ``ToolCallRequests`` efficiency assertion.

    Args:
        n: Expected total number of tool call requests.

    Returns:
        A ``ToolCallRequests`` assertion instance.
    """
    return ToolCallRequests(n=n)


def tool_call(
    name: str,
    *,
    step: int | None = None,
    args_contains: dict[str, object] | None = None,
    args_equals: dict[str, object] | None = None,
) -> ToolCall:
    """Create a ``ToolCall`` efficiency assertion.

    Args:
        name: Expected tool name.
        step: Optional 1-indexed step to restrict the search to.
        args_contains: If set, the tool call args must contain these key-value pairs.
        args_equals: If set, the tool call args must equal this dict exactly.

    Returns:
        A ``ToolCall`` assertion instance.
    """
    return ToolCall(
        name=name,
        step=step,
        args_contains=args_contains,
        args_equals=args_equals,
    )


# ---------------------------------------------------------------------------
# TrajectoryExpectations (two-tier builder)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryExpectations:
    """Two-tier assertion container for agent trajectories.

    Use ``.success()`` to add correctness assertions (hard-fail) and
    ``.expect()`` to add efficiency assertions (logged but never fail).

    Attributes:
        _success: Tuple of success assertions.
        _expectations: Tuple of efficiency assertions.
    """

    _success: tuple[SuccessAssertion, ...] = ()
    _expectations: tuple[EfficiencyAssertion, ...] = ()

    def success(self, *assertions: SuccessAssertion) -> TrajectoryExpectations:
        """Append correctness assertions that hard-fail the test when violated.

        Args:
            *assertions: One or more ``SuccessAssertion`` instances.

        Returns:
            A new ``TrajectoryExpectations`` with the assertions appended.
        """
        return TrajectoryExpectations(
            _success=(*self._success, *assertions),
            _expectations=self._expectations,
        )

    def expect(self, *assertions: EfficiencyAssertion) -> TrajectoryExpectations:
        """Append efficiency assertions that are logged but never fail.

        Args:
            *assertions: One or more ``EfficiencyAssertion`` instances.

        Returns:
            A new ``TrajectoryExpectations`` with the assertions appended.
        """
        return TrajectoryExpectations(
            _success=self._success,
            _expectations=(*self._expectations, *assertions),
        )


# ---------------------------------------------------------------------------
# Internal: trajectory construction & assertion runner
# ---------------------------------------------------------------------------


def _trajectory_from_result(result: Mapping[str, object]) -> AgentTrajectory:
    """Build an ``AgentTrajectory`` from a raw agent invoke result.

    Args:
        result: The mapping returned by ``agent.invoke()``.

    Returns:
        The constructed ``AgentTrajectory``.

    Raises:
        TypeError: If ``result['messages']`` is not a list.
    """
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


def _assert_expectations(
    trajectory: AgentTrajectory,
    expect: TrajectoryExpectations,
) -> None:
    """Run all assertions in *expect* against *trajectory*.

    Success assertions hard-fail the test via ``pytest.fail``. Efficiency
    assertions are logged as feedback but never cause a test failure.

    All assertions are logged to LangSmith regardless of outcome.

    Args:
        trajectory: The agent trajectory to validate.
        expect: The two-tier expectation container.
    """
    # Always log actual counts as feedback
    actual_steps = len(trajectory.steps)
    actual_tool_calls = sum(len(s.action.tool_calls) for s in trajectory.steps)
    t.log_feedback(key="agent_steps", value=actual_steps)
    t.log_feedback(key="tool_call_requests", value=actual_tool_calls)

    # Hard correctness checks
    for assertion in expect._success:
        result = assertion.check(trajectory)
        t.log_feedback(key=assertion.feedback_key, value=int(result))
        if not result:
            pytest.fail(
                f"success check failed: {assertion.describe_failure(trajectory)}\n\n"
                f"trajectory:\n{trajectory.pretty()}",
                pytrace=False,
            )

    # Soft efficiency checks (log only, never fail)
    for assertion in expect._expectations:
        result = assertion.check(trajectory)
        t.log_feedback(key=assertion.feedback_key, value=int(result))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_agent(
    agent: CompiledStateGraph[Any, Any],
    *,
    query: str | list[AnyMessage],
    model: BaseChatModel,
    initial_files: dict[str, str] | None = None,
    expect: TrajectoryExpectations | None = None,
    thread_id: str | None = None,
) -> AgentTrajectory:
    """Run agent eval against the given query.

    Args:
        agent: The compiled state graph to invoke.
        query: A string prompt or list of messages.
        model: The chat model (used for logging only).
        initial_files: Optional initial files to seed the agent with.
        expect: Optional trajectory expectations to validate.
        thread_id: Optional thread ID for the invocation.

    Returns:
        The resulting ``AgentTrajectory``.

    Raises:
        TypeError: If the invoke result is not a ``Mapping``.
    """
    if isinstance(query, str):
        invoke_inputs: dict[str, object] = {"messages": [{"role": "user", "content": query}]}
    else:
        invoke_inputs = {"messages": query}
    if initial_files is not None:
        invoke_inputs["files"] = {path: create_file_data(content) for path, content in initial_files.items()}

    if thread_id is None:
        thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    logged_inputs = dict(invoke_inputs)
    logged_inputs["model"] = str(getattr(model, "model", None) or getattr(model, "model_name", ""))

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
