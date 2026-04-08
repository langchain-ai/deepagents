"""Benchmark cases and evaluation runner for better-harness experiments."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from deepagents import create_deep_agent
from deepagents.backends.utils import create_file_data
from langchain.chat_models import init_chat_model
from langchain_core.tools import BaseTool, tool

from deepagents_evals.better_harness.assertions import (
    TrajectoryScorer,
    final_text_contains,
    llm_judge,
    tool_call,
    trajectory_from_result,
)
from deepagents_evals.better_harness.variants import (
    EXAMPLE_BASE_AGENT_PROMPT,
    patched_base_agent_prompt,
    render_variant_base_prompt,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AnyMessage
    from langgraph.graph.state import CompiledStateGraph

Split = Literal["optimization", "holdout"]
logger = logging.getLogger(__name__)


@tool
def slack_send_dm(user_id: str, message: str) -> str:
    """Send a direct message to a user on Slack."""
    return f"Sent DM to {user_id}: {message}"


@tool
def slack_post_channel(channel: str, message: str) -> str:
    """Post a message to a Slack channel."""
    return f"Posted to #{channel}: {message}"


@tool
def github_create_issue(repo: str, title: str, body: str) -> str:
    """Create a new GitHub issue."""
    return f"Created issue '{title}' in {repo} — {body}"


@tool
def github_create_pr(repo: str, title: str, head: str, base: str) -> str:
    """Create a pull request on GitHub."""
    return f"Created PR '{title}' in {repo} ({head} -> {base})"


@tool
def gmail_send_email(to: str, subject: str, body: str) -> str:
    """Send an email via Gmail."""
    return f"Sent email to {to}: {subject} — {body}"


@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"


TOOL_SELECTION_TOOLS: tuple[BaseTool | Callable[..., Any], ...] = (
    slack_send_dm,
    slack_post_channel,
    github_create_issue,
    github_create_pr,
    gmail_send_email,
    web_search,
)


@dataclass(frozen=True)
class HarnessVariant:
    """A harness candidate represented as an ordered set of prompt modules."""

    module_names: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        """Return a stable identifier for the variant."""
        return "baseline" if not self.module_names else "+".join(self.module_names)

    def add_module(self, module_name: str) -> HarnessVariant:
        """Return a new variant with an additional module appended."""
        if module_name in self.module_names:
            return self
        return HarnessVariant(module_names=(*self.module_names, module_name))

    def render_prompt(self, module_prompts: Mapping[str, str]) -> str | None:
        """Render the prompt additions for this variant."""
        blocks = [module_prompts[name].strip() for name in self.module_names]
        rendered = "\n\n".join(block for block in blocks if block)
        return rendered or None


@dataclass(frozen=True)
class BetterHarnessCase:
    """A single eval case used by the better-harness optimizer."""

    case_id: str
    category: str
    split: Split
    query: str
    scorer: TrajectoryScorer
    tools: tuple[BaseTool | Callable[..., Any] | dict[str, Any], ...] = ()
    system_prompt: str | None = None
    initial_files: dict[str, str] = field(default_factory=dict)
    recursion_limit: int = 12

@dataclass
class CaseResult:
    """The result of running one benchmark case."""

    case_id: str
    category: str
    split: Split
    passed: bool
    failure: str | None
    duration_s: float
    step_count: int
    tool_call_count: int
    final_text: str
    trajectory: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize the case result into JSON-compatible data."""
        return asdict(self)


@dataclass
class SuiteResult:
    """Aggregate results for a set of benchmark cases."""

    variant: HarnessVariant
    split: Split
    cases: list[CaseResult]

    @property
    def total(self) -> int:
        """Return the number of cases in the suite."""
        return len(self.cases)

    @property
    def passed_count(self) -> int:
        """Return the number of passing cases."""
        return sum(1 for case in self.cases if case.passed)

    @property
    def pass_rate(self) -> float:
        """Return the pass rate for the suite."""
        if not self.cases:
            return 0.0
        return self.passed_count / len(self.cases)

    @property
    def mean_duration_s(self) -> float:
        """Return the mean case duration."""
        if not self.cases:
            return 0.0
        return sum(case.duration_s for case in self.cases) / len(self.cases)

    @property
    def mean_step_count(self) -> float:
        """Return the mean step count."""
        if not self.cases:
            return 0.0
        return sum(case.step_count for case in self.cases) / len(self.cases)

    def passed_case_ids(self) -> set[str]:
        """Return the IDs of all passing cases."""
        return {case.case_id for case in self.cases if case.passed}

    def failures_by_case(self) -> dict[str, str]:
        """Return failure messages keyed by case id."""
        return {
            case.case_id: case.failure or "unknown failure"
            for case in self.cases
            if not case.passed
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize the suite result into JSON-compatible data."""
        return {
            "variant": self.variant.key,
            "split": self.split,
            "passed_count": self.passed_count,
            "total": self.total,
            "pass_rate": self.pass_rate,
            "mean_duration_s": self.mean_duration_s,
            "mean_step_count": self.mean_step_count,
            "cases": [case.to_dict() for case in self.cases],
        }


def build_default_benchmark() -> list[BetterHarnessCase]:
    """Return the default optimization/holdout benchmark used in this repo."""
    return [
        BetterHarnessCase(
            case_id="tool_indirect_email_report",
            category="tool_use",
            split="optimization",
            query="Email the weekly status report to manager@company.com with subject 'Week 10 Status'",
            tools=TOOL_SELECTION_TOOLS,
            scorer=(
                TrajectoryScorer()
                .expect(
                    agent_steps=2,
                    tool_call_requests=1,
                    tool_calls=[
                        tool_call(
                            name="gmail_send_email",
                            args_contains={"to": "manager@company.com"},
                        )
                    ],
                )
                .success(final_text_contains("Week 10", case_insensitive=True))
            ),
        ),
        BetterHarnessCase(
            case_id="followup_vague_send_report",
            category="conversation",
            split="optimization",
            query="Send a report to my team every week",
            scorer=TrajectoryScorer().success(
                llm_judge(
                    "The agent should ask what the report should contain or what data to include.",
                    "The agent should ask how the report should be delivered (email, Slack, etc.).",
                    "The agent should NOT ask about scheduling details since the user already specified 'every week'.",
                )
            ),
        ),
        BetterHarnessCase(
            case_id="followup_vague_summarize_emails",
            category="conversation",
            split="optimization",
            query="I want you to summarize my email every day",
            scorer=TrajectoryScorer().success(
                llm_judge(
                    "The agent should ask about the preferred summary format or level of detail.",
                    "The agent should assume summaries apply to all emails and should NOT ask which emails to summarize.",
                    "The followup questions should remain concise and directly relevant.",
                )
            ),
        ),
        BetterHarnessCase(
            case_id="followup_detailed_calendar_brief",
            category="conversation",
            split="optimization",
            query="Every morning at 5am, look at my Google Calendar and send me a brief of what's upcoming for the day",
            scorer=TrajectoryScorer().success(
                llm_judge(
                    "The agent should ask exactly one followup question about how the user wants to receive the brief (email, Slack, SMS, etc.).",
                    "The agent should NOT ask about schedule timing or scope because those were already provided.",
                )
            ),
        ),
        BetterHarnessCase(
            case_id="tool_chain_issue_then_notify",
            category="tool_use",
            split="optimization",
            query=(
                "Create a GitHub issue in org/backend titled 'Fix memory leak' with body 'OOM in prod', "
                "then post a message to #incidents saying the issue was created"
            ),
            tools=TOOL_SELECTION_TOOLS,
            scorer=(
                TrajectoryScorer()
                .expect(
                    tool_calls=[
                        tool_call(name="github_create_issue"),
                        tool_call(name="slack_post_channel", args_contains={"channel": "incidents"}),
                    ]
                )
                .success(final_text_contains("memory leak", case_insensitive=True))
            ),
        ),
        BetterHarnessCase(
            case_id="tool_direct_slack_dm",
            category="tool_use",
            split="holdout",
            query="Send a Slack DM to user U12345 saying 'Hello from evals'",
            tools=TOOL_SELECTION_TOOLS,
            scorer=(
                TrajectoryScorer()
                .expect(
                    agent_steps=2,
                    tool_call_requests=1,
                    tool_calls=[tool_call(name="slack_send_dm", args_contains={"user_id": "U12345"})],
                )
                .success(final_text_contains("U12345", case_insensitive=True))
            ),
        ),
        BetterHarnessCase(
            case_id="tool_chain_search_then_email",
            category="tool_use",
            split="holdout",
            query="Search for 'LangGraph 0.3 release notes' and email a summary to team@co.com with subject 'LangGraph Update'",
            tools=TOOL_SELECTION_TOOLS,
            scorer=(
                TrajectoryScorer()
                .expect(
                    tool_calls=[
                        tool_call(name="web_search"),
                        tool_call(name="gmail_send_email", args_contains={"to": "team@co.com"}),
                    ],
                )
                .success(final_text_contains("team@co.com", case_insensitive=True))
            ),
        ),
        BetterHarnessCase(
            case_id="followup_vague_monitor_system",
            category="conversation",
            split="holdout",
            query="Monitor our production system and alert me if something goes wrong",
            scorer=TrajectoryScorer().success(
                llm_judge(
                    "The agent should ask what metrics or signals define 'something going wrong'.",
                    "The agent should ask how the user wants to be alerted (Slack, email, PagerDuty, etc.).",
                    "The agent should NOT assume specific thresholds without asking.",
                )
            ),
        ),
        BetterHarnessCase(
            case_id="followup_vague_customer_support",
            category="conversation",
            split="holdout",
            query="Help me respond to customer questions faster",
            scorer=TrajectoryScorer().success(
                llm_judge(
                    "The agent should ask where customer questions come from (email, Slack, support tool, etc.).",
                    "The agent should ask about the domain or product to understand what kinds of questions to expect.",
                    "The agent should NOT ask whether responses should be automated vs. drafted unless the distinction is unclear from context.",
                )
            ),
        ),
    ]


def _build_model(model_name: str) -> BaseChatModel:
    """Initialize the benchmark model with deterministic settings."""
    return init_chat_model(model_name, temperature=0, timeout=90)


def _raise_invalid_invoke_result(raw_result: object) -> None:
    """Raise a descriptive type error for malformed invoke results."""
    msg = f"Expected invoke result to be Mapping, got {type(raw_result)!r}"
    raise TypeError(msg)


def _invoke_agent(
    *,
    agent: CompiledStateGraph[object, object, object, object],
    query: str | list[AnyMessage],
    initial_files: Mapping[str, str] | None = None,
    recursion_limit: int = 12,
) -> object:
    """Invoke a Deep Agent graph and return the raw result."""
    if isinstance(query, str):
        invoke_inputs: dict[str, Any] = {"messages": [{"role": "user", "content": query}]}
    else:
        invoke_inputs = {"messages": query}
    if initial_files:
        invoke_inputs["files"] = {
            path: create_file_data(content) for path, content in initial_files.items()
        }
    return agent.invoke(
        invoke_inputs,
        {
            "configurable": {"thread_id": str(uuid.uuid4())},
            "recursion_limit": recursion_limit,
        },
    )


def evaluate_case(
    case: BetterHarnessCase,
    *,
    model: BaseChatModel,
    variant: HarnessVariant,
    module_prompts: Mapping[str, str],
    base_prompt: str = EXAMPLE_BASE_AGENT_PROMPT,
) -> CaseResult:
    """Run a single benchmark case for a harness variant."""
    effective_base_prompt = render_variant_base_prompt(
        base_prompt=base_prompt,
        variant=variant,
        module_prompts=dict(module_prompts),
    )
    with patched_base_agent_prompt(effective_base_prompt):
        agent = create_deep_agent(
            model=model,
            tools=list(case.tools),
            system_prompt=case.system_prompt,
        )

    started_at = time.perf_counter()
    failure: str | None = None
    final_text = ""
    trajectory_text = ""
    step_count = 0
    tool_call_count = 0
    passed = False

    try:
        raw_result = _invoke_agent(
            agent=agent,
            query=case.query,
            initial_files=case.initial_files or None,
            recursion_limit=case.recursion_limit,
        )
        if not isinstance(raw_result, Mapping):
            _raise_invalid_invoke_result(raw_result)
        trajectory = trajectory_from_result(raw_result)
        final_text = trajectory.answer
        trajectory_text = trajectory.pretty()
        step_count = len(trajectory.steps)
        tool_call_count = sum(len(step.action.tool_calls) for step in trajectory.steps)

        failures = [
            assertion.describe_failure(trajectory)
            for assertion in case.scorer._success  # noqa: SLF001 - scorer intentionally exposes assertions through the builder contract
            if not assertion.check(trajectory)
        ]

        failure = "\n".join(failures) if failures else None
        passed = not failures
    except Exception as exc:  # noqa: BLE001
        failure = f"{type(exc).__name__}: {exc}"

    duration_s = time.perf_counter() - started_at
    return CaseResult(
        case_id=case.case_id,
        category=case.category,
        split=case.split,
        passed=passed,
        failure=failure,
        duration_s=round(duration_s, 4),
        step_count=step_count,
        tool_call_count=tool_call_count,
        final_text=final_text,
        trajectory=trajectory_text,
    )


def evaluate_variant(
    variant: HarnessVariant,
    *,
    cases: Sequence[BetterHarnessCase],
    model_name: str,
    module_prompts: Mapping[str, str],
    base_prompt: str = EXAMPLE_BASE_AGENT_PROMPT,
) -> SuiteResult:
    """Run a harness variant against a set of benchmark cases."""
    if not cases:
        msg = "cases must not be empty"
        raise ValueError(msg)

    model = _build_model(model_name)
    results: list[CaseResult] = []
    for index, case in enumerate(cases, 1):
        logger.info(
            "Evaluating variant=%s split=%s case=%s (%d/%d)",
            variant.key,
            case.split,
            case.case_id,
            index,
            len(cases),
        )
        result = evaluate_case(
            case,
            model=model,
            variant=variant,
            module_prompts=module_prompts,
            base_prompt=base_prompt,
        )
        logger.info(
            "Finished variant=%s case=%s passed=%s duration_s=%.2f",
            variant.key,
            case.case_id,
            result.passed,
            result.duration_s,
        )
        results.append(result)
    return SuiteResult(variant=variant, split=cases[0].split, cases=results)
