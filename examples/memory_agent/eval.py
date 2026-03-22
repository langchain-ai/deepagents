"""Evaluation framework: does the agent get better over time?

Simulates N days of agent usage with memory consolidation between days,
then measures whether responses improve. This is the "day N eval" —
the interesting question isn't how good the agent is on day 1, but
whether it's measurably better by day N.

Usage:
    python eval.py
    python eval.py --days 5 --tasks-per-day 3 --model anthropic:claude-sonnet-4-6
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.store.memory import InMemoryStore

from memory_agent.prompts import (
    AGENT_INSTRUCTIONS,
    GLOBAL_MEMORY_PROMPT,
    USER_MEMORY_PROMPT,
)

logger = logging.getLogger(__name__)


@dataclass
class EvalTask:
    """A single evaluation task."""

    query: str
    rubric: str
    category: str


@dataclass
class EvalResult:
    """Result of evaluating a single response."""

    task: EvalTask
    day: int
    response: str
    score: float
    reasoning: str


@dataclass
class DayResult:
    """Aggregated results for one simulated day."""

    day: int
    results: list[EvalResult] = field(default_factory=list)

    @property
    def avg_score(self) -> float:
        """Average score across all tasks for this day."""
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)


EVAL_TASKS: list[EvalTask] = [
    EvalTask(
        query="Write me a summary of the key points from our last discussion about the migration plan.",
        rubric="Does the agent acknowledge it has no prior context (day 1) or reference learned context about the user's project (later days)? Higher scores for leveraging memory.",
        category="personalization",
    ),
    EvalTask(
        query="Help me draft an email to the team about the Q3 roadmap.",
        rubric="Does the response match the user's preferred communication style? On day 1 this should be generic; on later days it should reflect learned tone/format preferences.",
        category="style_adaptation",
    ),
    EvalTask(
        query="What's the best approach for handling error retries in our service?",
        rubric="Does the agent incorporate domain-specific knowledge learned from prior conversations? Higher scores for referencing patterns/tools the user's team actually uses.",
        category="domain_knowledge",
    ),
    EvalTask(
        query="Review this function and suggest improvements: def process(data): return [x for x in data if x > 0]",
        rubric="Does the agent apply learned coding preferences (naming conventions, style, level of detail)? Later days should show more tailored feedback.",
        category="code_review",
    ),
    EvalTask(
        query="Create a brief status update for my manager.",
        rubric="Does the agent know the user's role, project context, and preferred update format? Higher scores for incorporating learned context vs generic template.",
        category="personalization",
    ),
]

SIMULATED_CONVERSATIONS: list[list[dict[str, str]]] = [
    [
        {
            "role": "user",
            "content": "I prefer concise, bullet-point responses. No fluff.",
        },
        {
            "role": "assistant",
            "content": "Got it — I'll keep responses concise with bullet points.",
        },
        {
            "role": "user",
            "content": "We're migrating from PostgreSQL to CockroachDB. The main challenge is our custom partitioning logic.",
        },
        {
            "role": "assistant",
            "content": "Understood. Key considerations for Postgres → CockroachDB migration with custom partitioning:\n- CockroachDB uses hash/range sharding, not Postgres-style declarative partitioning\n- You'll need to redesign partition logic around CockroachDB's automatic range splitting\n- Test query performance on the new topology before cutting over",
        },
    ],
    [
        {
            "role": "user",
            "content": "For code reviews, I care most about error handling and naming. I use Python with type hints everywhere.",
        },
        {
            "role": "assistant",
            "content": "Noted — I'll focus on error handling, naming quality, and type hint completeness in reviews.",
        },
        {
            "role": "user",
            "content": "Our team uses exponential backoff with jitter for retries. We have a shared retry util in libs/retry.py.",
        },
        {
            "role": "assistant",
            "content": "Good pattern. I'll reference your `libs/retry.py` utility when retry logic comes up.",
        },
    ],
    [
        {
            "role": "user",
            "content": "I'm a senior engineer leading the platform team. My manager is VP of Engineering, Sarah Chen.",
        },
        {
            "role": "assistant",
            "content": "Got it — platform team lead, reporting to Sarah Chen (VP Eng).",
        },
        {
            "role": "user",
            "content": "Status updates to Sarah should be 3-4 bullets max, lead with blockers, then progress, then next steps.",
        },
        {
            "role": "assistant",
            "content": "Will do — blockers → progress → next steps, 3-4 bullets.",
        },
    ],
]

JUDGE_PROMPT = """\
You are an evaluation judge. Score the agent's response on a scale of 1-5.

## Scoring rubric
{rubric}

## Scoring scale
1 = Poor: Generic response with no personalization or context awareness
2 = Below average: Minimal awareness of context
3 = Average: Some relevant context used but could be better
4 = Good: Clear use of learned context, mostly tailored
5 = Excellent: Highly personalized, leverages all available context effectively

## The user's query
{query}

## The agent's response
{response}

## Current day
Day {day} of the evaluation (day 1 = no prior memory, higher days = more memory available)

Respond with JSON: {{"score": <1-5>, "reasoning": "<brief explanation>"}}
"""


def _build_system_prompt(store: InMemoryStore, user_id: str) -> str:
    """Build system prompt with current memory state."""
    parts = [AGENT_INSTRUCTIONS]

    global_item = store.get(("memories", "global"), "/context.md")
    if global_item and global_item.value.get("content"):
        parts.append(
            GLOBAL_MEMORY_PROMPT.format(global_memory=global_item.value["content"])
        )

    user_item = store.get(("memories", "users", user_id), "/context.md")
    if user_item and user_item.value.get("content"):
        parts.append(USER_MEMORY_PROMPT.format(user_memory=user_item.value["content"]))

    return "\n\n".join(parts)


def _get_agent_response(
    model: Any,
    system_prompt: str,
    query: str,
) -> str:
    """Get a response from the agent model."""
    response = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]
    )
    return (
        response.content if isinstance(response.content, str) else str(response.content)
    )


def _judge_response(
    judge_model: Any,
    task: EvalTask,
    response: str,
    day: int,
) -> tuple[float, str]:
    """Have a judge model score the response."""
    prompt = JUDGE_PROMPT.format(
        rubric=task.rubric,
        query=task.query,
        response=response,
        day=day,
    )
    judge_response = judge_model.invoke([HumanMessage(content=prompt)])
    content = (
        judge_response.content
        if isinstance(judge_response.content, str)
        else str(judge_response.content)
    )

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            result = json.loads(content[start:end])
        else:
            return 3.0, "Failed to parse judge response"

    return float(result.get("score", 3)), result.get("reasoning", "")


def _simulate_day_conversations(
    store: InMemoryStore,
    user_id: str,
    day: int,
    conversations: list[list[dict[str, str]]],
) -> None:
    """Store simulated conversations so the cron job can consolidate them."""
    for i, conv in enumerate(conversations):
        thread_key = f"day{day}-conv{i}-{uuid.uuid4().hex[:8]}"
        store.put(
            ("threads", user_id),
            thread_key,
            {"messages": conv, "day": day},
        )


def _run_consolidation(
    store: InMemoryStore,
    user_id: str,
    model: Any,
) -> None:
    """Run memory consolidation (inline version of cron.py logic)."""
    from cron import consolidate_user

    consolidate_user(store=store, user_id=user_id, model=str(model.model))


def run_eval(
    *,
    days: int = 3,
    tasks_per_day: int | None = None,
    model_name: str = "anthropic:claude-sonnet-4-6",
    judge_model_name: str = "anthropic:claude-sonnet-4-6",
) -> list[DayResult]:
    """Run the day-N evaluation.

    Simulates multiple days of agent usage with memory consolidation
    between days, measuring whether responses improve.

    Args:
        days: Number of simulated days to run.
        tasks_per_day: Number of eval tasks per day. Defaults to all tasks.
        model_name: Model for the agent under evaluation.
        judge_model_name: Model for the judge.

    Returns:
        List of DayResult objects, one per simulated day.
    """
    store = InMemoryStore()
    user_id = "eval-user"
    model = init_chat_model(model_name)
    judge_model = init_chat_model(judge_model_name)

    tasks = EVAL_TASKS[:tasks_per_day] if tasks_per_day else EVAL_TASKS
    day_results: list[DayResult] = []

    for day in range(1, days + 1):
        logger.info("=== Day %d ===", day)

        if day > 1:
            conv_idx = (day - 2) % len(SIMULATED_CONVERSATIONS)
            _simulate_day_conversations(
                store, user_id, day - 1, [SIMULATED_CONVERSATIONS[conv_idx]]
            )
            _run_consolidation(store, user_id, model)

        system_prompt = _build_system_prompt(store, user_id)
        day_result = DayResult(day=day)

        for task in tasks:
            response = _get_agent_response(model, system_prompt, task.query)
            score, reasoning = _judge_response(judge_model, task, response, day)

            result = EvalResult(
                task=task,
                day=day,
                response=response,
                score=score,
                reasoning=reasoning,
            )
            day_result.results.append(result)
            logger.info("  [%s] score=%.1f: %s", task.category, score, reasoning[:80])

        day_results.append(day_result)
        logger.info("Day %d avg score: %.2f", day, day_result.avg_score)

    return day_results


def print_report(day_results: list[DayResult]) -> None:  # noqa: T201
    """Print a summary report of the evaluation."""
    print("\n" + "=" * 60)  # noqa: T201
    print("MEMORY AGENT EVALUATION REPORT")  # noqa: T201
    print("Does the agent improve over time?")  # noqa: T201
    print("=" * 60)  # noqa: T201

    for dr in day_results:
        print(f"\nDay {dr.day}: avg score = {dr.avg_score:.2f}")  # noqa: T201
        for r in dr.results:
            print(f"  [{r.task.category}] {r.score:.0f}/5 — {r.reasoning[:60]}")  # noqa: T201

    if len(day_results) >= 2:
        first = day_results[0].avg_score
        last = day_results[-1].avg_score
        delta = last - first
        print(f"\nImprovement: {first:.2f} → {last:.2f} (Δ = {delta:+.2f})")  # noqa: T201
        if delta > 0:
            print("✅ Agent improved over time!")  # noqa: T201
        elif delta == 0:
            print("➡️  No change detected.")  # noqa: T201
        else:
            print("❌ Agent got worse — investigate memory consolidation quality.")  # noqa: T201

    print("=" * 60)  # noqa: T201


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Run memory agent day-N evaluation")
    parser.add_argument("--days", type=int, default=3, help="Number of simulated days")
    parser.add_argument(
        "--tasks-per-day", type=int, default=None, help="Tasks per day (default: all)"
    )
    parser.add_argument(
        "--model", type=str, default="anthropic:claude-sonnet-4-6", help="Agent model"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="anthropic:claude-sonnet-4-6",
        help="Judge model",
    )
    args = parser.parse_args()

    results = run_eval(
        days=args.days,
        tasks_per_day=args.tasks_per_day,
        model_name=args.model,
        judge_model_name=args.judge_model,
    )
    print_report(results)
