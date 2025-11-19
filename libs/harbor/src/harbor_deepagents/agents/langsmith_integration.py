"""LangSmith integration for Harbor results feedback."""

import os
from typing import Optional

from langsmith import Client


def send_harbor_feedback(
    run_id: str,
    task_name: str,
    reward: float,
    agent_cost_usd: Optional[float] = None,
    total_steps: Optional[int] = None,
) -> None:
    """Send Harbor verification results to LangSmith as feedback.

    This function pushes Harbor's reward score and metadata to LangSmith
    as feedback on the agent's run, making it visible in the LangSmith UI
    alongside the execution trace.

    Args:
        run_id: LangSmith run ID from the agent execution
        task_name: Name of the Harbor task
        reward: Reward score from Harbor verifier (0.0 to 1.0, where 1.0 = 100% pass)
        agent_cost_usd: Optional cost in USD
        total_steps: Optional total number of steps taken

    Example:
        >>> send_harbor_feedback(
        ...     run_id="abc123",
        ...     task_name="web-scraper-task",
        ...     reward=1.0,
        ...     agent_cost_usd=0.012,
        ...     total_steps=11,
        ... )
    """
    # Check if LangSmith tracing is enabled
    if not os.getenv("LANGCHAIN_TRACING_V2"):
        return

    try:
        client = Client()

        # Main reward score feedback
        client.create_feedback(
            run_id=run_id,
            key="harbor_reward",
            score=reward,
            comment=f"Harbor Task: {task_name} | Score: {reward * 100:.0f}%",
        )

        # Optional cost feedback
        if agent_cost_usd is not None:
            client.create_feedback(
                run_id=run_id,
                key="harbor_cost_usd",
                score=agent_cost_usd,
                comment=f"Agent execution cost: ${agent_cost_usd:.4f}",
            )

        # Optional steps feedback
        if total_steps is not None:
            client.create_feedback(
                run_id=run_id,
                key="harbor_steps",
                score=total_steps,
                comment=f"Total steps: {total_steps}",
            )

    except Exception as e:
        # Don't fail the task if feedback fails
        print(f"Warning: Failed to send LangSmith feedback: {e}")


def get_langsmith_url(run_id: str) -> str:
    """Generate LangSmith URL for a given run ID.

    Args:
        run_id: LangSmith run ID

    Returns:
        Full URL to the run in LangSmith UI
    """
    project_name = os.getenv("LANGCHAIN_PROJECT", "default")
    return f"https://smith.langchain.com/o/default/projects/p/{project_name}/r/{run_id}"
