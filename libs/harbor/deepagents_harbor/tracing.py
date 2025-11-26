"""LangSmith integration for Harbor DeepAgents."""

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from langsmith import Client


def create_example_id_from_instruction(instruction: str) -> str:
    """Create a deterministic UUID from an instruction string.

    Normalizes the instruction by stripping whitespace and creating a
    SHA-256 hash, then converting to a UUID for LangSmith compatibility.

    Args:
        instruction: The task instruction string to hash

    Returns:
        A UUID string generated from the hash of the normalized instruction
    """
    # Normalize the instruction: strip leading/trailing whitespace
    normalized = instruction.strip()

    # Create SHA-256 hash of the normalized instruction
    hash_bytes = hashlib.sha256(normalized.encode("utf-8")).digest()

    # Use first 16 bytes to create a UUID
    example_uuid = uuid.UUID(bytes=hash_bytes[:16])

    return str(example_uuid)


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

    client = Client()

    # Main reward score feedback
    client.create_feedback(
        run_id=run_id,
        key="harbor_reward",
        score=reward,
    )


def get_langsmith_url(run_id: str) -> str:
    """Generate LangSmith URL for a given run ID.

    Args:
        run_id: LangSmith run ID

    Returns:
        Full URL to the run in LangSmith UI
    """
    project_name = os.getenv("LANGCHAIN_PROJECT", "default")
    return f"https://smith.langchain.com/o/default/projects/p/{project_name}/r/{run_id}"


load_dotenv()


def get_trace(
    job_id: str,
    project_name: str,
    is_root: bool = True,
) -> list:
    """Fetch LangSmith runs by job_id metadata.

    Args:
        job_id: Job ID value to filter by (stored in run metadata)
        project_name: LangSmith project name to search in
        is_root: If True, only return root runs (default: True)

    Returns:
        List of run objects matching the job_id filter

    Example:
        >>> runs = get_trace(
        ...     job_id="0cb8ca0c-f762-4723-ad1a-1d76c1d7a261",
        ...     project_name="sample"
        ... )
        >>> for run in runs:
        ...     print(f"Run: {run.name}, Status: {run.status}")
    """
    client = Client()

    # Build filter to match job_id in metadata
    filter_query = f'and(eq(metadata_key, "job_id"), eq(metadata_value, "{job_id}"))'

    # Fetch runs matching the filter
    runs = list(
        client.list_runs(
            project_name=project_name,
            filter=filter_query,
            is_root=is_root,
        )
    )

    return runs


def main():
    """CLI entry point for fetching LangSmith traces by job_id."""
    parser = argparse.ArgumentParser(description="Fetch LangSmith runs filtered by job_id metadata")
    parser.add_argument("--job-id", required=True, help="Job ID to filter by")
    parser.add_argument("--project", required=True, help="LangSmith project name")
    parser.add_argument(
        "--include-children",
        action="store_true",
        help="Include child runs (default: root runs only)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path (default: print to stdout)",
    )

    args = parser.parse_args()

    runs = get_trace(
        job_id=args.job_id,
        project_name=args.project,
        is_root=not args.include_children,
    )

    print(f"Found {len(runs)} run(s) matching job_id={args.job_id}")

    output_lines = []
    for run in runs:
        line = f"\nRun ID: {run.id}"
        line += f"\nName: {run.name}"
        line += f"\nStatus: {run.status}"
        line += f"\nRun Type: {run.run_type}"
        if run.start_time:
            line += f"\nStart Time: {run.start_time}"
        if run.end_time:
            line += f"\nEnd Time: {run.end_time}"
        if hasattr(run, "total_tokens") and run.total_tokens:
            line += f"\nTotal Tokens: {run.total_tokens}"
        line += f"\n{'-' * 80}"
        output_lines.append(line)

    output_text = "\n".join(output_lines)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"Results saved to {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()


class LangSmithTrajectoryExporter:
    """Export ATIF trajectories to LangSmith.

    This allows you to:
    - Visualize agent execution in LangSmith UI
    - Compare different runs
    - Analyze costs and latency
    - Debug agent behavior
    """

    def __init__(self, api_key: str | None = None):
        """Initialize exporter with optional API key.

        Args:
            api_key: LangSmith API key (or use LANGCHAIN_API_KEY env var)
        """
        self.client = Client(api_key=api_key)

    def export_trajectory(
        self,
        trajectory_path: Path | str,
        project_name: str = "harbor-deepagents",
    ) -> str:
        """Export an ATIF trajectory file to LangSmith.

        Args:
            trajectory_path: Path to ATIF trajectory JSON file
            project_name: LangSmith project name

        Returns:
            The run ID in LangSmith
        """
        trajectory_path = Path(trajectory_path)

        with open(trajectory_path) as f:
            atif = json.load(f)

        # Create root run
        run_id = self.client.create_run(
            name=f"harbor-{atif['agent']['name']}-{atif['session_id'][:8]}",
            run_type="chain",
            inputs={"instruction": self._extract_user_message(atif)},
            project_name=project_name,
            tags=["harbor", atif["agent"]["name"], "deepagents"],
            extra={
                "metadata": {
                    "agent_name": atif["agent"]["name"],
                    "agent_version": atif["agent"]["version"],
                    "model": atif["agent"].get("model_name"),
                    "session_id": atif["session_id"],
                    "schema_version": atif["schema_version"],
                    **atif["agent"].get("extra", {}),
                }
            },
        ).id

        # Add steps as child runs
        for step in atif["steps"]:
            self._add_step_run(
                step=step,
                parent_run_id=run_id,
                project_name=project_name,
            )

        # Update root run with final output and metrics
        final_message = self._extract_final_message(atif)
        final_metrics = atif.get("final_metrics", {})

        self.client.update_run(
            run_id=run_id,
            outputs={"result": final_message},
            extra={"metadata": final_metrics},
        )

        return run_id

    def export_job_trajectories(
        self,
        job_dir: Path | str,
        project_name: str = "harbor-deepagents",
    ) -> list[str]:
        """Export all trajectories from a Harbor job directory.

        Args:
            job_dir: Path to Harbor job directory
            project_name: LangSmith project name

        Returns:
            List of LangSmith run IDs
        """
        job_dir = Path(job_dir)
        trajectory_files = list(job_dir.rglob("trajectory.json"))

        run_ids = []
        for trajectory_path in trajectory_files:
            try:
                run_id = self.export_trajectory(trajectory_path, project_name)
                run_ids.append(run_id)
                print(f"✓ Exported {trajectory_path.parent.name}: {run_id}")
            except Exception as e:
                print(f"✗ Failed to export {trajectory_path}: {e}")

        return run_ids

    def _extract_user_message(self, atif: dict[str, Any]) -> str:
        """Extract the initial user message from ATIF."""
        for step in atif["steps"]:
            if step["source"] == "user":
                return step["message"]
        return ""

    def _extract_final_message(self, atif: dict[str, Any]) -> str:
        """Extract the final agent message from ATIF."""
        for step in reversed(atif["steps"]):
            if step["source"] == "agent":
                return step["message"]
        return ""

    def _add_step_run(
        self,
        step: dict[str, Any],
        parent_run_id: str,
        project_name: str,
    ) -> None:
        """Add a step as a child run in LangSmith."""
        run_type = self._get_run_type(step)

        inputs = {"message": step.get("message", "")}
        outputs = {}

        # Add tool call information
        if step.get("tool_calls"):
            tool_call = step["tool_calls"][0]  # Simplified: take first tool call
            inputs["tool"] = tool_call.get("function_name")
            inputs["arguments"] = tool_call.get("arguments")

        # Add observation information
        if step.get("observation"):
            results = step["observation"].get("results", [])
            if results:
                outputs["observation"] = results[0].get("content")

        # Create child run
        self.client.create_run(
            name=f"step-{step['step_id']}-{step['source']}",
            run_type=run_type,
            inputs=inputs,
            outputs=outputs if outputs else None,
            parent_run_id=parent_run_id,
            project_name=project_name,
            extra={
                "metadata": {
                    "step_id": step["step_id"],
                    "source": step["source"],
                    "timestamp": step.get("timestamp"),
                    **(step.get("metrics", {}) or {}),
                }
            },
        )

    def _get_run_type(self, step: dict[str, Any]) -> str:
        """Determine LangSmith run type from ATIF step."""
        if step.get("tool_calls"):
            return "tool"
        elif step["source"] == "agent":
            return "llm"
        else:
            return "chain"
