"""Export Harbor ATIF trajectories to LangSmith for visualization."""

import json
from pathlib import Path
from typing import Any

from langsmith import Client


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
