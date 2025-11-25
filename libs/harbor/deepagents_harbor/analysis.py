#!/usr/bin/env python3
"""Analyze job trials from a jobs directory.

Scans through trial directories, extracts trajectory data and success metrics.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from deepagents import create_deep_agent


@dataclass
class Trial:
    """Metadata for a single trial run."""

    trial_id: str
    reward: bool
    trajectory_path: Optional[Path] = None
    reward_path: Optional[Path] = None
    trial_dir: Optional[Path] = None


async def parse_reward(reward_path: Path) -> bool:
    """Parse the reward file. Returns True if reward is 1, False otherwise."""
    content = reward_path.read_text()
    reward_value = content.strip()
    return reward_value == "1"


def extract_task_metadata(trial_dir: Path) -> dict:
    """Extract task metadata from config.json and other files.

    Args:
        trial_dir: Path to the trial directory

    Returns:
        Dictionary containing task metadata
    """
    metadata = {}

    # Read config.json
    config_path = trial_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                metadata["task_name"] = config.get("task", {}).get("path", "")
                metadata["task_source"] = config.get("task", {}).get("source", "")
                metadata["git_url"] = config.get("task", {}).get("git_url", "")
                metadata["git_commit_id"] = config.get("task", {}).get("git_commit_id", "")
        except Exception:
            pass

    # Read result.json for additional metadata
    result_path = trial_dir / "result.json"
    if result_path.exists():
        try:
            with open(result_path, "r") as f:
                result = json.load(f)
                metadata["reward"] = result.get("verifier_result", {}).get("rewards", {}).get("reward", 0.0)
                metadata["started_at"] = result.get("started_at", "")
                metadata["finished_at"] = result.get("finished_at", "")
        except Exception:
            pass

    return metadata


def extract_task_instructions(trajectory_path: Path) -> Optional[str]:
    """Extract the task instructions from the trajectory file.

    Looks for the user message in the trajectory steps.
    """
    try:
        with open(trajectory_path, "r") as f:
            trajectory_data = json.load(f)

        # Find the user message in the steps
        for step in trajectory_data.get("steps", []):
            if step.get("source") == "user":
                return step.get("message", "")

        return None
    except Exception:
        return None


async def analyze_trial(trial_dir: Path) -> Optional[Trial]:
    """Analyze a single trial directory.

    Returns a Trial object even if trajectory or reward files are missing so incomplete
    trials can be reported.
    """
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    reward_path = trial_dir / "verifier" / "reward.txt"

    traj_exists = trajectory_path.exists()
    reward_exists = reward_path.exists()

    reward_value: Optional[bool]
    if reward_exists:
        reward_value = reward_path.read_text().strip() == "1"
    else:
        reward_value = None

    trial_id = trial_dir.name
    return Trial(
        trial_id=trial_id,
        reward=reward_value,
        trajectory_path=trajectory_path if traj_exists else None,
        reward_path=reward_path if reward_exists else None,
        trial_dir=trial_dir,
    )


async def scan_jobs_directory(jobs_dir: Path) -> list[Trial]:
    """Scan the jobs directory and extract all trial metadata."""
    if not jobs_dir.exists():
        print(f"Error: Directory {jobs_dir} does not exist")
        return []

    # List all directories within jobs_dir - each directory is a trial
    trial_dirs: list[Path] = [d for d in jobs_dir.iterdir() if d.is_dir()]

    print(f"Found {len(trial_dirs)} trial directories")

    trials: list[Trial] = []
    for trial_dir in trial_dirs:
        trial = await analyze_trial(trial_dir)
        trials.append(trial)
    return trials


def print_summary(trials: list[Trial]) -> None:
    """Print a summary of the analyzed trials."""
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total trials: {len(trials)}")

    successful = sum(1 for t in trials if t.reward is True)
    failed = sum(1 for t in trials if t.reward is False)
    incomplete = sum(1 for t in trials if t.reward is None)

    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Incomplete: {incomplete}")

    if trials:
        complete_trials = successful + failed
        if complete_trials > 0:
            success_rate = (successful / complete_trials) * 100
            print(f"Success rate (of completed trials): {success_rate:.1f}%")

    print("\n" + "=" * 80)
    print("TRIAL DETAILS")
    print("=" * 80)

    for trial in trials:
        if trial.reward is True:
            status = "✓ SUCCESS"
        elif trial.reward is False:
            status = "✗ FAILED"
        else:
            status = "? INCOMPLETE"

        print(f"\n{status} | {trial.trial_id}")

        if trial.trajectory_path:
            print(f"  Trajectory: {trial.trajectory_path}")
        else:
            print("  Trajectory: MISSING")

        if trial.reward_path:
            print(f"  Reward file: {trial.reward_path}")
        else:
            print("  Reward file: MISSING")


ANALYSIS_PROMPT = """\
# Trajectory Failure Analysis Prompt

You are analyzing a failed agent execution trajectory. Your goal is to identify where and why the agent failed, and suggest potential improvements.

## Trajectory Format

The trajectory is in ATIF (Agent Trajectory Interchange Format) with sequential steps:
- `source`: Who generated the step (system/user/agent)
- `message`: The content of the step
- `tool_calls`: (if present) Tools the agent attempted to use
- `observation`: (if present) Results from tool execution

## Analysis Task

Review the trajectory and provide:

### 1. FAILURE IDENTIFICATION
- **Failure Step**: Which step number failed?
- **What Failed**: Describe what went wrong (tool error, incorrect logic, incomplete execution, etc.)
- **Error Details**: Quote any error messages or failure indicators

### 2. EXECUTION ANALYSIS
- **What the Agent Did**: Trace the agent's actions step by step
- **Where It Went Wrong**: Identify the specific point where the agent diverged from the correct approach
- **Tool Usage Issues**: Were the right tools used with correct parameters?

### 3. ROOT CAUSE
Determine the underlying cause:
- Is this a tool execution error (tool failed or returned error)?
- Is this incorrect tool usage (wrong tool or wrong parameters)?
- Is this a logical/reasoning error (agent made wrong decision)?
- Is this incomplete execution (agent stopped too early)?
- Is this a resource/permission error?
- Is this agent confusion about the task requirements?

### 4. SUGGESTED IMPROVEMENTS
If clear from the trajectory, suggest:
- What the agent should have done differently
- Which component or capability needs improvement
- How to prevent this type of failure

## Guidelines

- Be concise but specific
- Quote exact error messages when present
- Focus on actionable insights
- Identify patterns in agent behavior that led to failure
"""  # noqa: E501




async def analyze_failed_trial(trial: Trial) -> Optional[str]:
    """
    Run deep agent analysis on a failed trial trajectory.

    Args:
        trial: The trial to analyze (should have reward=False)

    Returns:
        Analysis result as a string, or None if trajectory cannot be read
    """
    # Create the deep agent for trajectory analysis
    analysis_agent = create_deep_agent(tools=[], system_prompt=ANALYSIS_PROMPT)

    if trial.reward:
        # Skip successful trials
        return None

    if not trial.trajectory_path or not trial.trajectory_path.exists():
        return None

    # Read the trajectory file
    with open(trial.trajectory_path, "r") as f:
        trajectory_data = json.load(f)

    # Format trajectory as JSON string for the prompt
    trajectory_json = json.dumps(trajectory_data, indent=2)

    # Create the user message with the trajectory
    user_message = (
        f"Please analyze this failed agent trajectory:\n\n```json\n{trajectory_json}\n```\n"
    )

    # Run the deep agent analysis
    result = analysis_agent.invoke({"messages": [{"role": "user", "content": user_message}]})

    # Extract the analysis from the response
    analysis = result["messages"][-1].content
    return analysis


async def write_trial_analysis(
    trial: Trial, trial_dir: Path, output_dir: Path
) -> Optional[Path]:
    """
    Analyze a failed trial and write the results to a file.

    Args:
        trial: The trial to analyze
        trial_dir: Path to the trial directory
        output_dir: Directory where analysis files should be written

    Returns:
        Path to the written analysis file, or None if analysis failed
    """
    if trial.reward:
        # Skip successful trials
        return None

    # Extract metadata
    metadata = extract_task_metadata(trial_dir)

    # Extract task instructions
    task_instructions = None
    if trial.trajectory_path:
        task_instructions = extract_task_instructions(trial.trajectory_path)

    # Run the analysis
    analysis = await analyze_failed_trial(trial)
    if not analysis:
        return None

    # Create output file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{trial.trial_id}.md"

    # Write the analysis with metadata
    with open(output_file, "w") as f:
        f.write(f"# Analysis: {trial.trial_id}\n\n")

        # Write metadata section
        f.write("## Task Metadata\n\n")
        f.write(f"- **Trial ID**: {trial.trial_id}\n")
        f.write(f"- **Task Name**: {metadata.get('task_name', 'N/A')}\n")
        f.write(f"- **Task Source**: {metadata.get('task_source', 'N/A')}\n")
        f.write(f"- **Reward**: {metadata.get('reward', 0.0)}\n")

        if metadata.get("git_url"):
            f.write(f"- **Git URL**: {metadata['git_url']}\n")
        if metadata.get("git_commit_id"):
            f.write(f"- **Git Commit**: {metadata['git_commit_id']}\n")
        if metadata.get("started_at"):
            f.write(f"- **Started**: {metadata['started_at']}\n")
        if metadata.get("finished_at"):
            f.write(f"- **Finished**: {metadata['finished_at']}\n")

        # Write task instructions
        if task_instructions:
            f.write("\n## Task Instructions\n\n")
            f.write("```\n")
            f.write(task_instructions)
            f.write("\n```\n")

        # Write the analysis
        f.write("\n## Failure Analysis\n\n")
        f.write(analysis)
        f.write("\n")

    return output_file
