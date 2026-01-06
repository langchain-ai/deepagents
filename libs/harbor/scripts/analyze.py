#!/usr/bin/env python3
"""Analyze job trials from a jobs directory.

Scans through trial directories, extracts trajectory data and success metrics.
"""

import argparse
import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from deepagents import create_deep_agent


def scan_dataset_for_solutions(dataset_path: Path) -> dict[str, Path]:
    """Scan a dataset directory and create a mapping from task names to solution paths.

    Args:
        dataset_path: Path to the dataset directory (e.g., terminal-bench/)

    Returns:
        Dictionary mapping task names to their solution/solve.sh paths
        Example: {"chess-best-move": Path("terminal-bench/7bFm.../chess-best-move/solution/solve.sh")}
    """
    task_to_solution: dict[str, Path] = {}

    if not dataset_path.exists():
        print(f"Warning: Dataset path {dataset_path} does not exist")
        return task_to_solution

    # Iterate through hash directories
    for hash_dir in dataset_path.iterdir():
        if not hash_dir.is_dir():
            continue

        # Iterate through task directories within each hash
        for task_dir in hash_dir.iterdir():
            if not task_dir.is_dir():
                continue

            # Check if this is a valid task directory (has solution/solve.sh)
            solution_path = task_dir / "solution" / "solve.sh"
            if solution_path.exists():
                task_name = task_dir.name
                # Store the mapping (if task appears multiple times, last one wins)
                task_to_solution[task_name] = solution_path

    return task_to_solution


def find_task_directory(trial_dir: Path, task_name: str, task_source: str) -> Optional[Path]:
    """Find the task directory for a given trial.

    Args:
        trial_dir: Path to the trial directory
        task_name: Name of the task (from config.json)
        task_source: Source of the task (e.g., "terminal-bench")

    Returns:
        Path to the task directory if found, None otherwise
    """
    # Start from the trial directory and search for the task directory
    # The structure is typically: {task_source}/{hash}/{task_name}

    # Go up to find the task source directory
    current = trial_dir.parent.parent  # Go up from trial to jobs root
    task_source_dir = current / task_source

    if not task_source_dir.exists():
        return None

    # Search for the task in any hash subdirectory
    for hash_dir in task_source_dir.iterdir():
        if hash_dir.is_dir():
            task_dir = hash_dir / task_name
            if task_dir.exists():
                return task_dir

    return None


class TrialStatus(Enum):
    """Status of a trial execution."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Trial:
    """Metadata for a single trial run."""

    trial_id: str
    status: TrialStatus
    reward: Optional[bool] = None
    trajectory_path: Optional[Path] = None
    reward_path: Optional[Path] = None
    exception_path: Optional[Path] = None
    solution_path: Optional[Path] = None
    trial_dir: Optional[Path] = None
    tool_usage: Optional[dict[str, int]] = None


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
                metadata["reward"] = (
                    result.get("verifier_result", {}).get("rewards", {}).get("reward", 0.0)
                )
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


def count_tool_usage(trajectory_path: Path) -> dict[str, int]:
    """Count tool usage across all steps in a trajectory.

    Args:
        trajectory_path: Path to the trajectory.json file in ATIF format

    Returns:
        Dictionary mapping tool names to their usage counts
    """
    tool_counts: dict[str, int] = {}

    try:
        with open(trajectory_path, "r") as f:
            trajectory_data = json.load(f)

        # Iterate through all steps
        for step in trajectory_data.get("steps", []):
            # Check if this step has tool calls
            tool_calls = step.get("tool_calls")
            if tool_calls:
                # Count each tool call
                for tool_call in tool_calls:
                    tool_name = tool_call.get("function_name", "unknown")
                    tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1

        return tool_counts
    except Exception:
        return {}


def get_task_name_from_trial(trial_dir: Path) -> Optional[str]:
    """Extract the task name from a trial's config.json.

    Args:
        trial_dir: Path to the trial directory

    Returns:
        Task name if found, None otherwise
    """
    config_path = trial_dir / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                return config.get("task", {}).get("path", "")
        except Exception:
            pass
    return None


def enrich_trials_with_solutions(
    trials: list[Trial], solution_mapping: dict[str, Path]
) -> list[Trial]:
    """Update trials with solution paths from a pre-computed solution mapping.

    Args:
        trials: List of Trial objects to enrich
        solution_mapping: Dictionary mapping task names to solution paths

    Returns:
        The same list of trials (modified in place) for convenience
    """
    for trial in trials:
        if trial.trial_dir:
            task_name = get_task_name_from_trial(trial.trial_dir)
            if task_name and task_name in solution_mapping:
                trial.solution_path = solution_mapping[task_name]
    return trials


async def analyze_trial(
    trial_dir: Path, solution_mapping: Optional[dict[str, Path]] = None
) -> Optional[Trial]:
    """Analyze a single trial directory.

    Returns a Trial object even if trajectory or reward files are missing so incomplete
    trials can be reported.

    Status is determined as follows:
    - FAILED: If exception.txt exists or reward is False
    - COMPLETED: If reward is True
    - PENDING: Otherwise (no reward, no exception)
    """
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    reward_path = trial_dir / "verifier" / "reward.txt"
    exception_path = trial_dir / "exception.txt"

    # Read config to find the task directory for the solution
    config_path = trial_dir / "config.json"
    solution_path = None

    # First try to use the solution_mapping if provided
    if solution_mapping:
        task_name = get_task_name_from_trial(trial_dir)
        if task_name and task_name in solution_mapping:
            solution_path = solution_mapping[task_name]

    # Fall back to searching for the task directory
    if not solution_path and config_path.exists():
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                task_name = config.get("task", {}).get("path", "")
                task_source = config.get("task", {}).get("source", "")
                if task_name and task_source:
                    task_dir = find_task_directory(trial_dir, task_name, task_source)
                    if task_dir:
                        solution_path = task_dir / "solution" / "solve.sh"
        except Exception:
            pass

    traj_exists = trajectory_path.exists()
    reward_exists = reward_path.exists()
    exception_exists = exception_path.exists()
    solution_exists = solution_path and solution_path.exists()

    reward_value: Optional[bool]
    if reward_exists:
        reward_value = reward_path.read_text().strip() == "1"
    else:
        reward_value = None

    # Determine status
    if exception_exists:
        status = TrialStatus.FAILED
    elif reward_value is True:
        status = TrialStatus.COMPLETED
    elif reward_value is False:
        status = TrialStatus.FAILED
    else:
        status = TrialStatus.PENDING

    # Count tool usage if trajectory exists
    tool_usage = None
    if traj_exists:
        tool_usage = count_tool_usage(trajectory_path)

    trial_id = trial_dir.name
    return Trial(
        trial_id=trial_id,
        status=status,
        reward=reward_value,
        trajectory_path=trajectory_path if traj_exists else None,
        reward_path=reward_path if reward_exists else None,
        exception_path=exception_path if exception_exists else None,
        solution_path=solution_path if solution_exists else None,
        trial_dir=trial_dir,
        tool_usage=tool_usage,
    )


async def scan_jobs_directory(
    jobs_dir: Path, solution_mapping: Optional[dict[str, Path]] = None
) -> list[Trial]:
    """Scan the jobs directory and extract all trial metadata.

    Args:
        jobs_dir: Path to the jobs directory containing trial subdirectories
        solution_mapping: Optional pre-computed mapping from task names to solution paths.
                         If not provided, solutions will be searched for individually.
    """
    if not jobs_dir.exists():
        print(f"Error: Directory {jobs_dir} does not exist")
        return []

    # List all directories within jobs_dir - each directory is a trial
    trial_dirs: list[Path] = [d for d in jobs_dir.iterdir() if d.is_dir()]

    print(f"Found {len(trial_dirs)} trial directories")

    trials: list[Trial] = []
    for trial_dir in trial_dirs:
        trial = await analyze_trial(trial_dir, solution_mapping=solution_mapping)
        trials.append(trial)
    return trials


def print_summary(trials: list[Trial]) -> None:
    """Print a summary of the analyzed trials."""
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"Total trials: {len(trials)}")

    completed = sum(1 for t in trials if t.status == TrialStatus.COMPLETED)
    failed = sum(1 for t in trials if t.status == TrialStatus.FAILED)
    pending = sum(1 for t in trials if t.status == TrialStatus.PENDING)

    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Pending: {pending}")

    if trials:
        complete_trials = completed + failed
        if complete_trials > 0:
            success_rate = (completed / complete_trials) * 100
            print(f"Success rate (excluding pending): {success_rate:.1f}%")

        # Also show success rate including pending trials
        total_trials = len(trials)
        if total_trials > 0:
            overall_success_rate = (completed / total_trials) * 100
            print(f"Success rate (of all trials): {overall_success_rate:.1f}%")

    # Compute overall tool usage across all trials
    overall_tool_usage: dict[str, int] = {}
    trials_with_tools = 0
    for trial in trials:
        if trial.tool_usage:
            trials_with_tools += 1
            for tool_name, count in trial.tool_usage.items():
                overall_tool_usage[tool_name] = overall_tool_usage.get(tool_name, 0) + count

    if overall_tool_usage:
        print(f"\n{'=' * 80}")
        print("OVERALL TOOL USAGE")
        print(f"{'=' * 80}")
        print(f"Trials with tool usage data: {trials_with_tools}/{len(trials)}")
        print("\nTool usage across all trials:")
        # Sort by usage count (descending) then alphabetically
        sorted_overall_tools = sorted(overall_tool_usage.items(), key=lambda x: (-x[1], x[0]))
        for tool_name, count in sorted_overall_tools:
            print(f"  {tool_name}: {count}")

    print("\n" + "=" * 80)
    print("TRIAL DETAILS")
    print("=" * 80)

    # Sort trials: COMPLETED first, then FAILED, then PENDING
    status_order = {TrialStatus.COMPLETED: 0, TrialStatus.FAILED: 1, TrialStatus.PENDING: 2}
    sorted_trials = sorted(trials, key=lambda t: status_order[t.status])

    for trial in sorted_trials:
        if trial.status == TrialStatus.COMPLETED:
            status = "✓ COMPLETED"
        elif trial.status == TrialStatus.FAILED:
            status = "✗ FAILED"
        else:
            status = "⋯ PENDING"

        print(f"\n{status} | {trial.trial_id}")

        if trial.trajectory_path:
            print(f"  Trajectory: {trial.trajectory_path}")
        else:
            print("  Trajectory: MISSING")

        if trial.reward_path:
            print(f"  Reward file: {trial.reward_path}")
        else:
            print("  Reward file: MISSING")

        if trial.exception_path and trial.exception_path.exists():
            try:
                exception_content = trial.exception_path.read_text()
                # Show last 100 characters
                exception_snippet = (
                    exception_content[-100:] if len(exception_content) > 100 else exception_content
                )
                print(f"  Exception: ...{exception_snippet}")
            except Exception:
                print("  Exception: [Error reading exception file]")

        # Display tool usage if available
        if trial.tool_usage:
            # Sort tools by usage count (descending) then alphabetically
            sorted_tools = sorted(trial.tool_usage.items(), key=lambda x: (-x[1], x[0]))
            tool_summary = ", ".join([f"{tool}: {count}" for tool, count in sorted_tools])
            print(f"  Tool usage: {tool_summary}")


ANALYSIS_PROMPT = """\
# Trajectory Analysis Prompt

You are analyzing an agent execution trajectory. Your goal is to identify what happened during execution and, if the trial failed, determine why.

## IMPORTANT: Trial Status

The trial status will be explicitly provided to you. This status is the ground truth:
- **FAILED**: The agent did not successfully complete the task (reward = 0 or exception occurred)
- **PENDING**: The trial has not finished executing yet
- **COMPLETED**: The agent successfully completed the task (reward = 1)

**If the status is FAILED, then something went wrong, even if the agent reported success or the trajectory appears successful.** Your job is to identify what went wrong by carefully examining the details.

## Reference Solution

A reference solution script (solve.sh) will be provided when available. This script shows the correct approach to solving the task. Use this to:
- Compare the agent's approach against the known working solution
- Identify where the agent's actions diverged from the correct approach
- Understand what steps or commands the agent missed or executed incorrectly
- Determine if the agent used different tools/methods that led to failure

## Trajectory Format

The trajectory is in ATIF (Agent Trajectory Interchange Format) with sequential steps:
- `source`: Who generated the step (system/user/agent)
- `message`: The content of the step
- `tool_calls`: (if present) Tools the agent attempted to use
- `observation`: (if present) Results from tool execution

## Analysis Task

Review the trajectory with careful attention to subtle details and provide:

### 1. FAILURE IDENTIFICATION (for FAILED trials)

**Start by comparing the user's request to the agent's actual actions:**
- What exactly did the user ask for? (Quote the specific request)
- What exactly did the agent do? (Quote the actual tool calls and parameters)
- If a reference solution is provided, how does the agent's approach differ from it?
- Are there any discrepancies between what was requested and what was executed?

**Then identify:**
- **Failure Step**: Which step number failed or where did things go wrong?
- **What Failed**: Describe what went wrong (tool error, incorrect logic, incomplete execution, subtle mistakes, etc.)
- **Error Details**: Quote any error messages or failure indicators
- **Subtle Issues**: Look for problems that aren't obvious errors - small differences in parameters, values, or execution that don't match the request

**Special Case: Max Iterations Reached**
If the agent failed due to reaching the maximum iteration/recursion limit:
- **Evaluate Progress**: Was the agent making sensible progress toward the solution?
- **Direction Assessment**: Were the agent's actions moving it closer to completing the task?
- **Correctness**: Despite not finishing, were the steps taken correct and logical?
- **Compare to Solution**: If a reference solution is provided, was the agent following a similar approach?
- **Estimate Completion**: How close was the agent to completing the task when it hit the limit?
- **Root Cause**: Was the limit hit due to:
  - Agent making good progress but task simply required more steps?
  - Agent spinning in circles or repeating ineffective actions?
  - Agent pursuing a suboptimal approach that would take too many steps?
  - Agent getting stuck on a subtask or error recovery loop?

### 2. EXECUTION ANALYSIS
- **What the Agent Did**: Trace the agent's actions step by step
- **What Was Expected**: Based on the user's request and reference solution (if provided), what should have happened?
- **Where It Went Wrong**: Identify the specific point where the agent's actions diverged from what was needed
- **Tool Usage**: Examine all tool parameters carefully - verify they match what the user requested

### 3. ROOT CAUSE
Determine the underlying cause:
- Is this incorrect tool usage (wrong tool or wrong parameters)?
- Is this a logical/reasoning error (agent made wrong decision)?
- Is this a tool execution error (tool failed or returned error)?
- Is this incomplete execution (agent stopped too early)?
- Is this a resource/permission error?
- Is this agent confusion about the task requirements?
- Is this a subtle parameter mismatch (values that look correct but differ from the request)?

### 4. SUGGESTED IMPROVEMENTS
If clear from the trajectory, suggest:
- What the agent should have done differently (reference the solution script if available)
- Which component or capability needs improvement
- How to prevent this type of failure

## Guidelines

- **Pay close attention to details**: Even if the agent reported success, if the trial failed, find what went wrong
- **Use the reference solution**: When provided, compare the agent's approach systematically against it
- Look for subtle issues like path mistakes, incorrect values, or logical errors
- Be concise but specific
- Quote exact error messages when present
- Focus on actionable insights
- Identify patterns in agent behavior that led to failure
- Don't assume the agent is correct just because it reported success
"""  # noqa: E501


async def analyze_failed_trial(trial: Trial, analyze_pending: bool = False) -> Optional[str]:
    """
    Run deep agent analysis on a failed or pending trial trajectory.

    Args:
        trial: The trial to analyze
        analyze_pending: If True, analyze pending trials in addition to failed ones

    Returns:
        Analysis result as a string, or None if trajectory cannot be read
    """
    # Create the deep agent for trajectory analysis
    analysis_agent = create_deep_agent(tools=[], system_prompt=ANALYSIS_PROMPT)

    # Skip completed trials
    if trial.status == TrialStatus.COMPLETED:
        return None

    # Skip pending trials unless explicitly requested
    if trial.status == TrialStatus.PENDING and not analyze_pending:
        return None

    if not trial.trajectory_path or not trial.trajectory_path.exists():
        return None

    # Read the trajectory file
    with open(trial.trajectory_path, "r") as f:
        trajectory_data = json.load(f)

    # Format trajectory as JSON string for the prompt
    trajectory_json = json.dumps(trajectory_data, indent=2)

    # Read the solution script if available
    solution_content = None
    if trial.solution_path and trial.solution_path.exists():
        solution_content = trial.solution_path.read_text()

    # Create the user message with the trajectory and explicit status
    status_desc = "failed" if trial.status == TrialStatus.FAILED else "pending"
    status_upper = trial.status.value.upper()
    user_message = f"**TRIAL STATUS: {status_upper}**\n\n"

    # Add reference solution if available
    if solution_content:
        user_message += (
            f"**REFERENCE SOLUTION (solve.sh):**\n\n```bash\n{solution_content}\n```\n\n"
        )
    else:
        user_message += "**REFERENCE SOLUTION:** Not provided\n\n"

    user_message += (
        f"Please analyze this {status_desc} agent trajectory:\n\n```json\n{trajectory_json}\n```\n"
    )

    # Run the deep agent analysis
    result = analysis_agent.invoke({"messages": [{"role": "user", "content": user_message}]})

    # Extract the analysis from the response
    analysis = result["messages"][-1].content
    return analysis


async def write_trial_analysis(
    trial: Trial,
    trial_dir: Path,
    output_dir: Path,
    summary_only: bool = False,
    analyze_pending: bool = False,
) -> Optional[Path]:
    """
    Analyze a failed or pending trial and write the results to a file.

    Args:
        trial: The trial to analyze
        trial_dir: Path to the trial directory
        output_dir: Directory where analysis files should be written
        summary_only: If True, skip LLM analysis and only write metadata summary
        analyze_pending: If True, analyze pending trials in addition to failed ones

    Returns:
        Path to the written analysis file, or None if analysis was skipped
    """
    # Skip completed trials
    if trial.status == TrialStatus.COMPLETED:
        return None

    # Skip pending trials unless explicitly requested
    if trial.status == TrialStatus.PENDING and not analyze_pending:
        return None

    # Extract metadata
    metadata = extract_task_metadata(trial_dir)

    # Extract task instructions
    task_instructions = None
    if trial.trajectory_path:
        task_instructions = extract_task_instructions(trial.trajectory_path)

    # Run the LLM analysis unless summary_only is True
    analysis = None
    if not summary_only:
        analysis = await analyze_failed_trial(trial, analyze_pending=analyze_pending)
        if not analysis:
            # If we couldn't get analysis (e.g., missing trajectory), skip this trial
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
        f.write(f"- **Status**: {trial.status.value}\n")
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

        # Write the analysis if not summary_only
        if analysis:
            f.write("\n## Failure Analysis\n\n")
            f.write(analysis)
            f.write("\n")
        elif summary_only:
            f.write("\n## Analysis\n\n")
            f.write("*Summary only mode - detailed LLM analysis skipped*\n")

    return output_file


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze job trials from a jobs directory")
    parser.add_argument(
        "jobs_dir", type=Path, help="Path to the jobs directory (e.g., jobs-terminal-bench/)"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        help="Path to the dataset directory (e.g., terminal-bench/) to scan for solution files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for detailed analysis files (one per failed/pending trial)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary, skip detailed LLM analysis of trials",
    )
    parser.add_argument(
        "--analyze-pending",
        action="store_true",
        help="Analyze pending trials in addition to failed trials",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable format",
    )

    args = parser.parse_args()

    # Scan dataset for solutions if provided
    solution_mapping = None
    if args.dataset:
        print(f"Scanning dataset directory: {args.dataset}")
        solution_mapping = scan_dataset_for_solutions(args.dataset)
        print(f"Found {len(solution_mapping)} tasks with solutions\n")

    # Scan and analyze all trials
    trials = await scan_jobs_directory(args.jobs_dir, solution_mapping=solution_mapping)

    # Print human-readable summary
    print_summary(trials)

    # If output directory specified, run analysis on trials
    if args.output_dir:
        # Determine which trials to analyze based on status
        trials_to_analyze = [
            t for t in trials
            if t.status == TrialStatus.FAILED or (args.analyze_pending and t.status == TrialStatus.PENDING)
        ]

        if not trials_to_analyze:
            status_desc = "failed or pending" if args.analyze_pending else "failed"
            print(f"\nNo {status_desc} trials to analyze.")
        else:
            print(f"\n{'=' * 80}")
            analysis_mode = "SUMMARY" if args.summary_only else "DEEP ANALYSIS"
            trial_types = "FAILED/PENDING" if args.analyze_pending else "FAILED"
            print(f"RUNNING {analysis_mode} ON {trial_types} TRIALS")
            print(f"{'=' * 80}")
            print(f"Processing {len(trials_to_analyze)} trials...")
            print(f"Output directory: {args.output_dir}")
            if args.summary_only:
                print("Mode: Summary only (LLM analysis disabled)")
            if args.analyze_pending:
                print("Mode: Including pending trials")
            print()

            # Analyze each trial
            for i, trial in enumerate(trials_to_analyze, 1):
                status_label = trial.status.value.upper()
                print(f"[{i}/{len(trials_to_analyze)}] Analyzing {trial.trial_id} ({status_label})...")

                if trial.trial_dir is None:
                    print(f"  Warning: No trial directory found for {trial.trial_id}")
                    continue

                # Run the analysis and write to file
                try:
                    output_file = await write_trial_analysis(
                        trial,
                        trial.trial_dir,
                        args.output_dir,
                        summary_only=args.summary_only,
                        analyze_pending=args.analyze_pending,
                    )
                    if output_file:
                        print(f"  ✓ Analysis written to: {output_file}")
                    else:
                        print(f"  ✗ Skipped (no trajectory or already completed)")
                except Exception as e:
                    print(f"  ✗ Error: {e}")

            print(f"\n{'=' * 80}")
            print(f"Analysis complete. Results saved to: {args.output_dir}")
            print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(main())
