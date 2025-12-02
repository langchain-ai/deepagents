# Building DeepAgent Harnesses for Terminal Bench 2.0 with Harbor

## Overview

This repository demonstrates how to evaluate and improve your DeepAgent harness using [Harbor](https://github.com/HarborAI/harbor) and [LangSmith](https://smith.langchain.com).

### What is Harbor?

Harbor is an evaluation framework that simplifies running agents on challenging benchmarks. It provides:

- **Sandbox environments** (Docker, Modal, Daytona, E2B, etc.)
- **Automatic test execution** and verification
- **Reward scoring** (0.0 - 1.0 based on test pass rate)
- **Trajectory logging** in ATIF format (Agent Trajectory Interchange Format)

### What is Terminal Bench 2.0?

[Terminal Bench 2.0](https://github.com/HarborAI/harbor) is an evaluation benchmark that measures agent capabilities across several domains, testing how well an agent operates using a computer environment, primarily via the terminal. The benchmark includes 90+ tasks across domains like software engineering, biology, security, gaming, and more.

**Example tasks:**
- `path-tracing`: Reverse-engineer C program from rendered image
- `chess-best-move`: Find optimal move using chess engine
- `git-multibranch`: Complex git operations with merge conflicts
- `sqlite-with-gcov`: Build SQLite with code coverage, analyze reports

### The DeepAgent Architecture

The DeepAgent harness ships with design patterns validated as good defaults across agentic tasks:

1. **Detailed System Prompt**: Expansive, instructional prompts with tool guidance and examples
2. **Planning Middleware**: The `write_todos` tool helps the agent structure thinking and track progress
3. **Filesystem**: Provides `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep` for context management
4. **SubAgents**: The `task` tool spawns specialized subagents for isolated work

## Quick Start

```bash
# Install dependencies
uv sync --no-editable

# Configure API keys
cp .env.example .env

# Run via Docker (1 task)
uv run harbor run --agent-import-path deepagents_harbor:DeepAgentsWrapper \
  --dataset terminal-bench@2.0 -n 1 --jobs-dir jobs/terminal-bench --env docker

# Run via Daytona (10 tasks) -- requires DAYTONA_API_KEY
uv run harbor run --agent-import-path deepagents_harbor:DeepAgentsWrapper \
  --dataset terminal-bench@2.0 -n 10 --jobs-dir jobs/terminal-bench --env daytona
```

## The Evaluation & Improvement Loop

The agent development workflow:

```
DeepAgents (harness) → Harbor (evaluate) → LangSmith (analyze) → Improve → Repeat
```

### Using Harbor with LangSmith

LangSmith provides automatic tracing and observability for your agent runs. This section explains how to set up the integration.

### Prerequisites

Set up your LangSmith API credentials:

```bash
export LANGSMITH_API_KEY="your-api-key-here"
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"  # Optional, defaults to this
```

### Step 1: Create a LangSmith Dataset

Create a dataset from Harbor tasks. This downloads tasks from the Harbor registry and creates corresponding examples in LangSmith:

```bash
python scripts/create_langsmith_dataset.py terminal-bench --version 2
```

Options:
- `--version`: Specify the Harbor dataset version (default: "head")
- `--overwrite`: Overwrite cached remote tasks

### Step 2: Create an Experiment

Create an experiment session associated with your dataset:

```bash
python scripts/create_langsmith_dataset.py terminal-bench --create-experiment
```

Options:
- `--experiment-name`: Custom name for the experiment (auto-generated if not provided)

This will output:
- The experiment session ID
- A URL to view the experiment in LangSmith
- Instructions for setting the `LANGSMITH_PROJECT` environment variable

### Step 3: Run Benchmarking with Tracing

Run your Harbor benchmark with tracing enabled. You have two options:

#### Option A: Experiment View (Recommended for Benchmarking)

Use `LANGSMITH_EXPERIMENT` to associate runs with an experiment for side-by-side comparison:

Using make:
```bash
LANGSMITH_EXPERIMENT=<experiment-name> make run-terminal-bench-daytona
```

Or using Harbor directly:
```bash
LANGSMITH_EXPERIMENT=<experiment-name> harbor run terminal-bench --config configs/terminal-bench-daytona.yaml
```

Example:
```bash
# Create experiment and capture the name
python scripts/create_langsmith_dataset.py terminal-bench --create-experiment --experiment-name my-experiment

# Run benchmark with experiment tracing
LANGSMITH_EXPERIMENT=my-experiment make run-terminal-bench-daytona
```

#### Option B: Regular Tracing (Project View)

Use `LANGSMITH_PROJECT` if you just want to log traces without an experiment view:

```bash
LANGSMITH_PROJECT=my-project-name make run-terminal-bench-daytona
```

**Experiment view** groups runs by dataset examples and allows side-by-side comparison of different agent configurations. **Regular tracing** logs all runs to a project without the dataset association, useful for general development and debugging.

## Analyzing Results & Improving Your Harness

### What LangSmith Captures

With tracing enabled, LangSmith automatically captures:

- Every LLM call (prompts, outputs, tokens)
- Every tool invocation (arguments, results, errors)
- Performance metrics (latency, cost)
- **Reward scores** from Harbor's test verification (0.0 - 1.0)

### Reward Feedback Integration

Harbor automatically pushes reward scores to LangSmith, connecting **what happened** (execution trace) with **how well it worked** (test results). This allows you to filter runs by reward score and identify patterns in successful vs. failed runs.

### Common Patterns & Fixes

After running evaluations, analyze failed runs in LangSmith to identify improvement opportunities:

| Pattern | Symptom | Potential Fix |
|---------|---------|---------------|
| **Poor Planning** | Agent jumps into coding without reading requirements | Add upfront planning requirement to prompt |
| **Incorrect Tool Usage** | Uses `bash cat` instead of `read_file` | Improve tool descriptions with examples |
| **No Incremental Testing** | Writes 200 lines, then tests once | Prompt to test after each logical unit |
| **Hallucinated Paths** | Reads files before checking existence | Add "always `ls` before read" rule |
| **Wrong Model** | Model fails on complex reasoning | Use more capable model for hard tasks |

### Agent-Assisted Analysis

Because agent trajectories produce large amounts of data across runs, consider using agents to help analyze patterns:

- Use LangSmith's Insights Agent or your own agent
- Feed in trajectory data across multiple runs
- Task: **"Analyze these failed runs and identify common patterns"**

The agent can automatically:
- Group failures by task category
- Identify common error patterns
- Suggest prompt improvements
- Recommend tool redesigns

**This enables systematic, data-driven improvement instead of guessing.**

## Resources

- [DeepAgents Documentation](https://docs.langchain.com/oss/python/deepagents/overview)
- [Harbor GitHub](https://github.com/HarborAI/harbor)
- [LangSmith](https://smith.langchain.com)
