# Building DeepAgent Harnesses for Terminal Bench 2.0 with Harbor

## Overview

This repository demonstrates how to evaluate and improve your DeepAgent harness using [Harbor](https://github.com/laude-institute/harbor) and [LangSmith](https://smith.langchain.com).

### What is Harbor?

Harbor is an evaluation framework that simplifies running agents on challenging benchmarks. It provides:

- **Sandbox environments** (Docker, Modal, Daytona, E2B, etc.)
- **Automatic test execution** and verification
- **Reward scoring** (0.0 - 1.0 based on test pass rate)
- **Trajectory logging** in ATIF format (Agent Trajectory Interchange Format)

### What is Terminal Bench 2.0?

[Terminal Bench 2.0](https://github.com/laude-institute/terminal-bench-2) is an evaluation benchmark that measures agent capabilities across several domains, testing how well an agent operates using a computer environment, primarily via the terminal. The benchmark includes 90+ tasks across domains like software engineering, biology, security, gaming, and more.

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
uv sync

# Configure API keys - Choose one approach:

# Option 1: Use .env file (recommended for local development)
cp .env.example .env
# Edit .env and add your keys - they'll be automatically loaded

# Option 2: Export directly (useful for CI/CD or quick testing)
export ANTHROPIC_API_KEY="sk-ant-..."  # Required: For Claude model
export LANGSMITH_API_KEY="lsv2_..."    # Required: For tracing
export LANGSMITH_TRACING_V2=true       # Required: Enable LangSmith tracing
export LANGSMITH_ENDPOINT="https://api.smith.langchain.com"  # Optional: Default shown
# export DAYTONA_API_KEY="..."  # Optional: Only if using --env daytona

# Run via Docker (1 task)
uv run harbor run --agent-import-path deepagents_harbor:DeepAgentsWrapper \
  --dataset terminal-bench@2.0 -n 1 --jobs-dir jobs/terminal-bench --env docker

# Run via Daytona (10 tasks)
uv run harbor run --agent-import-path deepagents_harbor:DeepAgentsWrapper \
  --dataset terminal-bench@2.0 -n 10 --jobs-dir jobs/terminal-bench --env daytona
```

## LangSmith Integration

LangSmith provides tracing and observability for agent runs. The workflow:

```
DeepAgents → Harbor (evaluate) → LangSmith (analyze) → Improve → Repeat
```

### Prerequisites

Ensure your LangSmith credentials are configured (see Quick Start for .env or export options):

```bash
# Required environment variables:
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_TRACING_V2=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com  # Optional: defaults to this
```

### Step 1: Create Dataset and Experiment

```bash
# Create dataset from Harbor tasks
python scripts/harbor_langsmith.py create-dataset terminal-bench --version 2.0

# Create experiment session (outputs session ID and URL)
python scripts/harbor_langsmith.py create-experiment terminal-bench --name deepagents-baseline-v1
```

### Step 2: Run Benchmark with Tracing

```bash
# Option 1: For experiments (enables side-by-side comparison in LangSmith)
export LANGSMITH_EXPERIMENT="deepagents-baseline-v1"
make run-terminal-bench-daytona  # Runs 10 tasks on Daytona

# Option 2: For development (simpler project view in LangSmith)
export LANGSMITH_PROJECT="deepagents-development"
make run-terminal-bench-daytona

# Option 3: Run harbor directly (customize -n for number of tasks)
export LANGSMITH_EXPERIMENT="deepagents-baseline-v1"
uv run harbor run \
  --agent-import-path deepagents_harbor:DeepAgentsWrapper \
  --dataset terminal-bench@2.0 -n 10 --jobs-dir jobs/terminal-bench --env daytona
```

### Step 3: Add Feedback Scores

After the benchmark completes, push reward scores to LangSmith for filtering and analysis:

```bash
python scripts/harbor_langsmith.py add-feedback jobs/terminal-bench/2025-12-02__16-25-40 \
  --project-name deepagents-baseline-v1
```

This matches trials to traces and adds `harbor_reward` feedback (0.0-1.0) from Harbor's test results.

## Analyzing Results

LangSmith captures every LLM call, tool invocation, and performance metric. Combined with Harbor reward scores (added via Step 3), you can filter runs by performance and identify patterns in successful vs. failed runs.

### Common Patterns & Fixes

After running evaluations, analyze failed runs in LangSmith to identify improvement opportunities:

| Pattern                    | Symptom                                              | Potential Fix                              |
|----------------------------|------------------------------------------------------|--------------------------------------------|
| **Poor Planning**          | Agent jumps into coding without reading requirements | Add upfront planning requirement to prompt |
| **Incorrect Tool Usage**   | Uses `bash cat` instead of `read_file`               | Improve tool descriptions with examples    |
| **No Incremental Testing** | Writes 200 lines, then tests once                    | Prompt to test after each logical unit     |
| **Hallucinated Paths**     | Reads files before checking existence                | Add "always `ls` before read" rule         |
| **Wrong Model**            | Model fails on complex reasoning                     | Use more capable model for hard tasks      |

### Agent-Assisted Analysis

Use LangSmith's Insights Agent or your own agent to analyze trajectory data across runs. Task it with identifying common failure patterns, grouping errors by category, and suggesting prompt or tool improvements.

## Available Environments

Harbor supports multiple sandbox environments. Use the `--env` flag to select:

- `docker` - Local Docker containers (good for testing)
- `daytona` - Daytona cloud sandboxes (requires DAYTONA_API_KEY)
- `modal` - Modal cloud compute
- `runloop` - Runloop sandboxes

Makefile shortcuts are available for common workflows:
- `make run-terminal-bench-docker` - Run 1 task locally with Docker
- `make run-terminal-bench-daytona` - Run 10 tasks on Daytona
- `make run-terminal-bench-modal` - Run 4 tasks on Modal
- `make run-terminal-bench-runloop` - Run 10 tasks on Runloop

## Resources

- [DeepAgents Documentation](https://docs.langchain.com/oss/python/deepagents/overview)
- [Harbor GitHub](https://github.com/laude-institute/harbor)
- [LangSmith](https://smith.langchain.com)
