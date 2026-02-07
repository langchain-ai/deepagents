# Terminal Bench + Harbor: The Complete Guide

This is the single guide for running DeepAgents on Terminal Bench 2.0 via Harbor, analyzing results with LangSmith, and iterating on the harness to improve scores.

**What we're doing**: Running our agent harness against ~90 coding tasks, tracing every step to LangSmith, pulling traces down locally, classifying failures, and using those insights to improve prompts, tools, and middleware for the next run.

---

## Table of Contents

1. [Codebase & Architecture](#1-codebase--architecture)
2. [Running a Traced Benchmark](#2-running-a-traced-benchmark)
3. [Harbor Configuration Reference](#3-harbor-configuration-reference)
4. [Outputs & Monitoring](#4-outputs--monitoring)
5. [The Improvement Goal](#5-the-improvement-goal)
6. [Trace Analysis Process](#6-trace-analysis-process)

---

## 1. Codebase & Architecture

### What is DeepAgents?

DeepAgents is an agent harness — a LangGraph-based framework for running AI agents with planning, filesystem tools, shell access, and sub-agent delegation. It's provider-agnostic (Claude, GPT, Gemini, etc.) and ships with validated defaults for coding tasks.

### What is Harbor?

Harbor is an evaluation framework. It loads benchmark tasks, spins up sandboxed environments (Docker, Daytona, Modal), runs your agent, then automatically verifies results with test scripts. Each task gets a reward: `1.0` (pass) or `0.0` (fail).

### What is Terminal Bench 2.0?

A benchmark of ~90 coding tasks ranging from easy (`hello-world`) to very hard (`compile-compcert`). Tasks span software engineering, biology, security, gaming, and more. The agent gets a task description, a sandbox, and must produce a verifiable result.

### Architecture

```
Harbor CLI
    │
    ▼
DeepAgentsWrapper          ← libs/harbor/deepagents_harbor/deepagents_wrapper.py
    │                         Adapts our agent to Harbor's BaseAgent interface.
    │                         Constructs the 3-layer system prompt.
    │                         Sets up middleware stack. Saves ATIF trajectories.
    ▼
create_cli_agent()         ← libs/cli/deepagents_cli/agent.py
    │                         Creates the LangGraph agent with tools + middleware.
    ▼
HarborSandbox              ← libs/harbor/deepagents_harbor/backend.py
    │                         Wraps Harbor's environment for file I/O and shell execution.
    │                         Handles timeouts, output truncation, base64 encoding.
    ▼
Daytona / Docker / Modal      Cloud or local sandbox where code actually runs.
```

### Key Files

| File | What It Does |
|------|-------------|
| `libs/harbor/deepagents_harbor/deepagents_wrapper.py` | Main wrapper. Model init, system prompt construction, middleware setup, ATIF trajectory saving. **This is where most levers live.** |
| `libs/harbor/deepagents_harbor/backend.py` | Sandbox backend. Shell execution with 5-min per-command timeout, file ops, output truncation, build artifact cleanup reminders. |
| `libs/harbor/deepagents_harbor/middleware.py` | Four middleware classes: API error recovery, loop detection, context budget, pre-completion checklist. |
| `libs/harbor/deepagents_harbor/tracing.py` | LangSmith helper — deterministic UUID generation from task instructions. |
| `libs/cli/deepagents_cli/agent.py` | `create_cli_agent()` — builds the LangGraph agent graph. |
| `libs/cli/deepagents_cli/default_agent_prompt.md` | Base agent prompt with general coding best practices. |

### The 3-Layer System Prompt

The agent receives a system prompt built from three pieces (see `_get_full_system_prompt()` at line 417):

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Benchmark Preamble                        │
│  _get_benchmark_preamble() — lines 368-414          │
│                                                     │
│  "YOU ARE IN AN AUTOMATED BENCHMARK. NO HUMAN."     │
│  Sets context: execute don't explain, verify by     │
│  running, find all solutions, try alternatives.     │
│  Lists task legitimacy (bio, security, etc.)        │
├─────────────────────────────────────────────────────┤
│  Layer 2: Default CLI Agent Prompt                  │
│  get_default_coding_instructions()                  │
│                                                     │
│  General agent best practices: planning, tool       │
│  usage patterns, context management strategies.     │
│  Imported from deepagents-cli library.              │
├─────────────────────────────────────────────────────┤
│  Layer 3: Local Sandbox Context                     │
│  _format_local_context() — lines 172-365            │
│                                                     │
│  Dynamic per-task: cwd, language, package manager,  │
│  git info, directory tree, Makefile preview, test    │
│  command. Plus INTEGRITY RULES, CRITICAL REMINDERS  │
│  (12 sections), and PRE-COMPLETION CHECKLIST.       │
└─────────────────────────────────────────────────────┘
```

### The Middleware Stack

Four middleware classes run in order (configured in `deepagents_wrapper.py`):

| Middleware | File:Line | What It Does | Key Thresholds |
|-----------|-----------|-------------|----------------|
| **APIErrorRecoveryMiddleware** | `middleware.py` | Catches recoverable API errors (content filter + invalid image), returns recovery guidance, then injects continuation with `jump_to=model`. Context overflow is non-recoverable and bubbles to Harbor retry logic. | Classifies by error code and message patterns |
| **LoopDetectionMiddleware** | `middleware.py` | Tracks per-file edit counts. Warns when stuck, forces reflection when very stuck. | Defaults: `soft=7`, `hard=12`; both configurable via constructor (and model profile) |
| **ContextBudgetMiddleware** | `middleware.py` | Truncates oversized tool output and warns as context fills using a live estimate from current message state. | Defaults: `max_output_lines=200`, `warn_threshold_percent=70`, `max_context_tokens=128000`; configurable via constructor (and model profile) |
| **PreCompletionCheckMiddleware** | `middleware.py` | On finish attempts, injects checklist, blocks read-only exits (no `execute`/write activity), and enforces at least one verifier/test run when test commands are present in context. | One-time checklist + one-time activity gate + one-time test gate |

### Model Profiles

Harbor now supports runtime tuning profiles to avoid one-size-fits-all thresholds.

- `default`: baseline behavior (recursion 12000, loop 7/12, context budget 128k, summarization max-input 140k)
- `openai_reasoning`: tighter loop/context controls + higher recursion for Codex/GPT-5 style runs (context budget 300k, summarization max-input 400k)
- `anthropic_opus`: more context headroom + less aggressive loop forcing (context budget 180k, summarization max-input 200k by default)

How profile selection works:

- If `model_profile` is set explicitly, that profile is used.
- If omitted, profile is inferred from model name:
  - `codex` or `gpt-5` -> `openai_reasoning`
  - `claude-opus` -> `anthropic_opus`
  - otherwise -> `default`

Override profile with Harbor `--ak`:

```bash
--ak model_profile=openai_reasoning
```

### All Configurable Levers

These are the things you can change to affect agent performance:

| Lever | Location | Default | What It Controls |
|-------|----------|---------|-----------------|
| **Model** | `deepagents_wrapper.py:474` | `anthropic:claude-sonnet-4-5-20250929` | Which LLM runs the agent |
| **Model profile** | `deepagents_wrapper.py` | Auto-inferred | Runtime thresholds bundle (`default`, `openai_reasoning`, `anthropic_opus`) |
| **Temperature** | `deepagents_wrapper.py:447` | `0.0` | Output randomness (0 = deterministic) |
| **Reasoning effort** | `deepagents_wrapper.py:450` | `"high"` | For reasoning models: `"low"`, `"medium"`, `"high"`, `"xhigh"` |
| **Responses API** | `deepagents_wrapper.py:491` | Auto for codex/gpt-5 | Enables OpenAI Responses API for reasoning models |
| **Benchmark preamble** | `deepagents_wrapper.py:368-414` | Fixed text | The "no human present" framing |
| **Integrity rules** | `deepagents_wrapper.py:240-265` | 6 rules | Anti-cheating constraints (no data fabrication, no test overfitting, etc.) |
| **Critical reminders** | `deepagents_wrapper.py:268-349` | 12 sections | Behavior guidance (execute don't describe, verify by running, etc.) |
| **Pre-completion checklist** | `deepagents_wrapper.py:352-363` | 8 items | What agent must verify before finishing |
| **Loop soft threshold** | `middleware.py` | Profile default (`7`) | Edits before warning |
| **Loop hard threshold** | `middleware.py` | Profile default (`12`) | Edits before forced reflection |
| **Max output lines** | `middleware.py` | Profile default (`200`) | Lines before truncation |
| **Context warn %** | `middleware.py` | Profile default (`70`) | % of context before warning |
| **Max context tokens** | `middleware.py` | Profile default (`128000`) | Total context budget |
| **Checklist reminder** | `middleware.py:26-39` | Fixed text | What's injected before agent finishes |
| **Command timeout** | `backend.py:24` | `300` sec (5 min) | Per-command timeout in sandbox |
| **Directory tree depth** | `deepagents_wrapper.py:144` | `3` levels, `30` entries max | How much context agent gets about the sandbox |
| **File listing** | `deepagents_wrapper.py:562` | First `15` files | Files shown to agent at start |
| **Recursion limit** | `deepagents_wrapper.py` | Profile default (`12000`) | Max LangGraph steps before graph stops |
| **Web tools** | `deepagents_wrapper.py:610-612` | `http_request`, `fetch_url`, + `web_search` if Tavily configured | Tools available beyond sandbox |
| **Agent mode** | `deepagents_wrapper.py:479` | `use_cli_agent=True` | CLI agent (full features) vs SDK agent (simpler) |

---

## 2. Running a Traced Benchmark

### Prerequisites

These environment variables must be set (typically in `~/.zshrc`):

```bash
# Model provider keys (at least one required)
export ANTHROPIC_API_KEY="sk-ant-..."    # For Claude models
export OPENAI_API_KEY="sk-..."           # For GPT/Codex models

# LangSmith tracing (required for trace analysis)
export LANGSMITH_API_KEY="lsv2_..."
export LANGSMITH_TRACING=true

# Sandbox (required for cloud runs)
export DAYTONA_API_KEY="..."             # For --env daytona
```

### The Correct Tracing Setup

There are two tracing modes. **Use the project mode for day-to-day runs.**

#### Mode 1: Project Mode (recommended for most runs)

Traces go to a LangSmith project. Simple to browse and filter.

```bash
# IMPORTANT: DeepAgents CLI reads DEEPAGENTS_LANGSMITH_PROJECT, NOT LANGSMITH_PROJECT
export DEEPAGENTS_LANGSMITH_PROJECT="tb2-opus-v1"
export LANGSMITH_TRACING=true
```

#### Mode 2: Experiment Mode (for side-by-side comparison)

Traces are linked to a LangSmith experiment with reference examples. Enables comparison across runs.

```bash
# First, create the experiment:
python scripts/harbor_langsmith.py create-dataset terminal-bench --version 2.0
python scripts/harbor_langsmith.py create-experiment terminal-bench --name opus-baseline-v1

# Then run with:
export LANGSMITH_EXPERIMENT="opus-baseline-v1"
export LANGSMITH_TRACING=true
```

#### Setting a Job ID (critical for trace retrieval)

Every run auto-generates a job ID (`job-<8-hex-chars>`), but you should set a human-readable one:

```bash
export HARBOR_JOB_ID="opus46-tb2-2026-02-05"
```

This job ID is attached as metadata to every trace and is how you'll filter traces later in LangSmith. **Always set this to something descriptive.**

### The Complete Run Command

```bash
cd /path/to/deepagents/libs/harbor

source ~/.zshrc && \
DEEPAGENTS_LANGSMITH_PROJECT=tb2-opus \
LANGSMITH_TRACING=true \
HARBOR_JOB_ID=opus46-run-001 \
uv run harbor run \
  --agent-import-path deepagents_harbor:DeepAgentsWrapper \
  --model anthropic:claude-opus-4-6 \
  --ak model_profile=anthropic_opus \
  --dataset terminal-bench@2.0 \
  -n 20 \
  --jobs-dir jobs/opus46-run-001 \
  --env daytona \
  --timeout-multiplier 1.0
```

### Running in the Background

For long runs (full benchmark takes 1-2 hours):

```bash
source ~/.zshrc && \
DEEPAGENTS_LANGSMITH_PROJECT=tb2-opus \
LANGSMITH_TRACING=true \
HARBOR_JOB_ID=opus46-run-001 \
nohup uv run harbor run \
  --agent-import-path deepagents_harbor:DeepAgentsWrapper \
  --model anthropic:claude-opus-4-6 \
  --ak model_profile=anthropic_opus \
  --dataset terminal-bench@2.0 \
  -n 20 \
  --jobs-dir jobs/opus46-run-001 \
  --env daytona \
  --timeout-multiplier 1.0 \
  > /tmp/harbor-opus46.log 2>&1 &

echo "PID: $!"
```

Monitor the background run:

```bash
# Follow the log
tail -f /tmp/harbor-opus46.log

# Check if still running
ps aux | grep "harbor run"
```

---

## 3. Harbor Configuration Reference

### All Flags

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| **Agent** | | | |
| `--agent-import-path` | Yes | — | Always `deepagents_harbor:DeepAgentsWrapper` |
| `--model` / `-m` | No | `anthropic:claude-sonnet-4-5-20250929` | Model to use (see models table below) |
| `--ak` / `--agent-kwarg` | No | — | Pass kwargs to the wrapper's `__init__`. Format: `key=value`. Repeatable. |
| **Dataset** | | | |
| `--dataset` / `-d` | Yes | — | Always `terminal-bench@2.0` for TB2 |
| `--task-name` / `-t` | No | All tasks | Filter to specific task(s). Supports glob patterns. Repeatable. |
| `--exclude-task-name` / `-x` | No | — | Exclude specific task(s). Supports glob patterns. Repeatable. |
| **Job Settings** | | | |
| `--jobs-dir` / `-o` | Yes | — | Output directory for results |
| `--job-name` | No | Timestamp | Human-readable name for the job |
| `-k` / `--n-attempts` | No | `1` | **Number of trials per task.** Use `-k 2` for 2 independent attempts per task. |
| `--timeout-multiplier` | No | `1.0` | Multiply base 15-min timeout. Use `4.0` (1hr) or `8.0` (2hr). |
| `--debug` | No | — | Enable debug logging |
| **Orchestrator** | | | |
| `-n` / `--n-concurrent` | No | `4` | Number of **parallel workers** (NOT total tasks!) |
| `-r` / `--max-retries` | No | `0` | Retry failed trials N times |
| `--retry-include` | No | All except below | Exception types to retry on. Repeatable. |
| `--retry-exclude` | No | `AgentTimeoutError`, `VerifierTimeoutError`, etc. | Exception types to NOT retry on. Repeatable. |
| **Environment** | | | |
| `--env` / `-e` | No | `docker` | `daytona` (cloud), `docker` (local), `modal`, `runloop`, `gke` |
| `--no-delete` | No | Deletes | Keep sandbox alive after completion (useful for debugging) |
| `--override-cpus` | No | — | Override sandbox CPU count |
| `--override-memory` | No | — | Override sandbox memory (MB) |
| `--override-storage` | No | — | Override sandbox storage (MB) |
| `--override-gpus` | No | — | Override sandbox GPU count |
| **Traces** | | | |
| `--export-traces` | No | — | Export traces from job directory after completion |

**Agent kwargs** (`--ak`) are passed directly to `DeepAgentsWrapper.__init__()`. Key ones:

| Agent Kwarg | Default | Description |
|-------------|---------|-------------|
| `reasoning_effort=xhigh` | `high` | Reasoning effort for reasoning models (`low`, `medium`, `high`, `xhigh`) |
| `model_profile=default` | auto-inferred | Runtime profile (`default`, `openai_reasoning`, `anthropic_opus`) |
| `max_input_tokens=400000` | profile default | Override model context size used by summarization trigger logic |
| `max_context_tokens=300000` | profile default | Override `ContextBudgetMiddleware` token budget |
| `anthropic_betas=context-1m-2025-08-07` | auto for `claude-opus-4-6` | Comma-separated Anthropic beta flags passed to the model client. Opus 4.6 now defaults to `context-1m-2025-08-07` when unset. |
| `temperature=0.0` | `0.0` | Model temperature |
| `use_cli_agent=true` | `true` | Use CLI agent (vs SDK agent) |
| `experiment_name=my-exp` | — | Experiment name for LangSmith grouping |
| `experiment_tags=tag1,tag2` | — | Comma-separated tags for LangSmith filtering |

### Critical: `-n` Is Concurrent Workers

```bash
# This runs ALL ~90 tasks with 20 parallel workers
-n 20

# To run only specific tasks, use -t
-t chess-best-move -t regex-log -t password-recovery -n 3
```

### Available Models

| Model | Flag | Notes |
|-------|------|-------|
| Claude Sonnet 4.5 | `--model anthropic:claude-sonnet-4-5-20250929` | Default. Good balance. |
| Claude Opus 4.5 | `--model anthropic:claude-opus-4-5-20251101` | More capable, slower, more expensive. |
| Claude Opus 4.6 | `--model anthropic:claude-opus-4-6` | Latest Opus. |
| GPT-5.2-codex | `--model openai:gpt-5.2-codex` | Auto-enables Responses API + reasoning. |
| GPT-4o | `--model openai:gpt-4o` | Faster, less capable. |

### Timeout Policy

**Always use `--timeout-multiplier 1.0` (the default 15min base).** The agent must work within the standard time budget. Increasing timeouts masks agent inefficiency rather than fixing it.

### Example Runs

**3 specific tasks:**
```bash
DEEPAGENTS_LANGSMITH_PROJECT=tb2 LANGSMITH_TRACING=true \
HARBOR_JOB_ID=test-3-tasks \
uv run harbor run \
  --agent-import-path deepagents_harbor:DeepAgentsWrapper \
  --model anthropic:claude-opus-4-6 \
  --ak model_profile=anthropic_opus \
  --dataset terminal-bench@2.0 \
  -t hello-world -t regex-log -t chess-best-move \
  -n 3 \
  --jobs-dir jobs/test-3-tasks \
  --env daytona \
  --timeout-multiplier 1.0
```

**Full benchmark (all ~90 tasks):**
```bash
DEEPAGENTS_LANGSMITH_PROJECT=tb2 LANGSMITH_TRACING=true \
HARBOR_JOB_ID=full-opus46-v1 \
uv run harbor run \
  --agent-import-path deepagents_harbor:DeepAgentsWrapper \
  --model anthropic:claude-opus-4-6 \
  --ak model_profile=anthropic_opus \
  --dataset terminal-bench@2.0 \
  -n 20 \
  --jobs-dir jobs/full-opus46-v1 \
  --env daytona \
  --timeout-multiplier 1.0
```

### Task Difficulty Tiers

| Tier | Examples | Typical Pass Rate |
|------|----------|-------------------|
| Easy | `hello-world`, `password-recovery` | 90%+ |
| Medium | `regex-log`, `pytorch-model-cli`, `log-summary-date-ranges` | 70-90% |
| Hard | `chess-best-move`, `path-tracing`, `git-multibranch` | 40-70% |
| Very Hard | `gpt2-codegolf`, `compile-compcert`, `caffe-cifar-10`, `make-doom-for-mips` | 0-40% |

### Resuming a Failed or Interrupted Run

If a run gets interrupted (Ctrl+C, machine crash, Daytona flakiness), you don't have to re-run everything. Harbor can resume from the job directory, picking up only the incomplete trials:

```bash
uv run harbor jobs resume -p jobs/full-opus46-v1/<timestamp>
```

The `-p` path must point to the timestamped job directory containing `config.json`.

**Filtering out error types before resuming**: If certain trials failed due to infrastructure errors (e.g., DaytonaError) and you want to retry those specifically:

```bash
# Resume but remove DaytonaError trials so they get re-run
uv run harbor jobs resume \
  -p jobs/full-opus46-v1/<timestamp> \
  -f DaytonaError
```

You can pass `-f` multiple times to filter out multiple error types:

```bash
uv run harbor jobs resume \
  -p jobs/full-opus46-v1/<timestamp> \
  -f DaytonaError \
  -f ProcessInterruptedError
```

This removes the failed trials matching those error types from the completed set, so Harbor will re-run them.

### Listing All Available Tasks

```bash
uv run harbor datasets download terminal-bench@2.0
find ~/.cache/harbor/tasks -name "instruction.md" | \
  xargs -I {} dirname {} | xargs -I {} basename {} | sort
```

---

## 4. Outputs & Monitoring

### Output Directory Structure

Every run creates this structure:

```
jobs/<job-name>/<timestamp>/
├── config.json                       # Job configuration (flags, model, etc.)
├── job.log                           # Harbor's job-level log
├── result.json                       # AGGREGATED results — pass rate, per-task rewards
└── <task-name>__<random-id>/         # One directory per task trial
    ├── config.json                   # Trial-level config
    ├── trial.log                     # Trial log
    ├── exception.txt                 # If the trial crashed (error message)
    ├── result.json                   # Trial result with reward (0.0 or 1.0)
    ├── agent/
    │   └── trajectory.json           # ATIF format — full execution log
    └── verifier/
        ├── reward.txt                # "0" or "1"
        └── test-stdout.txt           # What the verifier tests printed
```

### Key Files to Check

**Overall pass rate** — the top-level `result.json`:

```bash
JOB_DIR="jobs/<name>/<timestamp>"

cat "$JOB_DIR/result.json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for name, data in d['stats']['evals'].items():
    print(f'Pass Rate: {data[\"metrics\"][0][\"mean\"]:.1%}')
    passed = data['reward_stats']['reward'].get('1.0', [])
    failed = data['reward_stats']['reward'].get('0.0', [])
    print(f'Passed ({len(passed)}): {passed}')
    print(f'Failed ({len(failed)}): {failed}')
"
```

**Per-task results** — quick scan:

```bash
JOB_DIR="jobs/<name>/<timestamp>"

for d in "$JOB_DIR"/*/; do
  [ -d "$d" ] || continue
  task=$(basename "$d")
  [[ "$task" == *.json || "$task" == *.log ]] && continue
  if [ -f "${d}result.json" ]; then
    reward=$(grep -o '"reward": [0-9.]*' "${d}result.json" | head -1 | grep -o '[0-9.]*')
    if [ "$reward" = "1.0" ]; then echo "PASS $task"; else echo "FAIL $task"; fi
  else
    echo ".... $task"
  fi
done
```

**Why a task failed** — check verifier and trajectory:

```bash
TASK_DIR="jobs/<name>/<timestamp>/<task>__<id>"

# What did the test say?
cat "$TASK_DIR/verifier/test-stdout.txt"

# Did it crash?
cat "$TASK_DIR/exception.txt" 2>/dev/null

# What did the agent do? (last few steps)
cat "$TASK_DIR/agent/trajectory.json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Steps: {len(d[\"steps\"])}')
print(f'Tokens: {d[\"final_metrics\"]}')
print(f'Last step: {d[\"steps\"][-1][\"message\"][:500]}')
"
```

### Monitoring a Running Job

**Live progress watch:**

```bash
watch -n 30 'JOB_DIR=$(ls -dt jobs/<name>/*/ | head -1)
for d in "$JOB_DIR"*/; do
  [ -d "$d" ] || continue
  task=$(basename "$d")
  [[ "$task" == *.json || "$task" == *.log ]] && continue
  if [ -f "${d}result.json" ]; then
    reward=$(grep -o "\"reward\": [0-9.]*" "${d}result.json" 2>/dev/null | head -1 | grep -o "[0-9.]*")
    if [ "$reward" = "1.0" ]; then echo "PASS $task"; else echo "FAIL $task"; fi
  else
    echo ".... $task"
  fi
done | sort'
```

**For background runs:**

```bash
# Check the nohup log
tail -20 /tmp/harbor-opus46.log

# Is the process still running?
ps aux | grep "harbor run" | grep -v grep

# Kill if needed
pkill -f "harbor run"
```

### ATIF Trajectory Format

The `trajectory.json` file uses ATIF (Agent Trajectory Interchange Format):

```json
{
  "schema_version": "ATIF-v1.2",
  "session_id": "chess-best-move__a1b2c3d",
  "agent": {
    "name": "deepagent-harbor",
    "version": "0.0.1",
    "model_name": "anthropic:claude-opus-4-6"
  },
  "steps": [
    {"step_id": 1, "source": "user", "message": "Your task is to..."},
    {"step_id": 2, "source": "agent", "message": "I'll start by...",
     "tool_calls": [{"function_name": "execute", "arguments": {"command": "ls"}}],
     "observation": {"results": [{"content": "file1.py\nfile2.py"}]}}
  ],
  "final_metrics": {
    "total_prompt_tokens": 150000,
    "total_completion_tokens": 5000,
    "total_steps": 25
  }
}
```

---

## 5. The Improvement Goal

### What We're Optimizing

We're building the best possible agent harness for coding tasks. The score on Terminal Bench is our proxy metric: **pass rate across ~90 tasks**. But the real goal is understanding *why* our agent fails and fixing the harness — not gaming the benchmark.

### The Iterative Loop

```
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │  1. RUN           Run benchmark with current harness│
  │       │                                             │
  │       ▼                                             │
  │  2. TRACE         Pull LangSmith traces + local     │
  │       │           trajectories for the run          │
  │       ▼                                             │
  │  3. ANALYZE       Classify failures into categories │
  │       │           (model logic, tool issues, etc.)  │
  │       ▼                                             │
  │  4. IDENTIFY      Find patterns — which failures    │
  │       │           are systematic? What's high-ROI?  │
  │       ▼                                             │
  │  5. BRAINSTORM    Propose harness changes:          │
  │       │           prompts, tools, middleware,        │
  │       │           multi-agent strategies             │
  │       ▼                                             │
  │  6. IMPLEMENT     Make the changes                  │
  │       │                                             │
  │       └─────────────────── REPEAT ──────────────────┘
```

### What We Can Change (The Improvement Dimensions)

#### Prompts
The biggest lever. The 3-layer system prompt is ~2000+ lines of guidance. Changes here include:
- Adding/removing critical reminders
- Changing the planning approach (e.g., requiring upfront planning vs. letting agent decide)
- Improving integrity rules
- Adding domain-specific hints for task categories
- Tuning the pre-completion checklist

#### Tools
The agent has: `execute`, `read_file`, `write_file`, `edit_file`, `ls`, `glob`, `grep`, `write_todos`, `read_todos`, `task` (sub-agents), `http_request`, `fetch_url`, `web_search`.

Changes here include:
- Adding new tools (e.g., specialized code analysis, test runners)
- Improving tool descriptions and examples so the model uses them better
- Modifying `HarborSandbox` behavior (timeouts, output handling, cleanup reminders)

#### Middleware
The four middleware classes intercept every model call and tool call. Changes here include:
- Adjusting thresholds (loop detection at 7/12, context budget at 200 lines / 70%)
- Adding new middleware (e.g., strategy tracking, progress monitoring)
- Changing recovery prompts (what the agent sees when it hits a limit)
- Adding middleware that inspects tool call patterns and suggests pivots

#### Multi-Agent Strategies
The `task` tool already lets the agent spawn sub-agents. Changes here include:
- Adding specialized sub-agents for specific task types
- Having a planner agent that breaks down the task before the executor starts
- Using a critic/reviewer agent that checks work before the agent declares done

#### Model Selection
Different models for different tasks or roles:
- Stronger model for hard tasks, cheaper model for easy ones
- Different reasoning effort levels
- Temperature tuning for exploration vs. exploitation

### What Good Looks Like

A good improvement cycle:
1. **Specific**: "14 tasks failed because the agent didn't verify output format" → add format verification to checklist
2. **Measurable**: "This change should fix at least 8 of those 14 failures"
3. **Targeted**: Changes one thing at a time so you can attribute score changes

A bad improvement cycle:
1. Rewriting the entire prompt based on vibes
2. Changing 5 things at once so you can't tell what helped
3. Overfitting to specific tasks instead of fixing systemic issues

---

## 6. Trace Analysis Process

This is the most important section. After running a benchmark, here's how you extract insights.

### Data Sources

You have two sources of truth for each run:

| Source | What It Contains | Best For |
|--------|-----------------|----------|
| **Local job directory** (`jobs/`) | `result.json` (rewards), `trajectory.json` (ATIF steps), `exception.txt`, verifier output | Quick triage: which tasks passed/failed/errored, verifier output, exceptions |
| **LangSmith traces** | Full conversation history with every LLM call, tool invocation, token counts, timing, metadata | Deep analysis: reading the agent's reasoning, understanding decision points, comparing strategies |

LangSmith traces are richer because they include the model's reasoning content and individual LLM call metadata that the ATIF trajectory doesn't capture.

### Step 1: Quick Triage from Local Results

Before pulling traces, understand the overall picture from `result.json`:

```bash
JOB_DIR="jobs/<name>/<timestamp>"

cat "$JOB_DIR/result.json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
evals = d['stats']['evals']
for name, data in evals.items():
    print(f'=== {name} ===')
    print(f'Pass Rate: {data[\"metrics\"][0][\"mean\"]:.1%}')

    # Passed/failed
    rewards = data['reward_stats']['reward']
    passed = rewards.get('1.0', [])
    failed = rewards.get('0.0', [])
    print(f'Passed ({len(passed)}): {sorted(passed)}')
    print(f'Failed ({len(failed)}): {sorted(failed)}')

    # Errors
    exceptions = data.get('exception_stats', {})
    for etype, trials in exceptions.items():
        print(f'{etype} ({len(trials)}): {sorted(trials)}')
"
```

This gives you the split: how many passed, failed, and errored (and which error type).

### Step 2: Download LangSmith Traces

Use the `langsmith-trace-analyzer` skill and its `download_traces.py` helper script.

#### Option A: Using the helper script

```bash
cd "$JOB_DIR"

# Create trace mapping from LangSmith (queries by job_id)
python /path/to/deepagents/skills/langsmith-trace-analyzer/download_traces.py \
  --job-id "opus46-run-001" \
  --project "tb2-opus" \
  --result-file result.json \
  --output-dir langsmith-traces \
  --create-mapping
```

This will:
1. Query LangSmith for all traces with `job_id = opus46-run-001`
2. Create a `langsmith_trace_mapping.json` linking trial names to trace IDs
3. Download each trace using `langsmith-fetch`
4. Organize them into `langsmith-traces/by-outcome/{passed,failed,errors}/`
5. Create a `manifest.json` with summary stats

#### Option B: Manual with langsmith-fetch CLI

```bash
# Install if needed
pip install langsmith-fetch

# Fetch recent traces from project
langsmith-fetch traces ./langsmith-traces \
  --project-uuid <your-project-uuid> \
  --limit 100 \
  --last-n-minutes 120 \
  --include-metadata \
  --format raw
```

#### Option C: Python SDK for custom filtering

```python
from langsmith import Client

client = Client()
job_id = "opus46-run-001"
project = "tb2-opus"

filter_query = f'and(eq(metadata_key, "job_id"), eq(metadata_value, "{job_id}"))'

for run in client.list_runs(
    project_name=project,
    filter=filter_query,
    is_root=True
):
    print(f"{run.id}: {run.status}, {run.metadata.get('task_name')}")
```

### Step 3: Organize Traces by Outcome

The `download_traces.py` script does this automatically, but the target structure is:

```
langsmith-traces/
├── manifest.json              # Index: trace_id → trial_name → outcome → file path
├── by-outcome/
│   ├── passed/                # reward = 1.0
│   │   └── {task}__{id}.json
│   ├── failed/                # reward = 0.0
│   │   └── {task}__{id}.json
│   └── errors/                # Exceptions
│       ├── GraphRecursionError/
│       ├── AgentTimeoutError/
│       ├── DaytonaError/
│       └── ProcessInterruptedError/
└── by-task/                   # Optional: group by task name
    └── {task_name}/
```

### Step 4: Classify Failures

Use this framework to categorize every non-passing trial:

#### Category 1: Infrastructure Errors (not agent failures)

| Error Type | What Happened | Action |
|-----------|---------------|--------|
| `DaytonaError` | Sandbox failed to provision | Exclude from agent analysis. Re-run these tasks. |
| `ProcessInterruptedError` | Run was manually killed | Note as incomplete. |
| `BadRequestError` | Malformed API request | Check if agent caused it or if it's a harness bug. |

#### Category 2: Resource Limit Errors (agent ran but hit a wall)

| Error Type | What Happened | Key Question |
|-----------|---------------|-------------|
| `GraphRecursionError` | Hit step limit (recursion_limit) | Was the agent looping or making slow progress? |
| `AgentTimeoutError` | Hit wall-clock timeout | Was the agent working or idle? |

**Stuck in loop** = agent failure (repeated same action). **Slow but progressing** = may need higher limits.

#### Category 3: Task Failures (reward = 0.0) — three sub-types

**Model/Logic failures:**

| Pattern | Evidence in Trace |
|---------|------------------|
| Misunderstanding | Early divergence from task spec |
| Wrong strategy | Committed to bad plan, didn't pivot |
| Incomplete solution | Missing edge cases, partial output |
| Gave up early | Few attempts, premature "I can't do this" |
| No verification | No test runs, no output checks |
| Format error | Right answer, wrong file path or format |

**Tool/Execution failures:**

| Pattern | Evidence in Trace |
|---------|------------------|
| Wrong tool or args | Syntax errors, wrong flags |
| Environment mismatch | Missing dependencies, wrong paths |
| Error non-recovery | Repeated same failing command without adapting |
| Tool timeout | Long gaps, partial output |

**Capability gaps:**

| Pattern | Evidence in Trace |
|---------|------------------|
| Knowledge gap | Wrong approaches for the domain |
| Skill gap | Can't write valid code in required language |
| Context limit | Forgetting earlier information |

### Step 5: Analyze at Scale (Parallel Agent Analysis)

For large runs (~90+ traces), use parallel analysis with the `langsmith-trace-analyzer` skill:

1. **Create a manifest** from your downloaded traces:

```python
import json
from pathlib import Path

traces_dir = Path("langsmith-traces/by-outcome")
manifest = {"traces": [], "total": 0}

for outcome in ["passed", "failed"]:
    for f in (traces_dir / outcome).glob("*.json"):
        manifest["traces"].append({
            "file": str(f),
            "trial_name": f.stem,
            "task_name": f.stem.rsplit("__", 1)[0],
            "outcome": outcome
        })

# Add error traces
for error_dir in (traces_dir / "errors").iterdir():
    if error_dir.is_dir():
        for f in error_dir.glob("*.json"):
            manifest["traces"].append({
                "file": str(f),
                "trial_name": f.stem,
                "task_name": f.stem.rsplit("__", 1)[0],
                "outcome": "error",
                "error_type": error_dir.name
            })

manifest["total"] = len(manifest["traces"])
with open("analysis_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
```

2. **Split into batches** and assign to parallel agents (4-6 agents, 50-100 traces each)

3. **Each agent reads traces and writes findings** as JSONL:

```json
{"trial_name": "chess__abc", "task_name": "chess-best-move", "outcome": "failed", "category": "model_logic", "subcategory": "wrong_strategy", "summary": "Agent tried brute-force search instead of using the chess engine", "root_cause": "Didn't explore available tools in the sandbox", "actionable": "Add reminder to check for pre-installed tools before implementing from scratch", "confidence": "high"}
```

4. **Aggregate findings**:

```python
import json
from collections import defaultdict

findings = [json.loads(line) for line in open("findings.jsonl") if line.strip()]

# By category
by_cat = defaultdict(list)
for f in findings:
    by_cat[f["category"]].append(f)

print("=== Failure Breakdown ===")
for cat, items in sorted(by_cat.items(), key=lambda x: -len(x[1])):
    print(f"\n{cat}: {len(items)}")
    by_sub = defaultdict(int)
    for i in items:
        by_sub[i.get("subcategory", "?")] += 1
    for sub, count in sorted(by_sub.items(), key=lambda x: -x[1]):
        print(f"  {sub}: {count}")

# Top actionable insights
print("\n=== Top Actionable Insights ===")
by_action = defaultdict(int)
for f in findings:
    if f.get("actionable") and f["actionable"].lower() != "none":
        by_action[f["actionable"]] += 1
for action, count in sorted(by_action.items(), key=lambda x: -x[1])[:15]:
    print(f"  ({count}x) {action}")
```

### Step 6: Brainstorm Changes

With categorized findings, map failures to improvement dimensions:

| Failure Category | Likely Fix Dimension | Example Changes |
|-----------------|---------------------|-----------------|
| **Misunderstanding** (model_logic) | Prompt | Add "restate the task requirements before starting" to planning guidance |
| **No verification** (model_logic) | Prompt + Middleware | Strengthen pre-completion checklist; add middleware that detects missing verification |
| **Wrong tool/args** (tool_issue) | Tool descriptions | Improve tool docstrings with examples for the failing patterns |
| **Error non-recovery** (tool_issue) | Middleware + Prompt | Lower loop detection threshold; add "try a different approach after 2 failures" |
| **Gave up early** (model_logic) | Prompt | Add "try at least 3 different approaches before declaring failure" |
| **Environment mismatch** (tool_issue) | Sandbox context | Show more environment details in Layer 3 of prompt |
| **Context limit** (capability_gap) | Middleware | Lower context budget warning threshold; add summarization middleware |
| **Stuck in loop** (resource_limit) | Middleware | Lower hard reflection threshold from 12 to 8 |
| **Knowledge gap** (capability_gap) | Model / Multi-agent | Use stronger model; add domain-specific sub-agent |

### Metrics to Track Across Runs

| Metric | Formula | What It Tells You |
|--------|---------|------------------|
| **Pass rate** | passed / total | Overall score |
| **Clean pass rate** | passed / (passed + failed) | Score excluding infra errors |
| **Error rate** | errors / total | Harness reliability |
| **Tokens per pass** | total_tokens / passed | Efficiency |
| **Category breakdown** | % model_logic, % tool_issue, etc. | Where to focus next |
| **Consistency** | tasks with 3/3 pass vs 0/3 pass | Reliability |

---

## Troubleshooting

### Traces Not Appearing in LangSmith

```bash
# Make sure you're using DEEPAGENTS_LANGSMITH_PROJECT, not LANGSMITH_PROJECT
DEEPAGENTS_LANGSMITH_PROJECT=tb2 LANGSMITH_TRACING=true uv run harbor run ...
```

### Daytona Stuck or Slow

- Some images (ML tasks, compilers) take longer to provision
- Reduce concurrent workers: `-n 5` instead of `-n 20`
- Try local Docker for testing: `--env docker`
- For very hard tasks: `--timeout-multiplier 8.0`

### Conversation History Offload Warnings

If you see repeated warnings like:
- `Failed to offload conversation history to /conversation_history/<thread>.md ...`
- `Offloading conversation history to backend failed during summarization.`

Current behavior (source of truth):
- For sandbox backends, summarization now appends history incrementally via `execute`/`aexecute` in chunks.
- This avoids full-file read+rewrite as history grows, which previously caused repeated `already exists` errors and command-size failures.
- If command-append fails, middleware falls back to the legacy read+write/edit path.

If warnings still appear, check these first:
- Confirm your backend `adownload_files()` path only applies image integrity validation to known image extensions (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`), not all file types.
- Confirm non-image files (especially `/conversation_history/*.md`) can be downloaded without image validation errors.
- Confirm backend `aedit()` supports exact multiline replacements for fallback paths.
- Run unit tests:

```bash
uv run pytest /Users/vivektrivedy/Documents/da1/deepagents/libs/deepagents/tests/unit_tests/middleware/test_summarization_middleware.py
uv run pytest tests/unit_tests/test_backend_image_validation.py tests/unit_tests/test_backend_edit.py
```

### Context Window And Summarization Too-Early Behavior

If summarization appears to happen too early for large-window models:
- Verify your run sets an appropriate `max_input_tokens` value (for example, `400000` for GPT-5.2 Codex).
- If using Anthropic long-context beta features, pass required beta flags through `anthropic_betas=...` and set matching `max_input_tokens`.
- Keep `max_context_tokens` lower than `max_input_tokens` to preserve headroom for tool output and response generation.
- Summarization trigger is `75%` of `max_input_tokens` for models with profile limits.
- If a provider still returns `context_length_exceeded`, Harbor middleware now bubbles that error (no in-agent recovery loop) so the orchestrator can retry the trial attempt with fresh context.

Example:

```bash
# GPT-5.2 Codex
--ak model_profile=openai_reasoning --ak max_input_tokens=400000 --ak max_context_tokens=300000

# Anthropic long-context beta (Opus 4.6 enables this by default)
--ak model_profile=anthropic_opus --ak max_input_tokens=1000000 --ak max_context_tokens=500000

# Explicit override (or disable with empty string)
--ak model_profile=anthropic_opus --ak anthropic_betas=context-1m-2025-08-07
```

### Task Timeout

**Always use `--timeout-multiplier 1.0`.** The agent must solve tasks within the default 15min budget.

### "OPENAI_API_KEY not set" or Similar

```bash
source ~/.zshrc   # Reload env vars before running
```

### Finding Your LangSmith Project UUID

```python
from langsmith import Client
client = Client()
for p in client.list_projects():
    print(f"{p.name}: {p.id}")
```
