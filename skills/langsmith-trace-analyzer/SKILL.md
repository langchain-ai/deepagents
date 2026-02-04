---
name: langsmith-trace-analyzer
description: Fetch, organize, and analyze LangSmith traces and threads. Use when working with LangSmith data, downloading traces, debugging agent runs, analyzing benchmark results (like Terminal Bench), or investigating failures. Covers the langsmith-fetch CLI, Python SDK, and analysis workflows.
allowed-tools: Read, Grep, Glob, Bash, Write, Edit, Task
---

# LangSmith Trace Analyzer

A comprehensive guide for fetching and analyzing traces from LangSmith projects.

## Linked Guides

| Guide | Description |
|-------|-------------|
| [terminal-bench-analysis.md](./terminal-bench-analysis.md) | Complete guide for analyzing Terminal Bench/Harbor benchmark results including failure categorization, storage patterns, and reporting |
| [parallel-analysis.md](./parallel-analysis.md) | Patterns for analyzing many traces in parallel using multiple agents writing to shared output |

---

## Part 1: Understanding LangSmith Data

### Core Concepts

LangSmith organizes data hierarchically:

```
Project (e.g., "my-agent-prod")
└── Threads (conversation sessions)
    └── Traces (single executions/requests)
        └── Runs (individual LLM calls, tool executions)
```

- **Run**: A single operation (LLM call, tool execution, chain step)
- **Trace**: A tree of runs representing one complete execution (one user request → response)
- **Thread**: A collection of traces representing a multi-turn conversation
- **Project**: A container for all traces/threads from an application

### What's in a Trace?

Each trace contains:
- **Messages**: The conversation (user prompts, assistant responses, tool calls/results)
- **Metadata**: Status, timing, token counts, costs, custom metadata
- **Feedback**: Human or automated evaluations attached to the trace

---

## Part 2: Tools for Fetching Traces

You have two options: the `langsmith-fetch` CLI or the Python SDK directly.

### Option A: langsmith-fetch CLI

**What it is**: A command-line tool optimized for fetching traces/threads.

**Installation**:
```bash
pip install langsmith-fetch
```

**When to use it**:
- Fetching traces by ID (you know the specific trace)
- Fetching recent traces by time (last N minutes, since timestamp)
- Bulk downloading to a directory
- Human-readable output for inspection
- Simple automation scripts

**When NOT to use it**:
- Filtering by custom metadata (job_id, task_name, etc.)
- Complex queries across traces
- Programmatic analysis requiring SDK objects

**Setup**:
```bash
export LANGSMITH_API_KEY=lsv2_...
export LANGSMITH_PROJECT=your-project-name  # Optional but recommended
```

**Key Commands**:

```bash
# Fetch single trace by ID
langsmith-fetch trace <trace-id>
langsmith-fetch trace <trace-id> --format raw --include-metadata --file output.json

# Fetch recent traces to directory (RECOMMENDED for bulk)
langsmith-fetch traces ./output-dir --limit 50 --include-metadata

# With time filtering
langsmith-fetch traces ./output --last-n-minutes 60 --limit 100
langsmith-fetch traces ./output --since 2026-01-01T00:00:00Z

# Fetch threads
langsmith-fetch thread <thread-id> --project-uuid <uuid>
langsmith-fetch threads ./output --limit 20
```

**All Flags**:

| Flag | Commands | Description | Default |
|------|----------|-------------|---------|
| `--project-uuid` | all | LangSmith project UUID | From env/config |
| `-n, --limit` | traces, threads | Max items to fetch | 1 |
| `--last-n-minutes` | traces, threads | Time window filter | None |
| `--since` | traces, threads | ISO timestamp filter | None |
| `--format` | all | `pretty`, `json`, or `raw` | `pretty` |
| `--file` | trace, thread | Save to file | stdout |
| `--include-metadata` | trace, traces | Include timing/tokens/costs | No |
| `--include-feedback` | trace, traces | Include feedback data | No |
| `--max-concurrent` | traces, threads | Parallel fetches (max 10) | 5 |
| `--filename-pattern` | traces, threads | Pattern with `{trace_id}`, `{index}` | `{trace_id}.json` |

**Output Formats**:
- `pretty`: Human-readable with colors (default, for terminal viewing)
- `json`: Pretty-printed JSON with syntax highlighting
- `raw`: Compact JSON, one line (best for programmatic use)

### Option B: Python SDK

**What it is**: The official LangSmith Python library with full API access.

**Installation**:
```bash
pip install langsmith
```

**When to use it**:
- Filtering by custom metadata (job_id, experiment tags, etc.)
- Complex queries (status, time ranges, metadata combinations)
- Building analysis pipelines
- When you need Run objects with all attributes

**Basic Usage**:

```python
from langsmith import Client

client = Client()  # Uses LANGSMITH_API_KEY from environment

# List recent traces
for run in client.list_runs(
    project_name="my-project",
    is_root=True,  # Only root traces, not child runs
    limit=10
):
    print(f"{run.id}: {run.status}, {run.total_tokens} tokens")

# Filter by custom metadata
filter_query = 'and(eq(metadata_key, "job_id"), eq(metadata_value, "abc123"))'
for run in client.list_runs(
    project_name="my-project",
    filter=filter_query,
    is_root=True
):
    print(run.id, run.metadata)

# Filter by status
for run in client.list_runs(
    project_name="my-project",
    filter='eq(status, "error")',
    is_root=True
):
    print(f"Error trace: {run.id}")

# Filter by time
from datetime import datetime, timedelta, timezone
start = datetime.now(timezone.utc) - timedelta(hours=24)
for run in client.list_runs(
    project_name="my-project",
    start_time=start,
    is_root=True
):
    print(run.id)
```

**Getting Full Trace Data**:

The SDK's `list_runs` returns Run objects with metadata but not full messages. To get messages, either:

1. Use `langsmith-fetch` CLI for the trace ID
2. Use the REST API directly:

```python
import requests

def fetch_trace_messages(trace_id, api_key):
    headers = {"X-API-Key": api_key}
    url = f"https://api.smith.langchain.com/runs/{trace_id}?include_messages=true"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    return data.get("messages") or data.get("outputs", {}).get("messages", [])
```

### Choosing Between CLI and SDK

| Scenario | Use |
|----------|-----|
| "Fetch this specific trace" | CLI: `langsmith-fetch trace <id>` |
| "Get last 50 traces" | CLI: `langsmith-fetch traces ./dir --limit 50` |
| "Find all traces for job X" | SDK: filter by metadata |
| "Find all error traces" | SDK: filter by status |
| "Download traces, organize by outcome" | SDK to find IDs → CLI to download |
| "Quick inspection" | CLI with `--format pretty` |
| "Build analysis pipeline" | SDK |

---

## Part 3: Trace Data Structure

When you fetch a trace with `--include-metadata`, you get:

```json
{
  "trace_id": "019c2754-dcf0-7971-ad86-ee82ed690b8a",
  "messages": [
    {"role": "user", "content": "Your task is to..."},
    {"role": "assistant", "content": [{"type": "tool_use", "name": "bash", ...}]},
    {"role": "tool", "content": "command output..."},
    {"role": "assistant", "content": "I've completed..."}
  ],
  "metadata": {
    "status": "success",
    "start_time": "2026-02-04T06:26:38.960267",
    "end_time": "2026-02-04T06:30:08.010179",
    "duration_ms": 209049,
    "token_usage": {
      "prompt_tokens": 413775,
      "completion_tokens": 2122,
      "total_tokens": 415897
    },
    "custom_metadata": {
      "job_id": "7f07b2d0-...",
      "task_name": "my-task",
      "model": "gpt-4",
      ...
    }
  },
  "feedback": []
}
```

**Key Fields**:
- `status`: `"success"`, `"error"`, or `"pending"`
- `messages`: The conversation history
- `custom_metadata`: Your application's metadata (task IDs, experiment tags, etc.)
- `token_usage`: LLM token counts
- `duration_ms`: Wall clock time

---

## Part 4: Storage Patterns

### Directory Structure for Trace Archives

When downloading many traces, organize them for easy access:

```
traces/
├── manifest.json           # Index of all traces with metadata
├── by-id/                  # Quick lookup by trace ID
│   └── {trace_id}.json
├── by-outcome/             # Organized by result
│   ├── passed/
│   ├── failed/
│   └── errors/
│       ├── TimeoutError/
│       └── OtherError/
└── by-task/                # For benchmark runs
    └── {task_name}/
        ├── trial_1.json
        └── trial_2.json
```

### Manifest File

Always create a manifest linking traces to their context:

```json
{
  "created_at": "2026-02-04T14:00:00Z",
  "source": {
    "project": "my-project",
    "job_id": "abc123",
    "filter": "traces from benchmark run X"
  },
  "summary": {
    "total": 267,
    "by_status": {"success": 235, "error": 32},
    "by_outcome": {"passed": 179, "failed": 56, "error": 32}
  },
  "traces": {
    "trace-id-1": {
      "file": "by-outcome/passed/task__id.json",
      "task": "task-name",
      "outcome": "passed",
      "tokens": 50000,
      "duration_ms": 120000
    },
    ...
  }
}
```

### Naming Conventions

For benchmark/experiment traces:
- `{task_name}__{trial_id}.json` - Easy to parse, matches trial directories
- Example: `crack-7z-hash__Lu7oSkD.json`

For general traces:
- `{trace_id}.json` - Unambiguous, matches LangSmith
- Or `{timestamp}_{trace_id}.json` - Sortable by time

---

## Part 5: Common Workflows

### Workflow: Download All Traces for a Job

When you have a job_id (e.g., from a benchmark run):

```python
import json
import subprocess
import time
from langsmith import Client

# Step 1: Find all traces for this job
client = Client()
job_id = "your-job-id"
project = "your-project"

filter_query = f'and(eq(metadata_key, "job_id"), eq(metadata_value, "{job_id}"))'
traces = {}
for run in client.list_runs(project_name=project, filter=filter_query, is_root=True):
    trial_name = run.metadata.get('trial_name', str(run.id))
    traces[trial_name] = {'trace_id': str(run.id), 'status': run.status}

print(f"Found {len(traces)} traces")

# Step 2: Download each trace
for name, info in traces.items():
    output_path = f"traces/{name}.json"
    subprocess.run([
        "langsmith-fetch", "trace", info['trace_id'],
        "--format", "raw", "--include-metadata",
        "--file", output_path
    ])
    time.sleep(1)  # Rate limiting
```

### Workflow: Quick Analysis of Recent Failures

```bash
# Download recent error traces
langsmith-fetch traces ./errors --limit 20 --include-metadata

# Find common patterns
grep -l '"status": "error"' errors/*.json | while read f; do
  echo "=== $f ==="
  cat "$f" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Status: {d[\"metadata\"][\"status\"]}')
print(f'Messages: {len(d[\"messages\"])}')
print(f'Last message: {str(d[\"messages\"][-1])[: 200]}...')
"
done
```

### Workflow: Compare Passing vs Failing Traces

```python
import json
from pathlib import Path

passed = list(Path("traces/passed").glob("*.json"))
failed = list(Path("traces/failed").glob("*.json"))

def avg_messages(files):
    total = sum(len(json.load(open(f))["messages"]) for f in files)
    return total / len(files) if files else 0

def avg_tokens(files):
    total = sum(
        json.load(open(f))["metadata"]["token_usage"]["total_tokens"] or 0
        for f in files
    )
    return total / len(files) if files else 0

print(f"Passed: {len(passed)} traces, avg {avg_messages(passed):.0f} messages, {avg_tokens(passed):.0f} tokens")
print(f"Failed: {len(failed)} traces, avg {avg_messages(failed):.0f} messages, {avg_tokens(failed):.0f} tokens")
```

---

## Part 6: Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| "LANGSMITH_API_KEY not found" | Env var not set | `export LANGSMITH_API_KEY=lsv2_...` |
| Empty traces | Some errors occur before agent starts | Check for DaytonaError/infra errors |
| Rate limited | Too many requests | Add delay between fetches, reduce `--max-concurrent` |
| JSON parse errors | Wrong format flag | Use `--format raw` for clean JSON |
| "No traces found" | Wrong project or filter | Verify project name, check filter syntax |
| Missing messages | Trace still pending | Check `status` field, wait for completion |

---

## Quick Reference

```bash
# Install
pip install langsmith-fetch langsmith

# Setup
export LANGSMITH_API_KEY=lsv2_...

# Fetch single trace
langsmith-fetch trace <id> --format raw --include-metadata --file out.json

# Bulk fetch recent
langsmith-fetch traces ./dir --limit 50 --include-metadata

# Filter by metadata (Python)
from langsmith import Client
client = Client()
for run in client.list_runs(project_name="X", filter='eq(metadata_key, "job_id")...', is_root=True):
    print(run.id)
```
