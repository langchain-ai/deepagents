# Terminal Bench Trace Analysis Guide

This guide covers analyzing traces from Terminal Bench (Harbor) benchmark runs, including failure categorization, storage organization, and systematic analysis approaches.

## Understanding Terminal Bench Structure

### What is Terminal Bench?

Terminal Bench is a benchmark suite that evaluates coding agents on 89 real-world software engineering tasks. Each task is run multiple times (trials) to measure consistency.

### Key Terminology

| Term | Meaning |
|------|---------|
| **Task** | A specific problem (e.g., "crack-7z-hash", "build-pov-ray") |
| **Trial** | One attempt at a task, identified by `{task}__{random_id}` |
| **Job** | A complete benchmark run (all tasks × N trials) |
| **Reward** | Task outcome: `1.0` (pass) or `0.0` (fail) |
| **Error** | Exception during execution (different from task failure) |

### Data Sources

After a benchmark run, you have:

1. **`result.json`**: Aggregated results with rewards and error classifications
2. **Trial directories**: Local logs and outputs for each trial
3. **LangSmith traces**: Full conversation history for each trial

The trace contains the complete agent interaction—what it saw, what it did, and how it responded to errors.

---

## Failure Classification Framework

Failures fall into three major categories, each requiring different analysis:

### Category 1: Infrastructure Errors

These are NOT agent failures—the agent never had a fair chance.

| Error Type | What Happened | Analysis Action |
|------------|---------------|-----------------|
| `DaytonaError` | Sandbox failed to start | Exclude from agent analysis |
| `ProcessInterruptedError` | Run was manually stopped | Note as incomplete |
| `BadRequestError` | API error (malformed request) | Check if agent caused it |

**How to identify**: Check `exception_stats` in `result.json`

**Storage**: Keep in `errors/infrastructure/` but don't count toward agent performance

### Category 2: Resource Limit Errors

Agent ran but hit a hard limit. May indicate agent issues OR task difficulty.

| Error Type | What Happened | Questions to Ask |
|------------|---------------|------------------|
| `GraphRecursionError` | Hit step limit (e.g., 1000 steps) | Was agent stuck in a loop? Making progress? |
| `AgentTimeoutError` | Exceeded wall clock time | Was agent working or idle? |

**Key distinction**:
- **Stuck in loop**: Agent failure (repeated same failing action)
- **Slow but progressing**: May need higher limits, not agent fix

**Analysis approach**:
1. Check message count and variety
2. Look for repeated tool calls with same errors
3. Identify if agent tried different strategies

### Category 3: Task Failures (reward=0.0)

Agent completed but didn't solve the task. This is where most analysis value is.

**Sub-categories**:

#### 3a. Model/Logic Failures
The model made poor decisions or had incorrect understanding.

| Pattern | Description | Evidence in Trace |
|---------|-------------|-------------------|
| **Misunderstanding** | Misread or misinterpreted requirements | Early divergence from task spec |
| **Wrong strategy** | Chose ineffective approach | Committed to bad plan, didn't pivot |
| **Incomplete solution** | Solved part of problem | Missing edge cases, partial output |
| **Gave up early** | Stopped before exhausting options | Few attempts, premature "I can't" |
| **Verification skip** | Didn't test solution | No test runs, no output checks |
| **Format error** | Right answer, wrong format | Output exists but wrong location/format |

#### 3b. Tool/Execution Failures
The model's reasoning was sound but tool use failed.

| Pattern | Description | Evidence in Trace |
|---------|-------------|-------------------|
| **Tool misuse** | Wrong tool or wrong arguments | Syntax errors, wrong flags |
| **Environment mismatch** | Assumed wrong environment state | Missing dependencies, wrong paths |
| **Error non-recovery** | Didn't adapt to tool errors | Repeated same failing command |
| **Timeout in tool** | Tool call took too long | Long gaps, partial output |

#### 3c. Capability Gaps
Task requires something the agent fundamentally can't do.

| Pattern | Description | Evidence in Trace |
|---------|-------------|-------------------|
| **Knowledge gap** | Lacks domain knowledge | Wrong approaches for the domain |
| **Skill gap** | Can't perform required action | Unable to write valid code in language X |
| **Context limit** | Task state exceeds context | Forgetting earlier information |

---

## Storage Organization

### Recommended Directory Structure

```
{job_dir}/
├── result.json                    # Original benchmark results
├── langsmith_trace_mapping.json   # Trace ID → trial name mapping
├── analysis/
│   ├── summary.md                 # High-level findings
│   ├── findings.jsonl             # Structured findings per trace
│   └── by_category/
│       ├── model_logic.md         # Model/logic failure analysis
│       ├── tool_issues.md         # Tool/execution failure analysis
│       └── resource_limits.md     # Timeout/recursion analysis
└── langsmith-traces/
    ├── manifest.json              # Index of all downloaded traces
    └── by-outcome/
        ├── passed/                # reward=1.0
        │   └── {task}__{id}.json
        ├── failed/                # reward=0.0
        │   └── {task}__{id}.json
        └── errors/                # Exceptions
            ├── GraphRecursionError/
            ├── AgentTimeoutError/
            ├── DaytonaError/
            └── ProcessInterruptedError/
```

### Manifest File Structure

```json
{
  "job_id": "7f07b2d0-...",
  "project_name": "tb2-codex-reasoning",
  "model": "gpt-5.2-codex",
  "settings": {
    "reasoning_effort": "xhigh",
    "trials_per_task": 3,
    "max_steps": 1000
  },
  "download_completed": "2026-02-04T14:00:00Z",
  "counts": {
    "total_trials": 267,
    "total_traces": 265,
    "missing_traces": 2,
    "passed": 179,
    "failed": 57,
    "errors": 31
  },
  "missing_traces": [
    "hf-model-inference__L7RbKjt",
    "polyglot-rust-c__DLqF8dG"
  ],
  "traces": {
    "crack-7z-hash__Lu7oSkD": {
      "trace_id": "019c2754-dcf0-...",
      "task_name": "crack-7z-hash",
      "outcome": "passed",
      "reward": 1.0,
      "error_type": null,
      "file": "by-outcome/passed/crack-7z-hash__Lu7oSkD.json",
      "messages": 51,
      "tokens": 415897,
      "duration_ms": 209049
    }
  }
}
```

### Findings File Format (JSONL)

Each line in `findings.jsonl` represents analysis of one trace:

```json
{"trial_name": "task__id", "task_name": "task", "outcome": "failed", "category": "model_logic", "subcategory": "misunderstanding", "summary": "Agent wrote output to wrong file path", "root_cause": "Missed '/app' prefix in task description", "evidence": {"line": 45, "quote": "Writing to output.txt..."}, "severity": "high", "actionable": "Improve path parsing in task understanding"}
```

**Required fields**:
- `trial_name`, `task_name`, `outcome`
- `category`: `infrastructure`, `resource_limit`, `model_logic`, `tool_issue`, `capability_gap`
- `subcategory`: Specific pattern from tables above
- `summary`: One sentence
- `root_cause`: Why this happened
- `actionable`: What could fix it (if anything)

---

## Analysis Workflow

### Phase 1: Triage (Quick Pass)

Goal: Categorize all failures without deep analysis.

```python
import json
from pathlib import Path
from collections import defaultdict

# Load all traces
traces_dir = Path("langsmith-traces/by-outcome")
findings = []

# Quick categorization based on message count and status
for outcome_dir in ["failed", "errors/GraphRecursionError", "errors/AgentTimeoutError"]:
    for trace_file in (traces_dir / outcome_dir).rglob("*.json"):
        data = json.load(open(trace_file))
        msg_count = len(data.get("messages", []))

        finding = {
            "trial_name": trace_file.stem,
            "task_name": trace_file.stem.rsplit("__", 1)[0],
            "outcome": "error" if "errors" in str(trace_file) else "failed",
            "messages": msg_count,
            "needs_deep_dive": msg_count > 10,  # Had substantial interaction
        }

        # Quick heuristics
        if msg_count <= 3:
            finding["category"] = "likely_infrastructure"
        elif "GraphRecursionError" in str(trace_file):
            finding["category"] = "resource_limit"
            finding["subcategory"] = "recursion_limit"
        elif "AgentTimeoutError" in str(trace_file):
            finding["category"] = "resource_limit"
            finding["subcategory"] = "timeout"
        else:
            finding["category"] = "needs_analysis"

        findings.append(finding)

# Summary
by_category = defaultdict(list)
for f in findings:
    by_category[f["category"]].append(f["trial_name"])

for cat, trials in sorted(by_category.items()):
    print(f"{cat}: {len(trials)}")
```

### Phase 2: Deep Dive (Selected Traces)

Focus on traces that need analysis. For each:

1. **Read the task prompt** (first user message)
2. **Identify agent's approach** (first few assistant messages)
3. **Find the failure point** (where things went wrong)
4. **Classify the root cause** (use categories above)
5. **Document findings** (add to findings.jsonl)

**Analysis prompt for reviewing a trace**:

```
Analyze this Terminal Bench trace to understand why the agent failed.

Task: {task_name}
Outcome: {outcome}
Messages: {count}

Read the trace and determine:

1. TASK UNDERSTANDING
   - What did the task require?
   - Did the agent correctly understand the requirements?
   - Quote any misunderstandings.

2. APPROACH
   - What strategy did the agent choose?
   - Was this a reasonable approach?
   - Did the agent consider alternatives?

3. FAILURE POINT
   - At which message did things go wrong?
   - What was the immediate cause?
   - Could the agent have recovered?

4. ROOT CAUSE CLASSIFICATION
   Category: [infrastructure | resource_limit | model_logic | tool_issue | capability_gap]
   Subcategory: [specific pattern]

5. ACTIONABLE INSIGHT
   - What change would help? (to agent, prompt, tools, or nothing)
   - How confident are you? (high/medium/low)

Output your finding as a single JSON object.
```

### Phase 3: Aggregation

After analyzing traces, aggregate findings:

```python
import json
from collections import defaultdict

findings = [json.loads(line) for line in open("findings.jsonl")]

# By category
print("=== By Category ===")
by_cat = defaultdict(list)
for f in findings:
    by_cat[f["category"]].append(f)
for cat, items in sorted(by_cat.items(), key=lambda x: -len(x[1])):
    print(f"{cat}: {len(items)}")
    by_sub = defaultdict(int)
    for i in items:
        by_sub[i.get("subcategory", "unspecified")] += 1
    for sub, count in sorted(by_sub.items(), key=lambda x: -x[1]):
        print(f"  - {sub}: {count}")

# By task (find consistently failing tasks)
print("\n=== Consistently Failing Tasks ===")
by_task = defaultdict(list)
for f in findings:
    by_task[f["task_name"]].append(f)
for task, items in sorted(by_task.items()):
    if all(i["outcome"] != "passed" for i in items) and len(items) >= 3:
        categories = [i["category"] for i in items]
        print(f"{task}: {categories}")

# Actionable insights
print("\n=== Top Actionable Insights ===")
by_action = defaultdict(int)
for f in findings:
    if f.get("actionable"):
        by_action[f["actionable"]] += 1
for action, count in sorted(by_action.items(), key=lambda x: -x[1])[:10]:
    print(f"({count}x) {action}")
```

---

## Key Metrics to Report

### Overall Performance
- **Pass rate**: passed / (passed + failed + errors)
- **Clean pass rate**: passed / (passed + failed) [excluding infrastructure errors]
- **Error rate**: errors / total

### By Category
- % infrastructure errors (not agent's fault)
- % resource limits (may need config change)
- % model/logic failures (agent improvement opportunities)
- % tool issues (tooling improvement opportunities)
- % capability gaps (fundamental limitations)

### Consistency
- Tasks with 3/3 pass (reliable)
- Tasks with 0/3 pass (systematic failure)
- Tasks with 1-2/3 pass (inconsistent)

### Efficiency
- Average tokens per successful task
- Average messages per successful task
- Token efficiency: passes per million tokens

---

## Report Template

```markdown
# Terminal Bench Analysis: {model_name}

## Summary
- **Model**: {model}
- **Date**: {date}
- **Tasks**: {n_tasks} tasks × {n_trials} trials = {total} trials
- **Pass Rate**: {pass_rate}% ({passed}/{total_evaluated})

## Results by Outcome

| Outcome | Count | % |
|---------|-------|---|
| Passed | {n} | {%} |
| Failed | {n} | {%} |
| Errors | {n} | {%} |

## Error Breakdown

| Error Type | Count | Notes |
|------------|-------|-------|
| GraphRecursionError | {n} | Agent hit step limit |
| AgentTimeoutError | {n} | Agent hit time limit |
| DaytonaError | {n} | Infrastructure failure |

## Failure Analysis

### Model/Logic Failures ({n})
{breakdown by subcategory}

Top issues:
1. {issue}: {count} occurrences
2. ...

### Tool Issues ({n})
{breakdown}

### Capability Gaps ({n})
{breakdown}

## Consistently Failing Tasks

| Task | Trials | Failure Pattern |
|------|--------|-----------------|
| {task} | 0/3 | {pattern} |

## Recommendations

1. **High Impact**: {recommendation}
2. **Medium Impact**: {recommendation}
3. **Low Impact/Future**: {recommendation}

## Appendix: Detailed Findings

See `findings.jsonl` for per-trace analysis.
```

---

## Common Patterns and What They Mean

### Pattern: High Message Count + Failure
- Agent worked hard but couldn't solve it
- Look for: strategy changes, error recovery attempts
- May indicate: task difficulty, not agent failure

### Pattern: Low Message Count + Failure
- Agent gave up quickly
- Look for: early incorrect conclusions, missed information
- May indicate: poor task understanding, over-confidence

### Pattern: GraphRecursionError with Repeated Actions
- Agent stuck in loop
- Look for: same tool call → same error → same tool call
- Indicates: poor error handling, no backtracking

### Pattern: Success on 1-2/3 Trials
- Agent CAN solve it but not reliably
- Compare passing vs failing traces
- Look for: what differed in the successful run

### Pattern: All 3 Trials Fail Identically
- Systematic issue
- Agent consistently makes same mistake
- High value for improvement (fix once, help all)
