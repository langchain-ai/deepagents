# Parallel Trace Analysis

This guide explains how to analyze many traces efficiently using multiple parallel agents, with all agents writing findings to a shared output file.

## Why Parallel Analysis?

When you have hundreds of traces:
- Sequential analysis takes hours
- Parallel agents can reduce this to minutes
- Shared output file enables aggregation

## Architecture

```
Orchestrator
    │
    ├─── Agent 1 ──► traces batch 1 ──┐
    ├─── Agent 2 ──► traces batch 2 ──┼──► findings.jsonl
    ├─── Agent 3 ──► traces batch 3 ──┤
    └─── Agent 4 ──► traces batch 4 ──┘
    │
    └─── Aggregate findings.jsonl ──► report
```

## Setup

### 1. Create Analysis Manifest

List all traces to analyze with metadata for routing:

```python
# create_manifest.py
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

print(f"Created manifest with {manifest['total']} traces")
```

### 2. Define Analysis Criteria

Create a file that tells agents exactly what to look for:

```markdown
# analysis_criteria.md

## Your Task

Analyze each assigned trace and produce a structured finding.

## What to Determine

For each trace:

1. **Outcome**: passed, failed, or error
2. **Category**: One of:
   - `infrastructure`: Error before agent could run
   - `resource_limit`: Hit step/time limit
   - `model_logic`: Agent reasoning failure
   - `tool_issue`: Tool use problem
   - `capability_gap`: Task beyond agent abilities

3. **Subcategory** (if model_logic or tool_issue):
   - model_logic: misunderstanding, wrong_strategy, incomplete, gave_up_early, no_verification, format_error
   - tool_issue: wrong_tool, wrong_args, error_non_recovery, environment_mismatch

4. **Summary**: One sentence describing what happened

5. **Root Cause**: Why this specific failure occurred

6. **Actionable Insight**: What change could help (or "none" if capability gap)

## Output Format

Write ONE line of JSON per trace to the shared findings file:

```json
{"trial_name": "...", "task_name": "...", "outcome": "...", "category": "...", "subcategory": "...", "summary": "...", "root_cause": "...", "actionable": "...", "confidence": "high|medium|low"}
```

## How to Analyze a Trace

1. Read the first message (task prompt)
2. Read agent's initial response (approach)
3. Scan for error messages or failures
4. Check final messages (did agent verify? give up?)
5. Classify using categories above
```

### 3. Create Batch Assignments

Split traces into batches for parallel processing:

```python
# create_batches.py
import json
import math

with open("analysis_manifest.json") as f:
    manifest = json.load(f)

traces = manifest["traces"]
n_agents = 4
batch_size = math.ceil(len(traces) / n_agents)

batches = []
for i in range(n_agents):
    start = i * batch_size
    end = min(start + batch_size, len(traces))
    batches.append({
        "agent_id": i + 1,
        "start_idx": start,
        "end_idx": end,
        "count": end - start,
        "traces": traces[start:end]
    })

for b in batches:
    with open(f"batch_{b['agent_id']}.json", "w") as f:
        json.dump(b, f, indent=2)
    print(f"Agent {b['agent_id']}: traces {b['start_idx']}-{b['end_idx']} ({b['count']} traces)")
```

## Execution

### Agent Prompt Template

Each parallel agent receives:

```
You are analyzing Terminal Bench traces to categorize failures.

**Your Assignment**
- Batch file: batch_{N}.json
- Analysis criteria: analysis_criteria.md
- Output: Append to findings.jsonl

**Instructions**
1. Read analysis_criteria.md to understand the classification framework
2. For each trace in your batch:
   a. Read the trace file
   b. Analyze according to criteria
   c. Append ONE JSON line to findings.jsonl
3. Report your count when done

**Output Format**
Each finding is ONE line of JSON (JSONL format):
{"trial_name": "...", "task_name": "...", "outcome": "...", "category": "...", "subcategory": "...", "summary": "...", "root_cause": "...", "actionable": "...", "confidence": "high|medium|low"}

**Important**
- Append to findings.jsonl, do not overwrite
- One JSON object per line
- Process ALL traces in your batch
- If a trace can't be read, log it and continue
```

### Writing to Shared File

Agents should append safely:

**Python approach**:
```python
import json
import fcntl  # File locking (Unix)

def write_finding(finding, output_file="findings.jsonl"):
    line = json.dumps(finding) + "\n"
    with open(output_file, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(line)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

**Shell approach** (simpler, usually safe for short writes):
```bash
echo '{"trial_name": "...", ...}' >> findings.jsonl
```

**Cross-platform safe approach**:
```python
import json
import os
import tempfile

def write_finding(finding, output_file="findings.jsonl"):
    line = json.dumps(finding) + "\n"
    # Write to temp file, then atomic append
    fd, temp_path = tempfile.mkstemp()
    try:
        os.write(fd, line.encode())
        os.close(fd)
        with open(temp_path) as src, open(output_file, "a") as dst:
            dst.write(src.read())
    finally:
        os.unlink(temp_path)
```

## Monitoring Progress

While agents run:

```bash
# Watch findings accumulate
watch -n 5 'wc -l findings.jsonl && tail -3 findings.jsonl'

# Check for valid JSON
tail -10 findings.jsonl | while read line; do
  echo "$line" | python3 -c "import json,sys; json.load(sys.stdin)" && echo "OK" || echo "INVALID"
done

# Quick category distribution
cat findings.jsonl | python3 -c "
import json, sys
from collections import Counter
cats = Counter()
for line in sys.stdin:
    if line.strip():
        cats[json.loads(line).get('category', '?')] += 1
for cat, n in cats.most_common():
    print(f'{cat}: {n}')
"
```

## Aggregation

After all agents complete:

```python
# aggregate.py
import json
from collections import defaultdict
from pathlib import Path

# Load all findings
findings = []
with open("findings.jsonl") as f:
    for line in f:
        if line.strip():
            try:
                findings.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: invalid JSON line")

print(f"Total findings: {len(findings)}")

# Check for duplicates
seen = set()
duplicates = []
for f in findings:
    if f["trial_name"] in seen:
        duplicates.append(f["trial_name"])
    seen.add(f["trial_name"])
if duplicates:
    print(f"Warning: {len(duplicates)} duplicates found")

# By category
print("\n=== By Category ===")
by_cat = defaultdict(list)
for f in findings:
    by_cat[f.get("category", "unknown")].append(f)
for cat, items in sorted(by_cat.items(), key=lambda x: -len(x[1])):
    print(f"{cat}: {len(items)}")

# By subcategory
print("\n=== By Subcategory ===")
by_sub = defaultdict(int)
for f in findings:
    key = f"{f.get('category', '?')}/{f.get('subcategory', '?')}"
    by_sub[key] += 1
for sub, count in sorted(by_sub.items(), key=lambda x: -x[1]):
    print(f"{sub}: {count}")

# Actionable insights
print("\n=== Actionable Insights ===")
by_action = defaultdict(int)
for f in findings:
    action = f.get("actionable", "")
    if action and action.lower() != "none":
        by_action[action] += 1
for action, count in sorted(by_action.items(), key=lambda x: -x[1])[:15]:
    print(f"({count}x) {action}")

# Save summary
summary = {
    "total_findings": len(findings),
    "by_category": {k: len(v) for k, v in by_cat.items()},
    "by_subcategory": dict(by_sub),
    "top_actions": dict(sorted(by_action.items(), key=lambda x: -x[1])[:20])
}
with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
```

## Best Practices

### Batch Sizing

| Total Traces | Recommended Agents | Batch Size |
|--------------|-------------------|------------|
| < 50 | 1-2 | 25-50 |
| 50-200 | 3-4 | 50-75 |
| 200-500 | 4-6 | 50-100 |
| > 500 | 6-8 | 75-100 |

Too small: Agent spawn overhead dominates
Too large: Loses parallelism benefit, risk of single agent failure

### Handling Failures

Agents should be resilient:

```python
def analyze_batch(batch_file, output_file):
    with open(batch_file) as f:
        batch = json.load(f)

    for trace_info in batch["traces"]:
        try:
            finding = analyze_trace(trace_info["file"])
            write_finding(finding, output_file)
        except Exception as e:
            # Log error but continue
            error_finding = {
                "trial_name": trace_info["trial_name"],
                "task_name": trace_info["task_name"],
                "outcome": "analysis_error",
                "category": "analysis_error",
                "summary": f"Could not analyze: {str(e)}"
            }
            write_finding(error_finding, output_file)
```

### Verification

After aggregation, verify completeness:

```python
# Check all traces were analyzed
with open("analysis_manifest.json") as f:
    manifest = json.load(f)
expected = {t["trial_name"] for t in manifest["traces"]}

with open("findings.jsonl") as f:
    analyzed = {json.loads(line)["trial_name"] for line in f if line.strip()}

missing = expected - analyzed
extra = analyzed - expected

print(f"Expected: {len(expected)}")
print(f"Analyzed: {len(analyzed)}")
print(f"Missing: {len(missing)}")
print(f"Extra: {len(extra)}")

if missing:
    print("Missing traces:", list(missing)[:10])
```

## Example: Quick Triage Run

For a fast first pass focusing on failures:

```
I need to triage 100 failed traces quickly.

Analysis criteria for this run:
- Focus only on traces in failed/ and errors/ directories
- For each trace, determine:
  1. Was this a model problem or infrastructure/tool problem?
  2. One sentence: what went wrong?
  3. Is this actionable? (yes/no)

Output to triage.jsonl with format:
{"trial": "...", "type": "model|tool|infra", "summary": "...", "actionable": true|false}

Spawn 4 agents, each handling ~25 traces.
```

## Example: Deep Dive on Category

After triage, deep dive on a specific category:

```
I want to understand all "model_logic/wrong_strategy" failures in detail.

For each trace in this category:
1. What strategy did the agent choose?
2. What would have been a better strategy?
3. Why might the agent have chosen poorly?
4. What prompt/context change could help?

Output detailed analysis to wrong_strategy_analysis.md
```
