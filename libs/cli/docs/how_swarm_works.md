# How Swarm Works

Swarm is a batch task execution system for deepagents-cli that allows you to run multiple tasks in parallel using subagents. It supports two modes:

1. **Task Execution** (`swarm_execute`) - Run explicit task definitions from a JSONL/CSV file
2. **CSV Enrichment** (`swarm_enrich`) - Fill in empty columns in a CSV using parallel research

---

## Quick Start

### Task Execution
```bash
# Create a task file
cat > tasks.jsonl << 'EOF'
{"id": "1", "description": "Summarize the README.md file"}
{"id": "2", "description": "List all Python files in src/"}
{"id": "3", "description": "Create a summary report"}
EOF

# Run via slash command
/swarm tasks.jsonl --num-parallel 5
```

### CSV Enrichment
```bash
# Create a CSV with empty columns to fill
cat > companies.csv << 'EOF'
ticker,company,ceo,headquarters,founded
AAPL,Apple,,,
MSFT,Microsoft,,,
NVDA,NVIDIA,,,
EOF

# Enrich via slash command
/swarm --enrich companies.csv
```

---

## Mode 1: Task Execution

Use this when you have **explicit tasks** you want to run in parallel.

### Input Formats

#### JSONL Format
```jsonl
{"id": "1", "description": "Analyze sales data for Q1"}
{"id": "2", "description": "Analyze sales data for Q2"}
{"id": "3", "description": "Compare Q1 and Q2 results"}
```

#### CSV Format
```csv
id,description,type
1,Analyze sales data for Q1,,
2,Analyze sales data for Q2,,
3,Compare Q1 and Q2 results,
```

### Task Schema

| Field | Required | Description |
|-------|----------|-------------|
| `id` | No | Unique identifier for the task (auto-generated if omitted) |
| `description` | Yes | Instructions for the subagent (`task`/`prompt` aliases supported) |
| `type` | No | Subagent type (default: "general-purpose") |
| `metadata` | No | User-defined data passed through to results |

### Invocation

#### Slash Command
```bash
/swarm <file> [--num-parallel N] [--concurrency N] [--output-dir DIR]

# Examples
/swarm tasks.jsonl
/swarm tasks.csv --num-parallel 20
/swarm analysis.jsonl --output-dir ./results/
```

#### Natural Language
```
Run the tasks in tasks.jsonl with 10 parallel workers
Execute batch tasks from my_tasks.csv
```

#### Direct Tool Call (programmatic)
The agent can call `swarm_execute(source="tasks.jsonl", num_parallel=10)` directly.

### Output

Results are written to a directory (default: `./batch_results/<timestamp>_<run_id>/`):

```
./batch_results/2025-02-01_143022_a1b2c3d4/
├── summary.json      # Execution statistics
├── results.jsonl     # All task outputs
└── failures.jsonl    # Failed tasks only
```

#### summary.json
```json
{
  "run_id": "a1b2c3d4",
  "started_at": "2026-02-07T12:34:56",
  "total": 50,
  "succeeded": 47,
  "failed": 2,
  "skipped": 1,
  "duration_seconds": 45.2,
  "concurrency": 10,
  "results_path": "./batch_results/.../results.jsonl",
  "failures_path": "./batch_results/.../failures.jsonl"
}
```

#### results.jsonl
```jsonl
{"task_id": "1", "status": "success", "output": "Q1 analysis: ...", "duration_ms": 2340}
{"task_id": "2", "status": "success", "output": "Q2 analysis: ...", "duration_ms": 1890}
{"task_id": "3", "status": "success", "output": "Comparison: ...", "duration_ms": 3200}
```

#### failures.jsonl
```jsonl
{"task_id": "5", "status": "failed", "error": "TimeoutError", "message": "Task timed out after 300s"}
```

---

## Mode 2: CSV Enrichment

Use this when you have a **CSV with missing data** that you want to fill in via research.

### How It Works

1. **Input**: CSV where some columns are filled (context) and some are empty (to research)
2. **Processing**: Each row becomes a task asking the subagent to fill empty columns
3. **Output**: Enriched CSV with the missing values filled in

### Input Format

```csv
ticker,company,ceo,cto,market_cap
AAPL,Apple,,,
MSFT,Microsoft,,,
NVDA,NVIDIA,,,
```

- **Filled columns** (`ticker`, `company`) → provide context
- **Empty columns** (`ceo`, `cto`, `market_cap`) → will be researched

### Auto-Generated Task

For each row, a task is automatically created:

```
Research and fill in the missing information.

**Known information:**
- ticker: AAPL
- company: Apple

**Columns to fill:** "ceo", "cto", "market_cap"

**Instructions:**
1. Use the known information as context to research the missing values
2. Return ONLY a valid JSON object with the missing column names as keys
3. If you cannot find a value, use null or "N/A"

**Required output format (JSON only):**
{
  "ceo": "...",
  "cto": "...",
  "market_cap": "..."
}
```

### Invocation

#### Slash Command
```bash
/swarm --enrich <file.csv> [--num-parallel N] [--concurrency N] [--output PATH] [--id-column COL]

# Examples
/swarm --enrich companies.csv
/swarm --enrich companies.csv --num-parallel 5
/swarm --enrich data.csv --output enriched_data.csv
/swarm --enrich companies.csv --id-column ticker
```

#### Natural Language
```
Fill in the missing data in companies.csv
Enrich the CSV file products.csv with the missing information
Research and complete the empty fields in my_data.csv
```

### Options

| Option | Description |
|--------|-------------|
| `--num-parallel N` | Number of parallel workers (default: 10, preferred flag) |
| `--concurrency N` | Backward-compatible alias for `--num-parallel` |
| `--output PATH` | Output file path (default: `<input>_enriched.csv`) |
| `--id-column COL` | Column to use as task ID for tracking (default: row number) |

### Output

#### Enriched CSV (main output)
```csv
ticker,company,ceo,cto,market_cap
AAPL,Apple,Tim Cook,Craig Federighi,$3.5T
MSFT,Microsoft,Satya Nadella,Kevin Scott,$3.1T
NVDA,NVIDIA,Jensen Huang,N/A,$2.8T
```

#### Execution Files (temporary)
Same structure as task execution, stored in `./batch_results/<timestamp>/`

---

## Comparison

| Aspect | Task Execution | CSV Enrichment |
|--------|----------------|----------------|
| **Use case** | Run explicit tasks | Fill in missing data |
| **Input** | JSONL or CSV with task definitions | CSV with empty columns |
| **Task definition** | You write the `description` | Auto-generated from schema |
| **Dependencies** | Not supported (all tasks run in parallel) | All tasks run in parallel |
| **Primary output** | `results.jsonl` in output dir | Enriched CSV file |
| **Slash command** | `/swarm file.jsonl` | `/swarm --enrich file.csv` |

---

## Examples

### Example 1: Parallel File Analysis

Analyze multiple files in parallel:

```jsonl
{"id": "readme", "description": "Summarize README.md in 2-3 sentences"}
{"id": "setup", "description": "List the dependencies from setup.py"}
{"id": "main", "description": "Describe what main.py does"}
{"id": "report", "description": "Create a project overview combining the above"}
```

```bash
/swarm analysis_tasks.jsonl --num-parallel 3
```

### Example 2: Research Multiple Topics

```jsonl
{"id": "ai", "description": "Research recent advances in AI (2024-2025), return 3 bullet points"}
{"id": "quantum", "description": "Research quantum computing progress (2024-2025), return 3 bullet points"}
{"id": "biotech", "description": "Research biotech breakthroughs (2024-2025), return 3 bullet points"}
```

### Example 3: Company Research (Enrichment)

```csv
ticker,company,ceo,founded,headquarters,employee_count
AAPL,Apple,,,,
GOOGL,Alphabet,,,,
AMZN,Amazon,,,,
META,Meta,,,,
TSLA,Tesla,,,,
```

```bash
/swarm --enrich companies.csv --num-parallel 5 --id-column ticker
```

### Example 4: Product Data Completion

```csv
product_name,brand,category,price_usd,weight_kg
iPhone 15 Pro,Apple,,,
Galaxy S24,Samsung,,,
Pixel 8,Google,,,
```

```bash
/swarm --enrich products.csv
```

---

## Error Handling

### Task Failures

When a task fails:
- Error is recorded in `results.jsonl` and `failures.jsonl`
- Other tasks continue running independently

### Enrichment Failures

When enrichment fails for a row:
- That row's empty columns remain empty in the output
- Other rows are still processed
- Check `failures.jsonl` for details

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `Task file not found` | File path is wrong | Check the file path |
| `blocked_by is not supported` | Task file includes dependency fields | Remove `blocked_by` from task definitions |
| `Duplicate task ID` | Two tasks have same ID | Use unique IDs |
| `TimeoutError` | Task took too long | Increase timeout or simplify task |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                              │
│  /swarm tasks.jsonl    OR    "run tasks in tasks.jsonl"        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SLASH COMMAND (app.py)                     │
│  Parses command → Builds prompt → Sends to agent                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         AGENT                                   │
│  Receives prompt → Calls swarm_execute or swarm_enrich tool     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SWARM MIDDLEWARE                             │
│  ┌─────────────┐    ┌─────────────┐                            │
│  │ parser.py   │    │enrichment.py│  ← Parse input files       │
│  └─────────────┘    └─────────────┘                            │
│          │                 │                                    │
│          ▼                 ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    graph.py                              │   │
│  │  Validate task IDs and scheduling metadata               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   executor.py                            │   │
│  │  Async worker pool, parallel scheduling                  │   │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐                    │   │
│  │  │Worker 1 │ │Worker 2 │ │Worker N │  ← Parallel exec   │   │
│  │  └────┬────┘ └────┬────┘ └────┬────┘                    │   │
│  │       │           │           │                          │   │
│  │       ▼           ▼           ▼                          │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │              SUBAGENTS                           │    │   │
│  │  │  Each task runs in isolated subagent context     │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT                                  │
│  Task Execution: ./batch_results/<timestamp>/                   │
│  CSV Enrichment: <input>_enriched.csv                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration

### Default Settings

| Setting | Default | Description |
|---------|---------|-------------|
| Concurrency | 10 | Max parallel workers |
| Max Concurrency | 50 | Hard limit on workers |
| Timeout | 300s | Per-task timeout |
| Output Directory | `./batch_results/<timestamp>_<run_id>/` | Where results are written |

### Overriding Defaults

Via slash command:
```bash
/swarm tasks.jsonl --num-parallel 20
```

Via natural language:
```
Run tasks.jsonl with 25 parallel workers
```
