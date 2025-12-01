# Building DeepAgent Harnesses for Terminal Bench 2.0 with Harbor

## Setup

```bash
export LANGCHAIN_API_KEY="your-api-key"
export LANGCHAIN_PROJECT="your-project-name"
export LANGCHAIN_TRACING_V2=true
```

## Workflow

### 1. Create Dataset and Experiment

```bash
# Create dataset
python scripts/create_langsmith_dataset.py terminal-bench --version 2.0

# Create experiment (save the session ID output)
python scripts/create_langsmith_dataset.py terminal-bench --create-experiment
```

### 2. Run Harbor Jobs

Run your Harbor jobs with LangSmith tracing enabled.

### 3. Associate Traces with Experiment

```bash
# Dry run
./scripts/associate_job_traces.py \
  jobs/terminal-bench/2025-11-26__22-44-45 \
  <experiment_session_id> \
  --dry-run

# Run
./scripts/associate_job_traces.py \
  jobs/terminal-bench/2025-11-26__22-44-45 \
  <experiment_session_id>
```
