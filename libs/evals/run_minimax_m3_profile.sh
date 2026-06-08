#!/usr/bin/env bash
# Runs the MiniMax M3 PROFILED eval config locally (kept awake via caffeinate).
#
# This run picks up the built-in MiniMax HarnessProfile automatically (model key
# `openrouter:minimax/minimax-m3` resolves to it), which:
#   - removes the `write_todos` tool (excludes TodoListMiddleware), and
#   - appends the behavioral suffix (completing_state_changes / find_a_permitted_path / report_back).
#
# Scope: full non-memory suite, MINUS the circular TODO-tool tests that only
# measure whether `write_todos` exists/fires (not task performance):
#   - tests/evals/test_langchain_middleware_todo.py  (whole module; builds with
#     create_agent + TodoListMiddleware directly, bypasses the harness)
#   - test_todos.py::test_write_todos_sequential_updates_returns_text
#   - test_todos.py::test_write_todos_three_steps_returns_text
#
# Baseline to compare against: the prior no-profile run (Notion "Baseline
# Harnesses Evaluation", MiniMax M3 = 0.83, 105/127). For a same-commit baseline,
# re-run with `_minimax.register()` commented out in
# libs/deepagents/deepagents/profiles/_builtin_profiles.py.
set -uo pipefail
cd "$(dirname "$0")"            # libs/evals

LOG=/tmp/minimax_m3_profile.log
REPORT=trial_runs/minimax_m3_profile.json
mkdir -p trial_runs
: > "$LOG"

# Load provider + LangSmith keys from .env into the environment.
set -a; . ./.env; set +a
export LANGSMITH_TRACING=true

echo "=== minimax m3 profiled run started $(date) ===" | tee -a "$LOG"

# caffeinate -dimsu holds power assertions for the lifetime of the eval process.
# Non-memory scope via --eval-category-exclude; circular TODO tests dropped via
# --ignore + -k after the `--` pytest passthrough.
caffeinate -dimsu uv run deepagents-evals run \
  --model openrouter:minimax/minimax-m3 \
  --eval-category-exclude memory \
  --report "$REPORT" \
  -- \
  --ignore=tests/evals/test_langchain_middleware_todo.py \
  -k "not (test_write_todos_sequential_updates_returns_text or test_write_todos_three_steps_returns_text)" \
  2>&1 | tee -a "$LOG"

code=${PIPESTATUS[0]}
echo "=== minimax m3 profiled run finished $(date) exit=$code ===" | tee -a "$LOG"
echo "$code" > /tmp/minimax_m3_profile.exitcode
