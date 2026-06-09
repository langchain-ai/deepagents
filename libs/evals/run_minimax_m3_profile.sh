#!/usr/bin/env bash
# Runs the MiniMax M3 PROFILED eval config locally (kept awake via caffeinate).
#
# This run picks up the built-in MiniMax HarnessProfile automatically (model key
# `openrouter:minimax/minimax-m3` resolves to it). The profile is suffix-only:
# it appends a behavioral suffix (completing_state_changes / find_a_permitted_path
# / report_back / manage_context) and does NOT change the tool set — write_todos
# is retained.
#
# Scope: full non-memory suite. For a same-commit no-profile baseline, re-run with
# `_minimax.register()` commented out in
# libs/deepagents/deepagents/profiles/_builtin_profiles.py (or run on the parent
# commit). For repeated trials, prefer the `evals_trials` GitHub workflow.
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
caffeinate -dimsu uv run deepagents-evals run \
  --model openrouter:minimax/minimax-m3 \
  --eval-category-exclude memory \
  --report "$REPORT" \
  2>&1 | tee -a "$LOG"

code=${PIPESTATUS[0]}
echo "=== minimax m3 profiled run finished $(date) exit=$code ===" | tee -a "$LOG"
echo "$code" > /tmp/minimax_m3_profile.exitcode
