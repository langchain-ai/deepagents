#!/usr/bin/env bash
# Curated reasoning eval set (23 tests) for cross-model comparison + hill-climbing.
# Audited 2026-06 — every test has sound, uniquely-correct gold (DB + communication);
# see project memory `tau2-airline-test-audit`. Run for any model:
#   ./run_reasoning_eval_set.sh openrouter:minimax/minimax-m3
#   ./run_reasoning_eval_set.sh anthropic:claude-opus-4-8
#   ./run_reasoning_eval_set.sh openai:gpt-5.5            [trials override: 2nd arg]
set -uo pipefail
cd "$(dirname "$0")"            # libs/evals

MODEL="${1:-openrouter:minimax/minimax-m3}"
TRIALS="${2:-3}"
SAFE=$(printf '%s' "$MODEL" | tr '/:' '__')
LOG="/tmp/reasoning_eval_${SAFE}.log"
mkdir -p trial_runs
: > "$LOG"

# Load provider + LangSmith keys from .env.
set -a; . ./.env; set +a
export LANGSMITH_TRACING=true

# 23-test curated set. Note the substring guard `(task_4 and not task_44 and not
# task_47)`: pytest -k is substring-based, and `task_4` alone would also catch
# task_44 (excluded, broken gold) and task_47 (selected separately). `task_23`
# intentionally matches BOTH task_23 (variant A: "minimize Mastercard") and
# task_23b (variant B: "minimize total credit spend").
KEXPR='(test_tau2_airline and (task_8 or task_11 or task_12 or task_15 or task_18 or task_20 or task_21 or task_23 or task_24 or task_25 or task_28 or task_29 or task_32 or task_47 or task_48 or task_49 or (task_4 and not task_44 and not task_47))) or test_two_tools_user_name_from_current_id or test_exact_word_count_and_z_starts or test_three_tools_find_user_then_city or test_five_steps_current_user_food_names_and_calories or (test_bfcl_v3 and multi_turn_miss_func_55)'

echo "=== reasoning eval set: $MODEL x $TRIALS trials, started $(date) ===" | tee -a "$LOG"

caffeinate -dimsu uv run --group test python scripts/run_trials.py \
  --model "$MODEL" \
  --trials "$TRIALS" \
  --out-dir "trial_runs/reasoning_${SAFE}" \
  -- -k "$KEXPR" 2>&1 | tee -a "$LOG"

code=${PIPESTATUS[0]}
echo "=== reasoning eval set finished $(date) exit=$code ===" | tee -a "$LOG"
echo "$code" > "/tmp/reasoning_eval_${SAFE}.exitcode"
