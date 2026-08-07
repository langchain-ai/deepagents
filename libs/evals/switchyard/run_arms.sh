#!/usr/bin/env bash
# Run the three benchmark arms back to back, one Switchyard config each.
#
#   ./run_arms.sh [arm ...]      # default: glm opus escalation
#
# Sequential on purpose: /v1/stats is a process-wide counter, so a reset-run-
# snapshot cycle only attributes cleanly when one arm owns the server at a time.
# pytest is also serial here (no -n), so there is no in-arm concurrency either.
#
# Per arm: swap the container, zero the counters, run the 145 non-memory tests,
# snapshot. Writes runs/<arm>.{json,log}. An arm that fails is logged and the
# next one still runs — a dead arm should not cost you the other two.
set -uo pipefail

# Provider keys live in the shell profile, not the session env, so a detached
# run must pull them in explicitly or the container starts with no credentials.
# shellcheck source=/dev/null
source ~/.zshrc 2>/dev/null || true

cd "$(dirname "$0")"
SW="$PWD"
EVALS="$(cd .. && pwd)"
RUNS="$SW/runs"
mkdir -p "$RUNS"

ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(glm opus escalation)

for arm in "${ARMS[@]}"; do
  cfg="$SW/routes-${arm}.toml"
  [ -f "$cfg" ] || { echo "!! no config $cfg, skipping"; continue; }

  echo "=============================================================="
  echo "ARM: $arm   ($(date -u +%H:%M:%SZ))"
  echo "=============================================================="

  docker rm -f sy-arm >/dev/null 2>&1
  docker run -d --name sy-arm -p 4000:4000 \
    -v "$cfg:/c.toml:ro" \
    -e BASETEN_API_KEY -e ANTHROPIC_API_KEY -e GOOGLE_API_KEY -e NVIDIA_API_KEY \
    switchyard-server:local --config /c.toml --host 0.0.0.0 --port 4000 >/dev/null

  ready=""
  for _ in $(seq 1 30); do
    curl -sf http://localhost:4000/health >/dev/null 2>&1 && { ready=1; break; }
    sleep 1
  done
  [ -n "$ready" ] || { echo "!! server never became healthy for $arm"; docker logs sy-arm 2>&1 | tail -20; continue; }

  python3 "$SW/collect_stats.py" reset

  # Abort early on an unhealthy arm. Healthy runs sit at 0.2-0.7% upstream
  # errors; two Nano-heavy arms have hit 9-35% from NVIDIA-side transport
  # failures and produced plausible-looking but invalid accuracy (whole
  # categories zeroed out). Without this the damage is only visible after a
  # full run. Waits for 60 requests so a couple of early blips can't trip it.
  ( while pgrep -f "pytest tests/evals" >/dev/null 2>&1; do
      sleep 60
      s=$(curl -sf http://localhost:4000/v1/stats 2>/dev/null) || continue
      read -r tot err <<<"$(printf '%s' "$s" | python3 -c \
        'import json,sys; d=json.load(sys.stdin); print(d.get("total_requests",0), d.get("total_errors",0))' 2>/dev/null)"
      [ -z "${tot:-}" ] && continue
      [ "$tot" -lt 60 ] && continue
      if [ $(( err * 100 / tot )) -ge 5 ]; then
        echo "!! ERROR RATE ${err}/${tot} >= 5% — killing arm ${arm}, results would be invalid"
        pkill -f "pytest tests/evals"
        break
      fi
    done ) &
  guard_pid=$!

  # Optional: apply a model's shipped harness profile to the routed run.
  # Profiles resolve by model spec, and a routed run presents as the route id,
  # so without this the model's accommodations are silently absent.
  #
  # The `${a[@]+"${a[@]}"}` form is required: macOS ships bash 3.2, where
  # expanding an empty array under `set -u` is an unbound-variable error rather
  # than an empty expansion.
  profile_args=()
  [ -n "${HARNESS_PROFILE:-}" ] && profile_args=(--harness-profile "$HARNESS_PROFILE")

  # Run from libs/evals so `tests.evals` imports resolve.
  ( cd "$EVALS" && LANGSMITH_TEST_SUITE="switchyard-${arm}" \
      uv run --group test pytest tests/evals \
        -v --tb=line \
        --model openai:switchyard \
        --base-url http://localhost:4000/v1 \
        ${profile_args[@]+"${profile_args[@]}"} \
        --eval-category-exclude memory ) 2>&1 | tee "$RUNS/${arm}.log" | tail -20

  kill "$guard_pid" 2>/dev/null
  python3 "$SW/collect_stats.py" snapshot "$RUNS/${arm}.json"
  echo "arm $arm done ($(date -u +%H:%M:%SZ))"
done

docker rm -f sy-arm >/dev/null 2>&1

echo
echo "=============================================================="
echo "ALL ARMS COMPLETE"
echo "=============================================================="
# shellcheck disable=SC2012
( cd "$EVALS" && uv run --group test python switchyard/collect_stats.py report "$RUNS"/*.json )
echo
echo "accuracy per arm:"
for arm in "${ARMS[@]}"; do
  line=$(grep -E '^results:|^correctness:' "$RUNS/${arm}.log" 2>/dev/null | tr '\n' ' ')
  printf '  %-11s %s\n' "$arm" "${line:-no log}"
done
