#!/usr/bin/env bash
# Run unified-evals autonomous tasks through a Switchyard sidecar in a remote
# LangSmith sandbox. This is a local launcher, not a local Docker execution.
#
#   ./run_harbor_arms.sh --image REGISTRY/IMAGE@sha256:DIGEST [options] [arm ...]
#
# Defaults to one task, one rollout, and the nano arm. Increase --n-tasks only
# after the smoke produces a reward and a per-trial switchyard-stats.json.
set -euo pipefail

# Provider credentials live in the developer shell and are forwarded by name;
# this script never prints or serializes their values.
# shellcheck source=/dev/null
source ~/.zshrc 2>/dev/null || true

cd "$(dirname "$0")"
SW="$PWD"
EVALS="$(cd .. && pwd)"
REPO="$(cd ../../.. && pwd)"
COMPOSE="$SW/compose/switchyard.yaml"

IMAGE="${SWITCHYARD_IMAGE:-}"
ROLLOUTS=1
N_TASKS=1
CONCURRENCY=1
ARMS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --image) IMAGE="$2"; shift 2 ;;
    --rollouts) ROLLOUTS="$2"; shift 2 ;;
    --n-tasks) N_TASKS="$2"; shift 2 ;;
    --concurrency) CONCURRENCY="$2"; shift 2 ;;
    *) ARMS+=("$1"); shift ;;
  esac
done
[ ${#ARMS[@]} -eq 0 ] && ARMS=(nano)

if ! [[ "$IMAGE" =~ ^[A-Za-z0-9.-]+(/[A-Za-z0-9._-]+)+@sha256:[0-9a-f]{64}$ ]]; then
  echo "error: --image must be a public digest-pinned image reference" >&2
  exit 2
fi
if ! [[ "$ROLLOUTS" =~ ^[1-9][0-9]*$ ]] || ! [[ "$N_TASKS" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --rollouts and --n-tasks must be positive integers" >&2
  exit 2
fi
if ! [[ "$CONCURRENCY" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: --concurrency must be a positive integer" >&2
  exit 2
fi

# The committed lite slice is the same task selection used by unified_evals.yml.
TASKS=$(cd "$REPO/.github/scripts/evals" && python3 -c "
from lite_tasks import include_tasks; print(include_tasks('autonomous'))")
[ -n "$TASKS" ] || { echo "error: could not resolve lite autonomous tasks" >&2; exit 1; }

include_args=()
count=0
for task in $TASKS; do
  [ "$count" -ge "$N_TASKS" ] && break
  include_args+=(--include-task-name "$task")
  count=$((count + 1))
done

echo "staging local agent dependencies..."
(cd "$EVALS" && make stage-harbor-local-deps >/dev/null)

export SWITCHYARD_IMAGE="$IMAGE"
for arm in "${ARMS[@]}"; do
  case "$arm" in
    glm) required_keys=(BASETEN_API_KEY) ;;
    opus) required_keys=(ANTHROPIC_API_KEY) ;;
    nano) required_keys=(NVIDIA_API_KEY) ;;
    escalation) required_keys=(ANTHROPIC_API_KEY BASETEN_API_KEY GOOGLE_API_KEY) ;;
    glm-nano) required_keys=(BASETEN_API_KEY GOOGLE_API_KEY NVIDIA_API_KEY) ;;
    opus-nano) required_keys=(ANTHROPIC_API_KEY GOOGLE_API_KEY NVIDIA_API_KEY) ;;
    *) echo "error: unknown Switchyard arm: $arm" >&2; exit 2 ;;
  esac

  missing=()
  for key in LANGSMITH_API_KEY OPENAI_API_KEY "${required_keys[@]}"; do
    [ -z "${!key:-}" ] && missing+=("$key")
  done
  if [ ${#missing[@]} -gt 0 ]; then
    echo "error: missing required variables: ${missing[*]}" >&2
    exit 1
  fi

  config="$SW/routes-${arm}.toml"
  [ -f "$config" ] || { echo "error: missing $config" >&2; exit 1; }
  experiment="switchyard-harbor-${arm}-$(date -u +%Y%m%d-%H%M%S)"
  jobs_dir="harbor-jobs/switchyard-${arm}"
  echo "arm=$arm tasks=$count rollouts=$ROLLOUTS experiment=$experiment"

  (
    cd "$EVALS"
    uv run --no-sync harbor run \
      --yes \
      --agent deepagents_harbor.switchyard_agent:SwitchyardLangGraph \
      --agent-kwarg project_path=deepagents_harbor/langgraph_project \
      --agent-kwarg config=langgraph.json \
      --agent-kwarg graph=bare \
      --agent-kwarg 'model_kwargs={"base_url":"http://switchyard:4000/v1","api_key":"switchyard","use_responses_api":false}' \
      --dataset harbor-index/harbor-index-1.0 \
      --model openai:switchyard \
      "${include_args[@]}" \
      -n "$CONCURRENCY" \
      --n-attempts "$ROLLOUTS" \
      --max-retries 0 \
      --env deepagents_harbor.switchyard_environment:SwitchyardLangSmithEnvironment \
      --extra-docker-compose "$COMPOSE" \
      --environment-kwarg "switchyard_config=$config" \
      --jobs-dir "$jobs_dir" \
      --agent-env 'LANGSMITH_API_KEY=${LANGSMITH_API_KEY}' \
      --agent-env 'LANGSMITH_TRACING=true' \
      --agent-env "LANGSMITH_PROJECT=$experiment" \
      --verifier-env 'OPENAI_API_KEY=${OPENAI_API_KEY}' \
      --verifier-env 'OPENAI_BASE_URL=https://api.openai.com/v1' \
      --verifier-env 'JUDGE_PROVIDER=openai' \
      --verifier-env 'JUDGE_MODELS=gpt-5.6-luna' \
      --verifier-env 'JUDGE_REPEATS=1' \
      --verifier-env 'JUDGE_CONCURRENCY=1' \
      --plugin langsmith \
      --plugin-kwarg dataset_name=harbor-index/harbor-index-1.0 \
      --plugin-kwarg experiment_name="$experiment"
  )

  latest_job=$(find "$EVALS/$jobs_dir" -mindepth 1 -maxdepth 1 -type d | sort | tail -1)
  python3 "$SW/collect_stats.py" validate "$latest_job"
  echo "Switchyard stats:"
  find "$latest_job" -name switchyard-stats.json -print
done
