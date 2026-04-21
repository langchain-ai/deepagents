#!/usr/bin/env bash
# Run Oolong dataset eval suites across one or more runners.
#
# One pytest invocation per (dataset, runner) pair. The suite name
# (``LANGSMITH_TEST_SUITE``) is pinned per dataset so every runner
# writes into the same LangSmith dataset — rows dedupe on task id so
# the per-runner experiments land side by side in the UI:
#
#   dataset: deepagents-py-oolong-trec-coarse
#     experiment: baseline-<STAMP>
#     experiment: rlm-<STAMP>
#     experiment: swarm-<STAMP>
#     experiment: shell-<STAMP>
#
# Mind the upsert semantics: if you later run with a larger
# ``OOLONG_MAX_PER_DATASET``, the dataset grows, and earlier smaller
# experiments only cover the subset they saw.
#
# Usage:
#   ./scripts/run_oolong.sh                          # all three datasets × all four runners
#   ./scripts/run_oolong.sh trec_coarse              # one dataset × all four runners
#   EVAL_RUNNERS="baseline rlm swarm" ./scripts/run_oolong.sh
#                                                    # skip shell (no sandbox cost)
#   EVAL_RUNNERS="baseline" ./scripts/run_oolong.sh trec_coarse
#                                                    # smoke: one runner, one dataset
#   OOLONG_MAX_PER_DATASET=3 ./scripts/run_oolong.sh
#                                                    # cap tasks for quick sweeps
#
# Requires ``LANGSMITH_TRACING=true`` + ``LANGSMITH_API_KEY`` in env.
# The ``shell`` runner additionally requires the ``deepagents-cli``
# LangSmith sandbox template to be available to the account.

set -euo pipefail

cd "$(dirname "$0")/.."

DATASETS=("${@:-trec_coarse multinli metaphors}")
# Word-split the single-arg default back into a proper array.
if [ "${#DATASETS[@]}" -eq 1 ]; then
    read -r -a DATASETS <<< "${DATASETS[0]}"
fi

# ``EVAL_RUNNERS`` is space-separated so it composes with shell env
# tools (``EVAL_RUNNERS="baseline rlm" ./run_oolong.sh``). Default to
# every runner we ship; user can narrow as needed.
read -r -a RUNNERS <<< "${EVAL_RUNNERS:-baseline rlm swarm shell}"

# Timestamp the experiment name so reruns create distinct experiments
# on the same dataset rather than overwriting the previous one.
# Seconds granularity is enough — you're not firing these in bursts.
# Stamp is shared across every (dataset, runner) invocation in one
# script run so you can find all of them with a single prefix search.
STAMP="$(date -u +%Y%m%d-%H%M%S)"

for ds in "${DATASETS[@]}"; do
    test_file="tests/evals/oolong/test_oolong_${ds}.py"
    suite="deepagents-py-oolong-${ds//_/-}"
    for runner in "${RUNNERS[@]}"; do
        # Include context_len in the experiment name when it's pinned
        # so a sweep over buckets stays legible in the LangSmith UI.
        # Leave it off otherwise — don't bake the literal "unset" in.
        if [ -n "${OOLONG_CONTEXT_LEN:-}" ]; then
            experiment="${runner}-ctx${OOLONG_CONTEXT_LEN}-${STAMP}"
        else
            experiment="${runner}-${STAMP}"
        fi

        echo ">>> Running ${ds} × ${runner} (suite=${suite}, experiment=${experiment})"
        # ``-k "${runner}"`` filters the parametrized (runner, task)
        # grid down to the current runner. The test id starts with
        # the runner name (see ``case_id`` in ``_test_body.py``), so
        # substring match is sufficient and stable.
        #
        # ``-s`` disables stdout capture so ``OOLONG_DEBUG_SCORING``
        # output (and anything else the test prints) surfaces live.
        # LangSmith trace URLs are also printed by the reporter and
        # only show under ``-s`` reliably.
        LANGSMITH_TEST_SUITE="${suite}" \
            LANGSMITH_EXPERIMENT="${experiment}" \
            uv run --group test pytest "${test_file}" -v -s -k "${runner}"
    done
done
