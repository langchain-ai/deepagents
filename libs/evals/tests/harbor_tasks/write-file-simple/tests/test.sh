#!/usr/bin/env bash
# Harbor verifier entrypoint for the `write-file-simple` task.
#
# Harbor copies this `tests/` directory to /tests in the container and runs
# this script after the agent finishes. It computes the deepagents trajectory +
# filesystem metrics and writes /logs/verifier/reward.json. All paths are fixed
# (no dynamic interpolation); the scoring logic lives in score.py.
set -euo pipefail

python3 /tests/score.py
