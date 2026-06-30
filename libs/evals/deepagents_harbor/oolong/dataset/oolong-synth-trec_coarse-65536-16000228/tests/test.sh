#!/usr/bin/env bash
set -uo pipefail
mkdir -p /logs/verifier
# score.py writes reward.json on every path; this is a last-resort fallback
# so the verifier never lacks a reward file.
if ! python3 /tests/score.py; then
  echo '{"score": 0.0}' > /logs/verifier/reward.json
fi
