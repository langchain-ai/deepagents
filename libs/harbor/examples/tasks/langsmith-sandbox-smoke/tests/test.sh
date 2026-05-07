#!/usr/bin/env bash
set -euo pipefail

expected="langsmith sandbox ok"
actual="$(cat /app/answer.txt 2>/dev/null || true)"

if [ "$actual" = "$expected" ]; then
  echo "1.0" > /logs/verifier/reward.txt
  exit 0
fi

echo "Expected '$expected' but got '$actual'" >&2
echo "0.0" > /logs/verifier/reward.txt
exit 1
