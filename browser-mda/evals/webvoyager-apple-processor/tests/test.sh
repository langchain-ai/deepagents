#!/bin/bash
# Grade the agent's final answer by keyword, deterministically.
#
# The agent's transcript is written by the MDA Harbor adapter under /logs/agent/.
# A keyword check is deliberate: an LLM judge re-introduces the grader noise that
# flipped tasks between identical result sets in earlier runs, and this question
# has a single factual answer.
set -uo pipefail
mkdir -p /logs/verifier

ANSWER=""
for f in /logs/agent/result.json /logs/agent/summary.json; do
  [ -f "$f" ] && ANSWER="${ANSWER} $(cat "$f")"
done

if [ -z "${ANSWER// }" ]; then
  echo '{"reward": 0, "found_answer": 0, "reason_no_output": 1}' > /logs/verifier/reward.json
  exit 0
fi

# The current Apple TV 4K ships an A15 Bionic. Accept the chip name in any casing
# and tolerate the "Bionic" suffix being absent.
if printf '%s' "$ANSWER" | grep -qiE 'a15[[:space:]]*(bionic)?'; then
  echo '{"reward": 1, "found_answer": 1, "correct_chip": 1}' > /logs/verifier/reward.json
else
  echo '{"reward": 0, "found_answer": 1, "correct_chip": 0}' > /logs/verifier/reward.json
fi
