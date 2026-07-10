#!/bin/sh
set -eu
answer=$(tr '[:upper:]' '[:lower:]' < /app/answer.txt | tr -cd '[:alnum:][:space:]')
expected=$(printf '%s' '$130,196.23' | tr -cd '[:alnum:][:space:]')
if [ "$answer" = "$expected" ]; then
  printf '1.0\n' > /logs/verifier/reward.txt
else
  printf '0.0\n' > /logs/verifier/reward.txt
fi
