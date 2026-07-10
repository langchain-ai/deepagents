#!/bin/sh
set -eu
answer=$(tr '[:upper:]' '[:lower:]' < /app/answer.txt | tr -cd '[:alnum:][:space:]')
expected=$(printf '%s' '$115,792.97' | tr -cd '[:alnum:][:space:]')
if [ "$answer" = "$expected" ]; then
  printf '1.0\n' > /logs/verifier/reward.txt
else
  printf '0.0\n' > /logs/verifier/reward.txt
fi
