#!/bin/sh
set -eu
# Faithful Letta model_judge grader; writes /logs/verifier/reward.txt itself.
python3 /tests/judge.py
