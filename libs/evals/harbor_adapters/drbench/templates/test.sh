#!/bin/sh
set -eu
# Faithful DRBench insights_recall grader; writes /logs/verifier/reward.txt itself.
python3 /tests/judge.py
