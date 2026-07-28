#!/bin/sh
set -eu
# Dual-judge LoHoSearch grader; writes /logs/verifier/reward.txt itself.
python3 /tests/judge.py
