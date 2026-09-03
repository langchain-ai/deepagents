#!/bin/sh
set -eu
# Faithful DRBench grader: insights_recall, distractor_recall, factuality, and
# report_quality, plus our composite. Writes /logs/verifier/reward.json itself.
python3 /tests/judge.py
