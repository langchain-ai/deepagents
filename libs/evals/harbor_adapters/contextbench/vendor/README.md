# Context-Bench source data

This directory vendors the Context-Bench filesystem-cloud corpus from Letta's
[`letta-evals`](https://github.com/letta-ai/letta-evals) repository at its `main`
branch:

- `letta-leaderboard/filesystem-agent/datasets/filesystem_cloud.jsonl`
- `letta-leaderboard/filesystem-agent/files/*.txt`
- `letta-leaderboard/filesystem-agent/rubric.txt` (the grading rubric used by
  the upstream `model_judge`; reproduced verbatim so our verifier scores the
  same way — see `../adapter.py` and `../templates/judge.py`)

The source repository is licensed under Apache-2.0. Its unmodified `LICENSE`
file is included alongside this attribution. The upstream repository does not
provide a `NOTICE` file.
