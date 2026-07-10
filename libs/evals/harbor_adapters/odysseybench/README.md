# odysseybench adapter — DEFERRED TO v2

**Status:** vendored (banked), **not built for v1**. The `context-retrieval-evals` dataset v1
ships **Context-Bench only** (see `docs/superpowers/specs/2026-07-09-harbor-context-retrieval-evals-design.md`,
Decision 2026-07-10).

## Why deferred

`microsoft/OdysseyBench` ships **no `testbed/` office data** in the source tree — upstream builds an
empty testbed at runtime (`utils/env.py::_prepare_docker_testbed` just `mkdir`s `data/`,`emails/`,`calendar/`),
so office files (`score.xlsx`, `agreement.pdf`, …) are never committed. Reproducing the office-execution
dimension is the "heavy adapter" path the design defers. The only tractable framing — a *memory-retrieval
subset* (seed the chat history, verify with the deterministic `evaluate_contain`/`evaluate_file_exist`
checkers) — substantially overlaps Context-Bench's retrieval signal, so it was judged not worth a second
source for v1.

## What's here (banked for v2)

- `vendor/LICENSE`, `vendor/NOTICE.txt` — upstream MIT license + attribution.
- `vendor/evaluate.py` — upstream `utils/evaluate.py`, unmodified (deterministic checkers).
- `vendor/apps/` — the read-helper modules `evaluate.py` imports at load.
- `vendor/tasks/` — 14 selected subtask data pairs (chat history + subtask JSON), pre-validated so
  every checker keyword is derivable from the seeded history.

## To build in v2

Execute `docs/superpowers/plans/2026-07-10-odysseybench-dataset.md` (Tasks 2–8): adapter `adapter.py`
+ `main.py` + `task_template/verify.py`, emit the 14 task dirs, oracle-verify 1.0. Reconsider the task
selection to target the genuinely-additive conversational/temporal-disambiguation subtasks, likely
paired with the raw-length axis (LoCoBench).
