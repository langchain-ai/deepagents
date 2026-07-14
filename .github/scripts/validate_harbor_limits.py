"""Single-model + resource-limit gate for the Harbor evals workflow.

The Harbor workflow (`.github/workflows/harbor.yml`) evaluates exactly ONE model
per dispatch. This runs in the `prep` job, before the matrix fans out, to fail
fast when:

* the resolved model set is not a single model (a model group like `all`/`set0`
  or a comma-separated `models_override` resolves to more than one), or
* the per-run limits are exceeded: `n_shards <= shard_matrix.MAX_SHARDS` (200),
  `concurrency <= 4`. `n_shards` does not bound concurrency directly — it may
  be set as high as the task count (one task per shard) for dynamic dispatch.
  Instead, the derived `shard_parallel` (the pool the shards drain through) is
  what's bounded, alongside the per-shard concurrent-trial count: `shard_parallel
  * min(concurrency, rollouts) <= 40`.

This module owns the derivation of `shard_parallel` for the single-model case:
`shard_parallel = min(MAX_TASKS_PER_MODEL // min(concurrency, rollouts),
n_shards)`, mirroring `unified_prep.derive_pool` (pinned together by
`test_validate_harbor_limits.py`'s drift-guard test). `main()` emits it as
`shard_parallel=<value>` to `$GITHUB_OUTPUT` for `harbor.yml` to pass straight
through to the leaf workflow's `shard_parallel` input.

Mirrors `.github/scripts/shard_matrix.py` (import-by-path, stdlib only) so it is
exercised by `test_validate_harbor_limits.py` under CI's
`pytest .github/scripts/test_*.py`. Inputs arrive via env (never `${{ }}` shell
interpolation) and are int-parsed here.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shard_matrix  # noqa: E402  (shard_matrix.py in same dir)

MAX_CONCURRENCY = 4
# Per-model concurrent-sandbox budget the derived shard_parallel must respect
# alongside concurrency (mirrors unified_prep.MAX_TASKS_PER_MODEL).
MAX_TASKS_PER_MODEL = 40


class LimitError(Exception):
    """Raised for an unparseable numeric input (rendered as a GitHub ::error::)."""


def parse_positive(name: str, raw: str | None) -> int:
    """Parse a required positive integer, or raise LimitError.

    Rejects empty, non-digit, and `< 1` values (``"0"``, ``"-1"``, ``"abc"``, ``""``).
    """
    text = (raw or "").strip()
    if not text.isdigit() or int(text) < 1:
        msg = f"Invalid {name} (must be a positive integer): {text!r}"
        raise LimitError(msg)
    return int(text)


def derive_shard_parallel(concurrency: int, rollouts: int, n_shards: int) -> int:
    """Derive the shard pool size the run job's shards drain through.

    Mirrors `unified_prep.derive_pool`'s `max_parallel` for a single model:
    `per_shard = min(concurrency, rollouts)` is the peak concurrent trials one
    1-task shard ever runs; the pool saturates the per-model
    `MAX_TASKS_PER_MODEL` (40) budget without exceeding `n_shards` itself (no
    point pooling more than there are shards to drain).
    """
    per_shard = max(1, min(concurrency, rollouts))
    return max(1, min(MAX_TASKS_PER_MODEL // per_shard, n_shards))


def validate_limits(
    models: list, n_shards: int, concurrency: int, rollouts: int
) -> list[str]:
    """Return a list of human-readable violations (empty when the run is valid)."""
    errors: list[str] = []
    if len(models) != 1:
        names = [m.get("model") for m in models]
        errors.append(
            f"this workflow evaluates a single model, but the selection resolved "
            f"to {len(models)}: {names}. Pick exactly one model — model groups "
            "(all/set0/frontier/...) and comma-separated overrides are not accepted."
        )
    if n_shards > shard_matrix.MAX_SHARDS:
        errors.append(f"n_shards={n_shards} exceeds the cap of {shard_matrix.MAX_SHARDS}")
    if concurrency > MAX_CONCURRENCY:
        errors.append(f"concurrency={concurrency} exceeds the cap of {MAX_CONCURRENCY}")

    # Defensive, not user-facing: derive_shard_parallel's floor division makes
    # this true by construction (never trips). Uses per_shard — the peak
    # concurrent trials one 1-task shard actually runs, min(concurrency,
    # rollouts) — rather than raw concurrency, since a shard with fewer
    # rollouts than its concurrency setting can't use the extra slots (same
    # per_shard substitution unified_prep.derive_pool's docstring describes
    # loosely as "concurrency"). An explicit raise (never a bare `assert`,
    # which `python -O` strips) so a formula regression fails loudly instead
    # of silently over-provisioning sandboxes.
    shard_parallel = derive_shard_parallel(concurrency, rollouts, n_shards)
    per_shard = max(1, min(concurrency, rollouts))
    if shard_parallel * per_shard > MAX_TASKS_PER_MODEL:
        msg = (
            f"internal error: derived shard_parallel={shard_parallel} * "
            f"per_shard={per_shard} exceeds {MAX_TASKS_PER_MODEL}"
        )
        raise AssertionError(msg)
    return errors


def main() -> None:
    """Entry point for the prep job: read env, validate, annotate + exit non-zero on failure."""
    try:
        models = json.loads(os.environ["MODEL_MATRIX"]).get("include", [])
        n_shards = parse_positive("n_shards", os.environ.get("N_SHARDS"))
        concurrency = parse_positive("concurrency", os.environ.get("CONCURRENCY"))
        rollouts = parse_positive("rollouts", os.environ.get("ROLLOUTS"))
    except LimitError as exc:
        print(f"::error::{exc}")  # noqa: T201
        sys.exit(1)

    errors = validate_limits(models, n_shards, concurrency, rollouts)
    for err in errors:
        print(f"::error::{err}")  # noqa: T201
    if errors:
        sys.exit(1)

    shard_parallel = derive_shard_parallel(concurrency, rollouts, n_shards)
    print(  # noqa: T201
        f"OK: single model ({models[0].get('model')}), n_shards={n_shards} "
        f"(<= {shard_matrix.MAX_SHARDS}), concurrency={concurrency} "
        f"(<= {MAX_CONCURRENCY}), derived shard_parallel={shard_parallel}"
    )
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:  # noqa: PTH123
            f.write(f"shard_parallel={shard_parallel}\n")
    else:
        print(f"shard_parallel={shard_parallel}")  # noqa: T201


if __name__ == "__main__":
    main()
