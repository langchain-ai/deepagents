"""Single-model + resource-limit gate for the Harbor evals workflow.

The Harbor workflow (`.github/workflows/harbor.yml`) evaluates exactly ONE model
per dispatch. This runs in the `prep` job, before the matrix fans out, to fail
fast when:

* the resolved model set is not a single model (a model group like `all`/`set0`
  or a comma-separated `models_override` resolves to more than one), or
* the per-run limits are exceeded: `n_shards <= 10`, `concurrency <= 4`. With a
  single model their product bounds concurrent sandboxes to `10 * 4 = 40`; the
  caps are checked independently (there is no separate product check).

Mirrors `.github/scripts/shard_matrix.py` (import-by-path, stdlib only) so it is
exercised by `test_validate_harbor_limits.py` under CI's
`pytest .github/scripts/test_*.py`. Inputs arrive via env (never `${{ }}` shell
interpolation) and are int-parsed here.
"""

from __future__ import annotations

import json
import os
import sys

MAX_SHARDS = 10
MAX_CONCURRENCY = 4


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


def validate_limits(models: list, n_shards: int, concurrency: int) -> list[str]:
    """Return a list of human-readable violations (empty when the run is valid)."""
    errors: list[str] = []
    if len(models) != 1:
        names = [m.get("model") for m in models]
        errors.append(
            f"this workflow evaluates a single model, but the selection resolved "
            f"to {len(models)}: {names}. Pick exactly one model — model groups "
            "(all/set0/frontier/...) and comma-separated overrides are not accepted."
        )
    if n_shards > MAX_SHARDS:
        errors.append(f"n_shards={n_shards} exceeds the cap of {MAX_SHARDS}")
    if concurrency > MAX_CONCURRENCY:
        errors.append(f"concurrency={concurrency} exceeds the cap of {MAX_CONCURRENCY}")
    return errors


def main() -> None:
    """Entry point for the prep job: read env, validate, annotate + exit non-zero on failure."""
    try:
        models = json.loads(os.environ["MODEL_MATRIX"]).get("include", [])
        n_shards = parse_positive("n_shards", os.environ.get("N_SHARDS"))
        concurrency = parse_positive("concurrency", os.environ.get("CONCURRENCY"))
    except LimitError as exc:
        print(f"::error::{exc}")  # noqa: T201
        sys.exit(1)

    errors = validate_limits(models, n_shards, concurrency)
    for err in errors:
        print(f"::error::{err}")  # noqa: T201
    if errors:
        sys.exit(1)
    print(  # noqa: T201
        f"OK: single model ({models[0].get('model')}), n_shards={n_shards} "
        f"(<= {MAX_SHARDS}), concurrency={concurrency} (<= {MAX_CONCURRENCY})"
    )


if __name__ == "__main__":
    main()
